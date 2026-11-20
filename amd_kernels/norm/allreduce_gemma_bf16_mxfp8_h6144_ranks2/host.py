# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The two-rank communicator: one IPC-mapped staging buffer per rank, the role call counts and the row tiers."""
import contextlib
import ctypes
import functools
import struct

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from . import common as C
from .kernel import compile_allreduce_gemma_bf16_mxfp8

# owner form from here: 21 rows all-row 10.20 vs owner 10.48 us, 28: 11.26 vs 10.98 (benchmark_allreduce_ranks2.py)
OWNER_MIN_ROWS = 24


@functools.cache
def _kernel(rpb, tiled, quantized, owner):
    return compile_allreduce_gemma_bf16_mxfp8(rpb, tiled, quantized, owner)


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``), later calls run the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _f32_bits(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


def _release(ptr, peer):
    hip = ops._hip()
    fn = hip.hipIpcCloseMemHandle if peer else hip.hipFree
    fn.argtypes = [ctypes.c_void_p]
    rc = fn(ctypes.c_void_p(ptr))
    assert rc == 0, f"HIP buffer release failed: {rc}"


def exchange(obj, rank, group) -> list:
    """Every rank's ``obj`` (broadcasts on the CPU group: all_gather_object fails on gloo under inference mode)."""
    ranks = sorted(dist.get_process_group_ranks(group=group))
    out = [[None] for _ in ranks]
    out[rank][0] = obj
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


def owned_rows(rows, rank):
    """[start, stop) of the rows whose sum, residual and norm ``rank`` owns."""
    q = (rows + 1) // 2
    return min(rank * q, rows), min((rank + 1) * q, rows)


def config(rows, owner):
    """(rows per batch, stagers, reducers, receivers) of a call, measured (benchmark_allreduce_ranks2.py --sweep)."""
    if owner:
        q = (rows + 1) // 2
        rpb = 1 if q <= 32 else 2 if q <= 256 else 4 if q <= 512 else 8
        nb = 2 * -(-q // rpb)
        nred = min(nb, 128 if rows <= 512 else 224)
        return rpb, min(nb, 128), nred, nred
    if rows <= 256:
        rpb = 1 if rows <= 64 else 2
        nb = -(-rows // rpb)
        return rpb, nb, nb, 0
    rpb, nstage, nred = (2, 32, 128) if rows <= 512 else (4, 224, 224) if rows <= 1024 else (8, 32, 128)
    nb = -(-rows // rpb)
    return rpb, min(nb, nstage), min(nb, nred), 0


def warm_calls():
    """(rows, owned_residual) of the first call of every compiled variant (rows per batch, form)."""
    seen, out = set(), []
    for owner in (False, True):
        for rows in range(1, C.MAX_ROWS + 1):
            form = owner and rows >= OWNER_MIN_ROWS
            key = (config(rows, form)[0], form)
            if key not in seen:
                seen.add(key)
                out.append((rows, owner))
    return out


class Bf16ArNormComm:
    """One communicator per stream, created collectively on two ranks.

    Eager and graph calls run the same kernel: each rank stages its partial rows in its own buffer and reads the
    peer's from there, so peer inputs are never read in place (``capture()`` is a no-op matching the four-rank API).
    All ranks call the same operations in the same order; close only after graph work completes."""

    config = None  # benchmarks may override ``config``

    def __init__(self, rank, group, device):
        if dist.get_world_size(group) != C.WORLD:
            raise ValueError("two ranks required")
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.current_device() != device.index or dist.get_rank(group) != rank:
            raise ValueError("select this rank device before creating the communicator")
        self.rank, self.group, self.device = rank, group, device
        ptr, handle = ops.allocate_shared_buffer_and_handle(C.BUF_BYTES)
        every = exchange(handle, rank, group)
        self.mine, self.peer = ptr, ops.open_mem_handle(every[1 - rank])
        self.counts = torch.zeros(3 * C.COUNT_WORDS, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    @contextlib.contextmanager
    def capture(self):
        yield

    def _launch(self, x, residual, weight, eps, tile, quantized, owner=False):
        if self.mine is None:
            raise RuntimeError("communicator is closed")
        m = x.shape[0]
        if not (0 < m <= C.MAX_ROWS and tuple(x.shape) == (m, C.H)):
            raise ValueError("expected 1..2048 rows of hidden 6144")
        if x.device != self.device or residual.device != self.device or weight.device != self.device:
            raise ValueError("all tensors must be on the communicator device")
        if x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("partial must be contiguous bf16")
        if residual.shape != x.shape or residual.dtype != x.dtype or residual.stride(1) != 1 or residual.stride(0) % 8:
            raise ValueError("residual must be bf16 with a row stride divisible by eight")
        if tuple(weight.shape) != (C.H,) or weight.dtype != x.dtype or not weight.is_contiguous():
            raise ValueError("weight must be contiguous bf16 [6144]")
        if tile < 0 or tile % 16:
            raise ValueError("code tile rows must be zero or a multiple of 16")
        padded = (m + max(tile, 1) - 1) // max(tile, 1) * max(tile, 1)
        out = torch.empty((padded if quantized else m, C.H), dtype=torch.float8_e4m3fn if quantized else torch.bfloat16,
                          device=self.device)
        scales = torch.empty(((padded + 31) // 32 * 32, C.H // 32), dtype=torch.uint8,
                             device=self.device) if quantized else None
        res = torch.empty_like(x)
        owner = owner and m >= OWNER_MIN_ROWS
        rpb, nstage, nred, nrecv = (self.config or config)(m, owner)
        assert 0 < nstage < 1024 and 0 < nred < 1024 and 0 <= nrecv < 1024
        _run_compiled(_kernel(rpb, tile > 0, quantized, owner), m, self.rank, nstage | nred << 10 | nrecv << 20,
                      residual.stride(0), x.data_ptr(), self.mine, self.peer, self.counts.data_ptr(),
                      residual.data_ptr(), weight.data_ptr(), out.data_ptr(), scales.data_ptr() if quantized else 0,
                      res.data_ptr(), _f32_bits(eps), torch.cuda.current_stream())
        return (out, scales, res) if quantized else (out, res)

    def ar_add_rmsnorm(self, x, residual, weight, eps=1e-6, owned_residual=False):
        """``owned_residual``: from ``OWNER_MIN_ROWS`` rows on, only this rank's ``owned_rows`` of ``residual`` are read
        and of the residual output written; the owner norms its rows and stages the bf16 result for the peer. Fewer rows
        run the all-row form. So pass a full residual or the residual output of an ``owned_residual`` call of the same
        row count. Outputs equal the default form's bit for bit."""
        return self._launch(x, residual, weight, eps, 0, False, owned_residual)

    def ar_add_rmsnorm_mxfp8(self, x, residual, weight, eps=1e-6, code_tile_rows=16, owned_residual=False):
        """``owned_residual``: as in ``ar_add_rmsnorm``; the owner stages its codes and scale bytes (about half the
        bytes of a bf16 row)."""
        return self._launch(x, residual, weight, eps, code_tile_rows, True, owned_residual)

    def owned_rows(self, rows):
        return owned_rows(rows, self.rank)

    def warm_up(self, weight, eps=1e-6):
        """Compile every row tier, both code layouts, bf16 output and both residual forms collectively."""
        for rows, owner in warm_calls():
            x = torch.zeros((rows, C.H), dtype=torch.bfloat16, device=self.device)
            self.ar_add_rmsnorm(x, x, weight, eps, owner)
            for tile in (0, 16):
                self.ar_add_rmsnorm_mxfp8(x, x, weight, eps, tile, owner)
        torch.cuda.synchronize(self.device)

    def close(self):
        if self.mine is None:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        _release(self.peer, True)
        _release(self.mine, False)
        self.mine = self.peer = None
