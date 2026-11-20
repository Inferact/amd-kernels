# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Communicators of the two-rank one-shot all-reduce + add + Gemma RMSNorm kernels: ``SharedBuffers`` (the IPC-mapped
buffer pair and the call counts of one stream), ``StagedArNormComm`` (the attention tail: one bf16 partial per rank)
and ``SlotsArNormComm`` (the MoE tail: four weighted slots + the shared expert per rank)."""

import contextlib
import functools

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from ..add_rmsnorm.host import _f32_bits, _out_or_empty, _run_compiled, code_tile_rows_for
from . import common as C
from . import oneshot as O

H = C.H
WORLD = 2
MAX_ROWS = C.MAX_ROWS - 1  # rows of one launch
CHUNK_ROWS = 3840  # larger calls launch per chunk: a multiple of every code tile (<= 256) and of the 32-row scales
LIMIT_ROWS = 1 << 16


@functools.cache
def _kernel(out: str, slots: bool, rpb: int, coded: bool):
    return O.compile_oneshot_ar_add_rmsnorm(out, slots, rpb, coded)


def exchange(obj, rank: int, group) -> list:
    """Every rank's ``obj`` (broadcasts on the CPU group: all_gather_object fails on gloo under inference mode)."""
    ranks = sorted(dist.get_process_group_ranks(group=group))
    out = [[None] for _ in ranks]
    out[rank][0] = obj
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


def config(rows: int, quant: bool, slots: bool) -> tuple[int, int, int, bool]:
    """(rows per batch, stagers, reducers, coded) of a call, measured for every output (benchmark_ar_add_rmsnorm_ranks2
    .py --sweep): the lossless code wins from 49-64 rows (48 rows: bf16 coded 14.82 / raw 15.25 us, MXFP8 16.54 /
    16.05)."""
    if rows <= 48:
        return 1, rows, rows, False
    rpb = 1 if rows <= 1024 else 2 if rows <= 2048 else 4
    nb = (rows + rpb - 1) // rpb
    nred = 32 if not quant and rows <= 96 else 64 if rows <= 256 else 128
    return rpb, min(nb, 32), min(nb, nred), True


def warm_rows(quant: bool, slots: bool) -> list[int]:
    """The first row count of every compiled variant (rows per batch, coded) of ``config``."""
    seen, out = set(), []
    for rows in range(1, MAX_ROWS + 1):
        rpb, _, _, coded = config(rows, quant, slots)
        if (rpb, coded) not in seen:
            seen.add((rpb, coded))
            out.append(rows)
    return out


def _overlaps(a: torch.Tensor, *others: torch.Tensor) -> bool:
    lo, hi = a.data_ptr(), a.data_ptr() + a.numel() * a.element_size()
    return any(t is not None and t.data_ptr() < hi and lo < t.data_ptr() + t.numel() * t.element_size()
               for t in others)


class SharedBuffers:
    """One uncached buffer per rank (its staged rows and the flags the peer stores), mapped by the peer, and the per-row
    call counts: a collective on the 2-rank CPU group, once per rank and stream; the communicators of one stream may
    share it (calls in the same order on both ranks)."""

    def __init__(self, rank: int, group, device: torch.device):
        if dist.get_world_size(group) != WORLD:
            raise ValueError("two ranks required")
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.rank, self.group, self.device = rank, group, device
        ptr, handle = ops.allocate_shared_buffer_and_handle(C.BUF_BYTES)
        every = exchange(handle, rank, group)
        self.mine = ptr
        self.peer = ops.open_mem_handle(every[1 - rank])
        self.counts = torch.zeros(2 * O.COUNT_WORDS, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    config = None  # benchmarks may override ``config``

    def launch(self, out_kind, slots, rows, sx, sr, x, sh, residual, weight, out, scales, res_out, eps):
        """One launch per ``CHUNK_ROWS`` rows above ``MAX_ROWS`` (row-wise op, the chunks' outputs are contiguous)."""
        chunk = rows if rows <= MAX_ROWS else CHUNK_ROWS
        xrow = (4 * C.ROWB) if slots else sx * 2
        orow = H if out_kind != "bf16" else C.ROWB
        for r0 in range(0, rows, chunk):
            n = min(chunk, rows - r0)
            rpb, nstage, nred, coded = (self.config or config)(n, out_kind != "bf16", slots)
            assert 0 < nstage < 1 << 15 and 0 < nred < 1 << 15
            _run_compiled(
                _kernel(out_kind, slots, rpb, coded), n, nstage | (nred << 16), sx, sr, x + r0 * xrow,
                sh + r0 * C.ROWB if sh else 0, self.mine, self.peer, residual + r0 * sr * 2, weight, out + r0 * orow,
                scales + r0 * (H // 32) if scales else 0, res_out + r0 * C.ROWB, self.counts.data_ptr(),
                _f32_bits(eps), torch.cuda.current_stream(),
            )


def _check_rows(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, rows: int):
    assert 0 < rows <= LIMIT_ROWS, rows
    assert residual.shape == (rows, H) and residual.dtype == torch.bfloat16
    assert residual.stride(1) == 1 and residual.stride(0) % 8 == 0
    assert weight.shape == (H,) and weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert x.device == residual.device == weight.device


def _mxfp8_outputs(rows, code_tile_rows, q_out, scale_out, res_out, device):
    assert code_tile_rows >= 0 and code_tile_rows % 16 == 0
    tiled = (rows + code_tile_rows - 1) // code_tile_rows * code_tile_rows if code_tile_rows else rows
    q = _out_or_empty(q_out, (tiled, H), torch.float8_e4m3fn, device)
    scale = _out_or_empty(scale_out, ((tiled + 31) // 32 * 32, H // 32), torch.uint8, device)
    res_out = _out_or_empty(res_out, (rows, H), torch.bfloat16, device)
    return q, scale, res_out


class StagedArNormComm:
    """The attention tail at two ranks: all-reduce of one bf16 [rows, H] partial per rank + residual add + Gemma
    RMSNorm, bf16 or MXFP8 out; eager and graph calls alike (``capture()`` is a no-op matching the four-rank API).
    One communicator per stream; ``buffers``: share another communicator's (the same stream)."""

    def __init__(self, rank: int, group, device: torch.device, buffers: SharedBuffers | None = None):
        self.buffers = buffers or SharedBuffers(rank, group, device)
        self.rank, self.group, self.device = rank, group, self.buffers.device

    @contextlib.contextmanager
    def capture(self):
        yield

    def supports(self, rows: int, mxfp8: bool = False) -> bool:
        """All-reduce row counts supported for either output, eager and graphs."""
        return 0 < rows <= LIMIT_ROWS

    def accepts(self, x: torch.Tensor, residual: torch.Tensor) -> bool:
        """Whether ``ar_add_rmsnorm`` / ``ar_add_rmsnorm_mxfp8`` take these tensors: bf16 [rows, H] with unit column
        stride and 16-byte rows."""
        return (
            x.dim() == 2
            and x.shape[1] == H
            and self.supports(x.shape[0])
            and residual.shape == x.shape
            and x.dtype == residual.dtype == torch.bfloat16
            and x.stride(1) == 1
            and residual.stride(1) == 1
            and x.stride(0) % 8 == 0
            and residual.stride(0) % 8 == 0
        )

    def _input(self, x, *outputs):
        # a reducer may write a row before this rank's stager of it reads it: outputs must not overlap the input
        return x.clone() if _overlaps(x, *outputs) else x

    def ar_add_rmsnorm(self, x, residual, weight, eps, out=None, res_out=None):
        """``add_rmsnorm.gemma_add_rmsnorm`` of all-reduce(x) + residual: (bf16 out, residual out); every rank passes
        its own bf16 [rows, H] partial (``reference.allreduce_rows_reference``)."""
        rows = x.shape[0]
        assert self.accepts(x, residual), (tuple(x.shape), x.stride(), residual.stride())
        _check_rows(x, residual, weight, rows)
        out = _out_or_empty(out, (rows, H), torch.bfloat16, self.device)
        res_out = _out_or_empty(res_out, (rows, H), torch.bfloat16, self.device)
        x = self._input(x, out, res_out)
        self.buffers.launch("bf16", False, rows, x.stride(0), residual.stride(0), x.data_ptr(), 0,
                            residual.data_ptr(), weight.data_ptr(), out.data_ptr(), 0, res_out.data_ptr(), eps)
        return out, res_out

    def ar_add_rmsnorm_mxfp8(self, x, residual, weight, eps, code_tile_rows=None, q_out=None, scale_out=None,
                             res_out=None, *, gemm_weight=None):
        """``add_rmsnorm.gemma_add_rmsnorm_mxfp8`` of all-reduce(x) + residual; every rank passes its own bf16
        [rows, H] partial. Returns (codes in the (16, 16) layout padded to ``code_tile_rows`` when > 0, row-major at
        0, the tiled norm's arithmetic in both; AITER-layout scales; residual out). ``gemm_weight`` instead of
        ``code_tile_rows``: the layout that GEMM reads at this row count."""
        rows = x.shape[0]
        if gemm_weight is not None:
            code_tile_rows = code_tile_rows_for(rows, gemm_weight)
        assert code_tile_rows is not None
        assert self.accepts(x, residual)
        _check_rows(x, residual, weight, rows)
        assert rows <= MAX_ROWS or CHUNK_ROWS % max(code_tile_rows, 1) == 0, code_tile_rows
        q, scale, res_out = _mxfp8_outputs(rows, code_tile_rows, q_out, scale_out, res_out, self.device)
        x = self._input(x, q, scale, res_out)
        self.buffers.launch("tiled" if code_tile_rows else "rowmajor", False, rows, x.stride(0), residual.stride(0),
                            x.data_ptr(), 0, residual.data_ptr(), weight.data_ptr(), q.data_ptr(), scale.data_ptr(),
                            res_out.data_ptr(), eps)
        return q, scale, res_out

    def warm_up(self, weight: torch.Tensor, eps: float, code_tile_rows: int = 16) -> None:
        """Compile every variant before serving, both code layouts (a collective: every rank calls it)."""
        for rows in sorted(set(warm_rows(False, False)) | set(warm_rows(True, False))):
            x = torch.zeros((rows, H), dtype=torch.bfloat16, device=self.device)
            r = torch.zeros_like(x)
            self.ar_add_rmsnorm(x, r, weight, eps)
            for tile in (0, code_tile_rows or 16):
                self.ar_add_rmsnorm_mxfp8(x, r, weight, eps, tile)
        torch.cuda.synchronize(self.device)


class SlotsArNormComm:
    """The MoE tail at two ranks: all-reduce of each rank's ``bf16((((s0 + s1) + s2) + s3) + shared)`` + residual add +
    Gemma RMSNorm + MXFP8 quant; eager and graph calls alike. One communicator per stream; ``buffers``: share another
    communicator's (the same stream)."""

    def __init__(self, rank: int, group, device: torch.device, buffers: SharedBuffers | None = None):
        self.buffers = buffers or SharedBuffers(rank, group, device)
        self.rank, self.group, self.device = rank, group, self.buffers.device

    def supports(self, rows: int) -> bool:
        """Row counts ``slots4_ar_add_rmsnorm_mxfp8`` takes, eager and captured."""
        return 0 < rows <= LIMIT_ROWS

    def slots4_ar_add_rmsnorm_mxfp8(self, slots, shared, residual, weight, eps, code_tile_rows=None, q_out=None,
                                    scale_out=None, res_out=None, *, gemm_weight=None):
        """``add_rmsnorm.gemma_add_rmsnorm_mxfp8`` of all-reduce(slot 0 + .. + slot 3 + shared) + residual. Every rank
        passes its own contiguous bf16 [rows, 4, H] slots and [rows, H] shared output. Returns (codes in the (16, 16)
        layout when ``code_tile_rows`` > 0, row-major at 0; AITER-layout scales; residual out); ``gemm_weight`` instead
        of ``code_tile_rows``: the layout that GEMM reads at this row count."""
        rows = slots.shape[0]
        if gemm_weight is not None:
            code_tile_rows = code_tile_rows_for(rows, gemm_weight)
        assert code_tile_rows is not None
        assert slots.shape == (rows, 4, H) and slots.dtype == torch.bfloat16 and slots.is_contiguous()
        assert shared.shape == (rows, H) and shared.dtype == torch.bfloat16 and shared.is_contiguous()
        _check_rows(slots, residual, weight, rows)
        assert rows <= MAX_ROWS or CHUNK_ROWS % max(code_tile_rows, 1) == 0, code_tile_rows
        q, scale, res_out = _mxfp8_outputs(rows, code_tile_rows, q_out, scale_out, res_out, self.device)
        if _overlaps(slots, q, scale, res_out) or _overlaps(shared, q, scale, res_out):
            slots, shared = slots.clone(), shared.clone()
        self.buffers.launch("tiled" if code_tile_rows else "rowmajor", True, rows, 0, residual.stride(0),
                            slots.data_ptr(), shared.data_ptr(), residual.data_ptr(), weight.data_ptr(), q.data_ptr(),
                            scale.data_ptr(), res_out.data_ptr(), eps)
        return q, scale, res_out

    def warm_up(self, weight: torch.Tensor, eps: float, code_tile_rows: int = 16) -> None:
        """Compile every variant before serving, both code layouts (a collective: every rank calls it)."""
        for rows in warm_rows(True, True):
            slots = torch.zeros((rows, 4, H), dtype=torch.bfloat16, device=self.device)
            row = torch.zeros((rows, H), dtype=torch.bfloat16, device=self.device)
            for tile in (0, code_tile_rows or 16):
                self.slots4_ar_add_rmsnorm_mxfp8(slots, row, row.clone(), weight, eps, tile)
        torch.cuda.synchronize(self.device)
