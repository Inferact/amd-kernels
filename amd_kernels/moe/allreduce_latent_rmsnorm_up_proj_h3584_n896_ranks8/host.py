# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Communicator, row-count dispatch and entry of the eight-rank latent all-reduce + RMSNorm + up projection."""
import ctypes
import functools

import torch
import torch.distributed as dist

from amd_kernels import warm_up
from amd_kernels.common import hip_runtime as ops

from . import tail_kernel as T
from .ipc import alloc_handle, close_handle, exchange, f32_bits, open_handle, run_compiled
from .norm_math import K, N, WORLD

NAME = "moe/allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8"
MAX_ROWS = T.MAX_ROWS


@functools.cache
def _kernel(mode, mpg, rb, ts=False, groups=T.GROUPS, nsub=T.NSUB, slots=False):
    return T.compile_tail(mode, mpg, rb, ts, groups, nsub, slots)


def replicated_partial_buffer(rows: int, device, hidden: int = 7168):
    """Persistent ``[rows, hidden]`` bf16 partial buffer of the replicated-shared layout: the kernel writes only this
    rank's 896 columns, the rest stays -0.0 (bit pattern 0x8000), so the final all-reduce adds seven -0.0 to every
    value and returns it bit for bit (x + -0.0 = x for every x, +0.0 would turn a real -0.0 into +0.0)."""
    return torch.full((rows, hidden), -0.0, dtype=torch.bfloat16, device=device)


def _release(ptr, peer):
    hip = ops._hip()
    fn = hip.hipIpcCloseMemHandle if peer else hip.hipFree
    fn.argtypes = [ctypes.c_void_p]
    rc = fn(ctypes.c_void_p(ptr))
    assert rc == 0, f"HIP buffer release failed: {rc}"


def _mode(comm, partial, norm_weight, up_weight, shared, out=None, eps=1e-6):
    """What a call fixes about its kernels besides the row count (``amd_kernels.warm_up``)."""
    return ()


def _sizes(mode):
    """Row counts that reach every (rows per tile group, rows per all-reduce workgroup) variant."""
    return (64, 96, 128, 192, 256)


def _inputs(mode, n, comm, partial, norm_weight, up_weight, shared, out=None, eps=1e-6):
    dev = partial.device
    z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    return (comm, z(n, K), norm_weight, up_weight, z(n, N), None, eps), {}


@warm_up.entry(NAME, mode=_mode, sizes=_sizes, inputs=_inputs)
def allreduce_latent_rmsnorm_up_proj(comm, partial, norm_weight, up_weight, shared, out=None, eps=1e-6):
    """``out = shared + rmsnorm(allreduce(partial)) @ up_weight^T`` (see reference.py); ``out`` defaults to ``shared``
    (in place). Returns ``out``."""
    return comm._launch(partial, norm_weight, up_weight, shared, shared if out is None else out, eps)


NAME_SLOTS = NAME + "/slots"


def _slots_sizes(mode):
    return (16, 64)  # nsub 1 (up to 32 rows) and nsub 2 (33..64)


def _slots_inputs(mode, n, comm, slots, reverse_sorted, norm_weight, up_weight, shared, out=None, eps=1e-6):
    dev = slots.device
    z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    return (comm, z(n * T.SLOTS, K), torch.zeros(n * T.SLOTS, dtype=torch.int32, device=dev), norm_weight, up_weight,
            z(n, N), None, eps), {}


def _slots_mode(comm, slots, reverse_sorted, norm_weight, up_weight, shared, out=None, eps=1e-6):
    return ()


@warm_up.entry(NAME_SLOTS, mode=_slots_mode, sizes=_slots_sizes, inputs=_slots_inputs)
def allreduce_slots_rmsnorm_up_proj(comm, slots, reverse_sorted, norm_weight, up_weight, shared, out=None, eps=1e-6):
    """The same call with the routed experts' 16-slot sum folded in (1..64 rows): ``slots`` is the experts' slot buffer
    (bf16 [sorted rows, 3584], router weights applied), ``reverse_sorted`` (int32 [rows * 16], token t slot k at
    ``16 t + k``) names its rows; the partial of this rank is ``bf16(sum_k slots[reverse_sorted[16 t + k]])`` (fp32,
    slot order), everything else as in ``allreduce_latent_rmsnorm_up_proj``."""
    return comm._launch(slots, norm_weight, up_weight, shared, shared if out is None else out, eps,
                        reverse_sorted=reverse_sorted)


class LatentTailComm:
    """One communicator per stream, created collectively on eight ranks (``device``: this rank's GPU).

    All ranks make the same calls in the same order; close only after the work completes. ``fused=False`` runs the
    all-reduce and the up projection as two launches (the same kernels in their single-role modes).
    """

    def __init__(self, rank, group, device, fused=True):
        if dist.get_world_size(group) != WORLD:
            raise ValueError("eight ranks required")
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.current_device() != device.index or dist.get_rank(group) != rank:
            raise ValueError("select this rank device before creating the communicator")
        self.rank, self.group, self.device, self.fused = rank, group, device, fused
        ptr, handle = ops.allocate_shared_buffer_and_handle(T.BUF_BYTES)
        peers = exchange(handle, rank, group)
        self.bufs = [ptr if r == rank else ops.open_mem_handle(peers[r]) for r in range(WORLD)]
        self.lat_buf = torch.zeros(T.LAT_ALLOC // 2, dtype=torch.bfloat16, device=device)
        self.lat = self.lat_buf[:MAX_ROWS * K].view(MAX_ROWS, K)
        lat_peers = exchange(alloc_handle(self.lat_buf.data_ptr()), rank, group)
        self.lat_opened = [open_handle(raw) for r, (raw, _) in enumerate(lat_peers) if r != rank]
        opened = iter(self.lat_opened)
        lats = [self.lat_buf.data_ptr() if r == rank else next(opened) + lat_peers[r][1] for r in range(WORLD)]
        self.buf_table = torch.tensor(self.bufs + lats, dtype=torch.int64, device=device)
        self.counts = torch.zeros(T.CNT_WORDS, dtype=torch.int32, device=device)
        self.groups_override = 0
        self.rb_override = 0  # rows per all-reduce workgroup (2, 4 or 8) for a sweep; 0: ``tail_kernel.rb_of``
        self.ts = None  # a [workgroup, wave, 8] int32 tensor of phase stamps (a profiling build), set by the caller
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def _launch(self, x, norm_weight, w, shared, out, eps, modes=None, reverse_sorted=None):
        """``x``: the [rows, 3584] partial, or with ``reverse_sorted`` ([rows * 16] int32) the routed experts' slot
        buffer [sorted rows, 3584] whose rows ``reverse_sorted[16 t + k]`` are the 16 slots of token t."""
        if not self.bufs:
            raise RuntimeError("communicator is closed")
        slots = reverse_sorted is not None
        m = reverse_sorted.numel() // T.SLOTS if slots else x.shape[0]
        dev = self.device
        if not (0 < m <= (T.MAX_SLOT_ROWS if slots else MAX_ROWS)) or (not slots and tuple(x.shape) != (m, K)):
            raise ValueError(f"expected 1..{T.MAX_SLOT_ROWS if slots else MAX_ROWS} rows of width {K}")
        if slots and (reverse_sorted.dtype != torch.int32 or reverse_sorted.device != dev
                      or not reverse_sorted.is_contiguous() or x.ndim != 2 or x.shape[1] != K
                      or not x.is_contiguous() or reverse_sorted.numel() != m * T.SLOTS):
            raise ValueError("slots: contiguous bf16 [sorted rows, 3584] and contiguous int32 reverse_sorted [rows * 16]")
        for t in (x, norm_weight, w, shared, out):
            if t.dtype != torch.bfloat16 or t.device != dev or t.stride(-1) != 1:
                raise ValueError("operands must be bf16 on the communicator device with unit inner stride")
        if (x.stride(0) % 8 and not slots) or shared.stride(0) % 4 or out.stride(0) % 4:
            raise ValueError("row strides must keep 16 B (latent) and 8 B (shard) alignment")
        if tuple(norm_weight.shape) != (K,) or tuple(w.shape) != (N, K) or not w.is_contiguous():
            raise ValueError(f"norm weight [{K}] and contiguous up shard [{N}, {K}] required")
        if tuple(shared.shape) != (m, N) or tuple(out.shape) != (m, N):
            raise ValueError(f"shared and out must be [{m}, {N}]")
        stream = torch.cuda.current_stream()
        nsub = T.nsub_of((m + 15) // 16, slots)
        groups = self.groups_override or T.groups_of((m + 15) // 16)
        mpg = T.mpg_of((m + 15) // 16, groups)
        rb = self.rb_override or (T.rb_of_slots(m) if slots else T.rb_of(m))
        ts = 0 if self.ts is None else self.ts.data_ptr()
        args = (m, (m + rb - 1) // rb, self.rank, self.buf_table.data_ptr(), self.counts.data_ptr(), x.data_ptr(),
                x.stride(0), norm_weight.data_ptr(), w.data_ptr(), shared.data_ptr(), shared.stride(0),
                out.data_ptr(), out.stride(0), f32_bits(eps), 0, ts,
                reverse_sorted.data_ptr() if slots else 0, stream)
        for mode in modes or (("fused",) if self.fused else ("ar", "up")):
            run_compiled(_kernel(mode, mpg, rb, bool(ts), groups, nsub, slots), *args)
        return out

    def close(self):
        if not self.bufs:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for r, ptr in enumerate(self.bufs):
            _release(ptr, r != self.rank)
        for ptr in self.lat_opened:
            close_handle(ptr)
        self.bufs = []
        self.lat_opened = []
