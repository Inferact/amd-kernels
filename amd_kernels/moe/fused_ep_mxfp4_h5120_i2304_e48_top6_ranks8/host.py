# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Expert-parallel host: the per-rank pool and control words mapped on every peer, the token-count dispatch and the
launch of the fused kernel (``kernel.compile_fused``)."""
import ctypes
import functools
from dataclasses import dataclass

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from amd_kernels.common.tensor_shim import _run_compiled

from .kernel import (CTL_BYTES, E, H, I, NE_MAX, READY_STRIDE, TM, TOPK, compile_fused, max_blocks, pool_layout)

MAX_TOKENS = TM
ND = 128  # dispatch workgroups per rank
# upper token bound -> (bm, GEMM1 K-waves, GEMM1 prefetch, GEMM1 16-column pairs per wave, GEMM1 ring stages,
# GEMM2 columns, GEMM2 prefetch, GEMM2 ring stages, cached weights for an expert with several blocks, waves per
# workgroup, persistent workgroups); tiles as moe/mxfp4_h5120_i2304_e48_top6 (its benchmark sweeps)
DISPATCH = (
    (2, (16, 4, 3, 2, 0, 256, 4, 0, False, 4, 256)),
    (8, (16, 4, 3, 4, 0, 256, 4, 0, False, 4, 256)),
    (48, (16, 4, 3, 2, 0, 256, 4, 0, False, 4, 512)),
    (160, (32, 1, 3, 4, 3, 256, 4, 3, True, 4, 256)),
    (448, (64, 1, 2, 2, 4, 256, 3, 6, True, 8, 256)),
    (MAX_TOKENS, (128, 1, 2, 2, 3, 256, 3, 3, True, 8, 256)),
)


def sized_tokens(m: int, rows_hint=None) -> int:
    """The token count the tiles are sized for: twice ``m`` up to 448 (the rank that owns hot experts receives more
    rows than its tokens' share), at least ``rows_hint``. Measured on 8 ranks (uniform / zipf / one rank's experts only):
    from 9 tokens the 32-row tier (bm 32, 160) is no slower than the 16-row one when routing is uniform and 25 % faster
    when it is not (T 16 hot 155 us against 206); from 49 tokens the 8-wave 64-row tier costs 3 % uniform and saves 6 %
    (zipf) and 25 % (hot)."""
    s = max(m, min(2 * m, DISPATCH[-2][0]), min(rows_hint or 0, MAX_TOKENS), 1)
    if m > 8:
        s = max(s, DISPATCH[3][0])
    if m > 48:
        s = max(s, DISPATCH[3][0] + 1)
    return s


def config_for(n_tokens: int):
    return next(cfg for bound, cfg in DISPATCH if n_tokens <= bound)


@dataclass
class ExpertWeights:
    """One layer's local experts (48, or ``epr`` <= 63 with redundant experts) in the kernel layouts (a pure
    permutation of the checkpoint bytes)."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor

    @classmethod
    def from_row_major(cls, w13, w13_scale, w2, w2_scale) -> "ExpertWeights":
        """From the checkpoint layouts of ``reference`` (uint8 views of the packed FP4 and e8m0 tensors); once, before
        graph capture. Same bytes as AITER's ``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4``."""
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        n = w13.shape[0]
        assert n <= NE_MAX and w13.shape[1:] == (2 * I, H // 2) and w2.shape[1:] == (H, I // 2)
        u8 = torch.uint8
        return cls(
            shuffle_weight_a16w4(w13.view(u8), 16, True).view(u8).contiguous(),
            shuffle_scale_a16w4(w13_scale.view(u8).reshape(-1, H // 32), n, True).view(u8).contiguous(),
            shuffle_weight_a16w4(w2.view(u8), 16, False).view(u8).contiguous(),
            shuffle_scale_a16w4(w2_scale.view(u8).reshape(-1, I // 32), n, False).view(u8).contiguous(),
        )


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = ops.load_hip_runtime()
    lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
    return lib


def _alloc_pool(nbytes: int, device: torch.device) -> tuple[int, bytes]:
    """A hipMalloc pool (coarse-grained, never freed) -> (pointer, IPC handle bytes)."""
    hip = _hip()
    ptr = ctypes.c_void_p()
    with torch.cuda.device(device):
        rc = hip.hipMalloc(ctypes.byref(ptr), nbytes)
    assert rc == 0, f"hipMalloc({nbytes}) failed: {rc}"
    handle = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    return ptr.value, ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))


def _open_pool(handle: bytes, device: torch.device) -> int:
    hip = _hip()
    ptr = ctypes.c_void_p()
    h = _IpcHandle()
    ctypes.memmove(ctypes.addressof(h), handle, len(handle))
    with torch.cuda.device(device):
        rc = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), h, 1)
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value


def _exchange(obj, rank: int, world: int, group) -> list:
    out = [[None] for _ in range(world)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class ExpertParallel:
    """One rank's pool, control words and scratch (a collective: every rank of the group creates it, before graph
    capture). One per process serves every layer; two calls must not run concurrently on one device. Rank ``r`` owns
    the global experts ``48 r`` .. ``48 r + 47``."""

    def __init__(self, rank: int, world: int, group, device: torch.device, epr: int = E):
        assert world in (2, 4, 8) and 1 <= epr <= NE_MAX
        self.rank, self.world, self.device, self.epr, self.group = rank, world, device, epr, group
        self.layout = pool_layout(world, epr)
        ptr, handle = _alloc_pool(self.layout["total"], device)
        ctl_ptr, ctl_handle = ops.allocate_shared_buffer_and_handle(CTL_BYTES)
        with torch.cuda.device(device):
            rc = _hip().hipMemset(ctypes.c_void_p(ctl_ptr), 0, CTL_BYTES)
        assert rc == 0, f"hipMemset(ctl) failed: {rc}"
        peers = _exchange((handle, ctl_handle), rank, world, group)
        pools = [ptr if r == rank else _open_pool(h, device) for r, (h, _) in enumerate(peers)]
        ctls = [ctl_ptr if r == rank else ops.open_mem_handle(h) for r, (_, h) in enumerate(peers)]
        self.pool_addrs = pools + [0] * (8 - world)
        self.ctl_addrs = ctls + [0] * (8 - world)
        self.loc = torch.zeros(24576, dtype=torch.int32, device=device)
        self.ready = torch.zeros((max_blocks(world, 16, epr) + 1) * READY_STRIDE // 4, dtype=torch.int32, device=device)
        rows = max_blocks(world, 128, epr) * 128
        self.q = torch.empty(rows * I, dtype=torch.uint8, device=device)
        self.s = torch.empty(rows * (I // 32), dtype=torch.uint8, device=device)
        self._rows = rows
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def warm_up(self, w: ExpertWeights) -> None:
        """A collective before graph capture: runs every variant (token-count tier x MXFP8 / bf16 rows) once on batches
        nobody routes, so nothing compiles while serving."""
        dev = self.device
        first = {}
        for m in range(1, MAX_TOKENS + 1):
            first.setdefault(config_for(sized_tokens(m)), m)
        for m in first.values():
            ids = torch.full((m, TOPK), -1, dtype=torch.int32, device=dev)
            tw = torch.zeros((m, TOPK), dtype=torch.float32, device=dev)
            self(torch.zeros((m, H), dtype=torch.bfloat16, device=dev), None, w, tw, ids)
            self(torch.zeros((m, H), dtype=torch.float8_e4m3fn, device=dev),
                 torch.zeros((m, H // 32), dtype=torch.uint8, device=dev), w, tw, ids)
        torch.cuda.synchronize(dev)
        dist.barrier(group=self.group)

    def __call__(self, x, x_scales, w: ExpertWeights, topk_weights, topk_ids, out=None, rows_hint=None):
        """The routed experts of this rank's ``m`` tokens over the experts of every rank: bf16 ``[m, 5120]``.
        ``x``: bf16 ``[m, 5120]`` (quantized in the dispatch) or MXFP8 e4m3 with its e8m0 ``x_scales`` ``[m, 160]``;
        ``topk_ids`` ``[m, 6]`` int32 physical expert ids, rank-major (an id outside [0, epr * world) is not routed),
        ``topk_weights`` FP32. Every rank calls it once per layer, idle ranks with ``m`` = 0 included. ``rows_hint``: a
        token count the tiles are sized for at least (a rank receives about ``6 x`` the mean count over the ranks)."""
        m = x.shape[0]
        assert 0 <= m <= MAX_TOKENS and x.shape[1] == H
        dev = x.device
        bm, kw, pf1, ni, nb1, tn2, pf2, nb2, hot, nwv, nw = config_for(sized_tokens(m, rows_hint))
        quant_in = x.dtype == torch.bfloat16
        if out is None:
            out = torch.empty((max(m, 1), H), dtype=torch.bfloat16, device=dev)
        tid = topk_ids.contiguous().int().view(-1)
        tw = topk_weights.contiguous().float().view(-1)
        xs = x_scales.view(torch.uint8).view(-1) if x_scales is not None else tid
        assert w.w13.numel() == self.epr * 2 * I * (H // 2), (w.w13.shape, self.epr)
        fn = compile_fused(world=self.world, epr=self.epr, nd=ND, nw=nw, bm=bm, kw=kw, pf1=pf1, ni=ni, nb1=nb1,
                           tn2=tn2, pf2=pf2, nb2=nb2, hot=hot, nwv=nwv, quant_in=quant_in)
        _run_compiled(fn, x.view(-1) if quant_in else x.view(torch.uint8).view(-1), xs, tid, tw, w.w13.view(-1),
                      w.w13_scale.view(-1), self.q, self.s, w.w2.view(-1), w.w2_scale.view(-1), out, self.loc,
                      self.ready, self.rank, m, self._rows, *self.pool_addrs, *self.ctl_addrs,
                      torch.cuda.current_stream())
        return out[:m]
