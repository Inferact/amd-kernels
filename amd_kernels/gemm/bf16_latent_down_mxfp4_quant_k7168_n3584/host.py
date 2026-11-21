# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the latent down projection + MXFP4 quant: the load-time weight layout, K split by rows, the launch."""
import functools

import torch

from amd_kernels.warm_up import entry

from .kernel import (CNT_STRIDE, MT_MAX, NB, TILES, compile_bf16_latent_down_mxfp4_quant_k7168_n3584, partial_bytes,
                     tile_peers_share_l2, xcc_probe)
from .reference import K, N

MAX_M = 16 * MT_MAX
KS_BY_MT = {mt: 4 for mt in range(1, MT_MAX + 1)}


def supports(w: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    """Whether ``w`` is the [3584, 7168] bf16 contiguous down-projection weight (either layout), without bias."""
    return bias is None and tuple(w.shape) == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous()


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """The load-time layout [224 groups, 112 K blocks, 16 rows, 64 K] of the weight, as [3584, 7168] (bit-exact)."""
    assert w.shape == (N, K) and w.dtype == torch.bfloat16
    return w.view(N // 16, 16, K // 64, 64).permute(0, 2, 1, 3).contiguous().view(N, K)


def shuffle_weight_(w: torch.Tensor) -> torch.Tensor:
    """shuffle_weight in place: ``w`` keeps its storage and holds the load-time layout."""
    ws = shuffle_weight(w)
    w.copy_(ws)
    del ws
    return w


def unshuffle_weight(ws: torch.Tensor) -> torch.Tensor:
    assert ws.shape == (N, K) and ws.dtype == torch.bfloat16
    return ws.view(N // 16, K // 64, 16, 64).permute(0, 2, 1, 3).contiguous().view(N, K)


def m_tile_fragments(m: int) -> int:
    return (m + 15) // 16


@functools.cache
def _kernel(mt: int, ks: int, shuf: bool, with_bf16: bool, xcd: bool):
    return compile_bf16_latent_down_mxfp4_quant_k7168_n3584(mt, ks, int(shuf), int(with_bf16), int(xcd))


_XCD_LOCAL: dict = {}


def _xcd_local(device: torch.device) -> bool:
    """Whether the K slices of a tile may exchange partials through one L2: the workgroups placed on one XCC share an
    L2, so a probe launch (once per device) checks the dispatcher's placement. Under graph capture: no (not cached)."""
    key = device.index
    if key not in _XCD_LOCAL:
        if torch.cuda.is_current_stream_capturing():
            return False
        ids = torch.full((TILES * 4,), -1, dtype=torch.int32, device=device)
        xcc_probe(ids.data_ptr(), torch.cuda.current_stream())
        _XCD_LOCAL[key] = tile_peers_share_l2(ids.cpu().tolist())
    return _XCD_LOCAL[key]


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


_WORK: dict = {}


def _workspace(device: torch.device, stream: int):
    """Partial slabs and tile tickets of one (device, stream); the kernel leaves every ticket at 0."""
    key = (device.index, stream)
    ws = _WORK.get(key)
    if ws is None:
        size = max(partial_bytes(mt, ks) for mt, ks in KS_BY_MT.items())
        ws = (torch.empty(size, dtype=torch.uint8, device=device),
              torch.zeros(TILES * CNT_STRIDE, dtype=torch.uint8, device=device))
        _WORK[key] = ws
    return ws


def _down_quant(x: torch.Tensor, w: torch.Tensor, out: tuple | None = None, *, shuffled: bool = False,
                out_bf16: torch.Tensor | None = None):
    """(codes [M, 1792] fp4x2, e8m0 scales [M, 112]) = MXFP4 of bf16 (x [M, 7168] @ W^T), fp32 accumulate.

    W = w (row-major [3584, 7168]) or shuffle_weight's layout. ``out`` = (codes, scales) preallocated (no allocation
    then); ``out_bf16`` [M, 3584] also receives the rounded bf16 projection. Any M (above 192 rows: launches of 192
    rows). Deterministic, no atomic output adds, CUDA-graph safe. One call at a time per stream (the workspace).
    """
    m = x.shape[0]
    assert x.ndim == 2 and x.shape[1] == K and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert supports(w) and x.device == w.device
    if out is None:
        codes = torch.empty(m, N // 2, dtype=torch.uint8, device=x.device).view(torch.float4_e2m1fn_x2)
        scales = torch.empty(m, N // 32, dtype=torch.uint8, device=x.device).view(torch.float8_e8m0fnu)
    else:
        codes, scales = out
    assert codes.shape == (m, N // 2) and codes.is_contiguous() and scales.shape == (m, N // 32)
    assert scales.is_contiguous() and codes.device == scales.device == x.device
    if out_bf16 is not None:
        assert out_bf16.shape == (m, N) and out_bf16.dtype == torch.bfloat16 and out_bf16.is_contiguous()
    if m > MAX_M:
        for i in range(0, m, MAX_M):
            _down_quant(x[i:i + MAX_M], w, (codes[i:i + MAX_M], scales[i:i + MAX_M]), shuffled=shuffled,
                        out_bf16=None if out_bf16 is None else out_bf16[i:i + MAX_M])
        return codes, scales
    if m:
        mt = m_tile_fragments(m)
        stream = torch.cuda.current_stream()
        part, cnt = _workspace(x.device, stream.cuda_stream)
        exe = _kernel(mt, KS_BY_MT[mt], bool(shuffled), out_bf16 is not None, _xcd_local(x.device))
        o = out_bf16 if out_bf16 is not None else codes
        _run(exe, x.data_ptr(), w.data_ptr(), codes.data_ptr(), scales.data_ptr(), o.data_ptr(), part.data_ptr(),
             cnt.data_ptr(), m, stream)
    return codes, scales


def _mode(x, w, out=None, *, shuffled=False, out_bf16=None):
    """What a call fixes about its kernels besides M (``amd_kernels.warm_up``): the weight layout, the bf16 output."""
    return bool(shuffled), out_bf16 is not None, _xcd_local(x.device)


def _inputs(mode, n, x, w, out=None, *, shuffled=False, out_bf16=None):
    kw = {"shuffled": shuffled}
    if mode[1]:
        kw["out_bf16"] = torch.empty(n, N, dtype=torch.bfloat16, device=x.device)
    return (torch.zeros(n, K, dtype=torch.bfloat16, device=x.device), w), kw


bf16_latent_down_mxfp4_quant_k7168_n3584 = entry(
    "gemm/bf16_latent_down_mxfp4_quant_k7168_n3584", mode=_mode, sizes=lambda mode: [16 * mt for mt in range(1, MT_MAX + 1)],
    inputs=_inputs)(_down_quant)
