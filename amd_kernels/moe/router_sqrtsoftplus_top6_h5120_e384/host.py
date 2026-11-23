# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache and entry of the router GEMM + sqrt-softplus top-6."""

import functools
import struct

import torch
from amd_kernels.warm_up import entry

from . import kernel as _kernel

H, E, TOPK = _kernel.H, _kernel.E, _kernel.TOPK


@functools.cache
def _gemm_exe(a: int, b: int, ksp: int, kw: int, chunk: int, xm: int, xn: int, wshuf: bool = False):
    return _kernel.compile_router_gemm(a, b, ksp, kw, chunk, xm, xn, wshuf)


def shuffle_gate(weight: torch.Tensor) -> torch.Tensor:
    """The load-time layout of the gate ``[384, 5120]`` bf16: ``[expert tile, k step, lane, 8]`` with lane = 16 g + l and
    element ``w[16 tile + l, 32 step + 8 g + e]``, so a wave's fragment load is one 1 KB run. A permutation (bit-exact);
    the caller replaces the weight once, before graph capture, and passes ``weight_shuffled=True``."""
    assert weight.dtype == torch.bfloat16 and weight.shape == (E, H)
    return weight.reshape(E // 16, 16, H // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().reshape(E, H)


@functools.cache
def _topk_exe(ksp: int, has_padding: bool):
    return _kernel.compile_router_topk(ksp, has_padding)


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


# (a, b, ksp, kw, chunk, xm, xn) per row-count bucket: the fastest of a random sweep at its upper bound
# (benchmarks/moe/router_sqrtsoftplus_top6_h5120_e384/sweep_tiles.py)
_TILES = (
    (16, (1, 1, 8, 1, 10, 1, 8)),
    (32, (2, 1, 8, 2, 10, 1, 8)),
    (48, (1, 2, 5, 4, 8, 2, 4)),
    (64, (2, 1, 5, 1, 8, 2, 4)),
    (96, (3, 3, 16, 1, 5, 1, 8)),
    (128, (2, 3, 16, 1, 5, 2, 4)),
    (192, (3, 3, 8, 2, 5, 1, 8)),
    (256, (2, 4, 5, 8, 4, 8, 1)),
    (384, (3, 3, 8, 1, 5, 2, 4)),
    (512, (2, 3, 2, 4, 5, 8, 1)),
    (768, (3, 6, 8, 2, 2, 2, 4)),
    (1024, (4, 3, 10, 1, 4, 8, 1)),
)
_LARGE = (4, 3, 1, 4, 5, 8, 1)
# the same for the load-time layout of the gate (``shuffle_gate``): a wave's weight fragment is one 1 KB run
# (the same random sweep with ``weight_shuffled``)
_TILES_SHUF = (
    (16, (1, 1, 8, 1, 10, 1, 8)),
    (32, (1, 2, 8, 4, 5, 2, 4)),
    (48, (1, 3, 10, 4, 4, 1, 8)),
    (64, (1, 2, 10, 2, 8, 2, 4)),
    (96, (3, 2, 8, 1, 5, 2, 4)),
    (128, (2, 3, 8, 4, 5, 4, 2)),
    (192, (2, 4, 5, 2, 4, 4, 2)),
    (256, (2, 3, 4, 4, 5, 8, 1)),
    (384, (3, 2, 8, 1, 5, 2, 4)),
    (512, (2, 3, 4, 2, 5, 8, 1)),
    (768, (3, 3, 8, 1, 5, 8, 1)),
    (1024, (2, 4, 4, 1, 5, 8, 1)),
)


def _tile(m: int, shuffled: bool = False) -> tuple[int, int, int, int, int, int, int]:
    """(a, b, ksp, kw, chunk, xm, xn) of the GEMM at ``m`` rows."""
    for bound, tile in _TILES_SHUF if shuffled else _TILES:
        if m <= bound:
            return tile
    return _LARGE


def _router_topk(x, weight, bias, routed_scaling_factor=1.5, *, is_padding=None, topk_weights=None, topk_ids=None,
                 partials=None, weight_shuffled=False):
    m = x.shape[0]
    assert x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape[1] == H and x.stride(1) == 1
    assert weight.dtype == torch.bfloat16 and weight.shape == (E, H) and weight.is_contiguous()
    assert bias.dtype == torch.float32 and bias.shape == (E,) and bias.is_contiguous()
    dev = x.device
    if topk_weights is None:
        topk_weights = torch.empty((m, TOPK), dtype=torch.float32, device=dev)
    if topk_ids is None:
        topk_ids = torch.empty((m, TOPK), dtype=torch.int32, device=dev)
    tile = _tile(m, weight_shuffled)
    ksp = tile[2]
    if partials is None:
        partials = torch.empty((ksp, m, E), dtype=torch.float32, device=dev)
    assert topk_weights.shape == (m, TOPK) and topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
    assert topk_ids.shape == (m, TOPK) and topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
    assert partials.shape == (ksp, m, E) and partials.dtype == torch.float32 and partials.is_contiguous()
    if is_padding is not None:
        assert is_padding.dtype == torch.bool and is_padding.shape == (m,) and is_padding.is_contiguous()
    if m:
        stream = torch.cuda.current_stream()
        _run_compiled(_gemm_exe(*tile, weight_shuffled), x.data_ptr(), weight.data_ptr(), partials.data_ptr(), m, x.stride(0), stream)
        _run_compiled(
            _topk_exe(ksp, is_padding is not None),
            partials.data_ptr(),
            bias.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            topk_weights.data_ptr(),
            topk_ids.data_ptr(),
            m,
            struct.unpack("<i", struct.pack("<f", routed_scaling_factor))[0],
            stream,
        )
    return topk_weights, topk_ids


def _mode(x, weight, bias, routed_scaling_factor=1.5, *, is_padding=None, weight_shuffled=False, **kw):
    return (is_padding is not None, weight_shuffled)


def _sizes(mode):
    return tuple(bound for bound, _ in _TILES) + (1025,)  # the shuffled table has the same bounds; 1025: ``_LARGE``


def _inputs(mode, n, x, weight, bias, routed_scaling_factor=1.5, *, is_padding=None, weight_shuffled=False, **kw):
    pad = None if is_padding is None else torch.zeros(n, dtype=torch.bool, device=x.device)
    return (torch.zeros(n, H, dtype=torch.bfloat16, device=x.device), weight, bias, routed_scaling_factor), {
        "is_padding": pad,
        "weight_shuffled": weight_shuffled,
    }


router_topk = entry("moe/router_sqrtsoftplus_top6_h5120_e384", mode=_mode, sizes=_sizes, inputs=_inputs)(_router_topk)
router_topk.__doc__ = """Router of one MoE layer: ``logits = x @ weight.T`` (fp32 accumulate), per-expert
``score = sqrt(softplus(logit))``, the 6 experts with the largest ``score + bias``, weights ``score`` of the picks
scaled to sum ``routed_scaling_factor`` (``reference.py``). ``x`` bf16 [m, 5120] (unit column stride), ``weight`` bf16
[384, 5120], ``bias`` fp32 [384]; ``is_padding`` bool [m]: those rows get id -1 and weight 0; ``weight_shuffled=True``: ``weight`` is
``shuffle_gate(gate)`` (a load-time permutation, faster GEMM). Returns (topk_weights fp32
[m, 6], topk_ids int32 [m, 6]) in descending ``score + bias`` order; outputs and the fp32 logits scratch may be the
caller's buffers. CUDA-graph safe."""


def warmup_router_topk(device: torch.device | None = None) -> None:
    """Compile every variant before graph capture."""
    dev = device or "cuda"
    x = torch.zeros(16, H, dtype=torch.bfloat16, device=dev)
    w = torch.zeros(E, H, dtype=torch.bfloat16, device=dev)
    b = torch.zeros(E, dtype=torch.float32, device=dev)
    for pad in (None, torch.zeros(16, dtype=torch.bool, device=dev)):
        router_topk(x, w, b, is_padding=pad)
        router_topk(x, shuffle_gate(w), b, is_padding=pad, weight_shuffled=True)
