# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the indexer query RoPE + MXFP4 quant."""

import functools
import struct

import torch

from amd_kernels.warm_up import entry

from . import kernel as _kernel


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(cs_bf16: bool):
    return _kernel.compile_q_rope_mxfp4_quant(cs_bf16)


@functools.cache
def _on_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def supports_q_rope_mxfp4_quant(q: torch.Tensor, cos_sin_cache: torch.Tensor) -> bool:
    return (
        _on_gfx950()
        and q.dtype == torch.bfloat16
        and q.dim() == 3
        and tuple(q.shape[1:]) == (_kernel.H, _kernel.D)
        and q.is_contiguous()
        and cos_sin_cache.dtype in (torch.float32, torch.bfloat16)
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[1] == _kernel.ROT
    )


def _q_rope_mxfp4_quant(
    q: torch.Tensor,  # [n, 32, 128] bf16, contiguous
    positions: torch.Tensor,  # [n] int64
    cos_sin_cache: torch.Tensor,  # [rows, 64] fp32 or bf16
    weights: torch.Tensor,  # [n, 32] bf16
    weights_scale: float,
    q_fp4: torch.Tensor | None = None,  # [n, 32, 64] u8
    q_scale: torch.Tensor | None = None,  # [n, 32, 4] u8
    weights_out: torch.Tensor | None = None,  # [n, 32] fp32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP4 query of the RoPE'd heads, its e8m0 scales and ``weights * weights_scale`` in fp32; the outputs are
    allocated unless given."""
    n = q.shape[0]
    dev = q.device
    if q_fp4 is None:
        q_fp4 = torch.empty((n, _kernel.H, _kernel.D // 2), dtype=torch.uint8, device=dev)
    if q_scale is None:
        q_scale = torch.empty((n, _kernel.H, _kernel.D // 32), dtype=torch.uint8, device=dev)
    if weights_out is None:
        weights_out = torch.empty((n, _kernel.H), dtype=torch.float32, device=dev)
    if n == 0:
        return q_fp4, q_scale, weights_out
    assert q.is_contiguous() and positions.dtype == torch.int64 and positions.is_contiguous()
    assert weights.dtype == torch.bfloat16 and weights.is_contiguous() and tuple(weights.shape) == (n, _kernel.H)
    assert cos_sin_cache.is_contiguous()
    _run_compiled(
        _get_kernel(cos_sin_cache.dtype == torch.bfloat16),
        q.data_ptr(),
        positions.data_ptr(),
        cos_sin_cache.data_ptr(),
        weights.data_ptr(),
        q_fp4.data_ptr(),
        q_scale.data_ptr(),
        weights_out.data_ptr(),
        n,
        cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", weights_scale))[0],
        torch.cuda.current_stream(dev),
    )
    return q_fp4, q_scale, weights_out


def _mode(q, positions, cos_sin_cache, *args, **kwargs):
    """What a call fixes about its kernel (``amd_kernels.warm_up``; the row count does not)."""
    return (cos_sin_cache.dtype == torch.bfloat16,)


def _inputs(mode, n, q, positions, cos_sin_cache, weights, weights_scale, *args, **kwargs):
    dev = q.device
    return (
        torch.zeros((n, *q.shape[1:]), dtype=q.dtype, device=dev),
        torch.zeros(n, dtype=torch.int64, device=dev),
        cos_sin_cache,
        torch.zeros((n, weights.shape[1]), dtype=weights.dtype, device=dev),
        weights_scale,
    ), {}


q_rope_mxfp4_quant = entry("indexer/q_rope_mxfp4_quant_h32_d128_rot64", mode=_mode, sizes=lambda mode: (8,),
                           inputs=_inputs)(_q_rope_mxfp4_quant)
