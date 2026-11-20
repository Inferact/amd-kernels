# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Paired gate/up MXFP8 GEMM + SwiGLU-OAI, bf16 out, at 1-16 rows (one 16-row block of codes), for the dense MLP
N 12288 (gate | up of I 6144) x K 6144 (TP2): the one-K-wave kernel (``gateup_swiglu_bf16_onewave``) and the
LDS-reduced multi-K-wave kernel (``gateup_swiglu_bf16``). Callers: the TP2 dense decoder block's MLP and the DSpark
draft's MLP."""

import functools
import struct

import torch

N, K, I = 12288, 6144, 6144
# form -> (K-waves, W / A lookahead); K-waves 1 runs the one-wave kernel. 14.9 / 15.3 us at 8 rows, no AGPR parking
# (benchmark_gateup_swiglu_bf16.py --sweep)
CONFIG = {"fp32": (1, (7, 8)), "bf16_chain": (3, (4, 8))}


@functools.cache
def _launcher(form, kw, la):
    if kw == 1:
        from .gateup_swiglu_bf16_onewave import compile_mxfp8_gateup_swiglu_bf16_onewave
        return compile_mxfp8_gateup_swiglu_bf16_onewave(form, LA=la)
    from .gateup_swiglu_bf16 import compile_mxfp8_gateup_swiglu_bf16
    return compile_mxfp8_gateup_swiglu_bf16(form, KW=kw, LA=la)


def _bits(v):
    return struct.unpack('<i', struct.pack('<f', v))[0]


def mxfp8_gateup_swiglu_bf16(activation, activation_scale, weight, weight_scale, m,
                             alpha=1.702, beta=1.0, limit=7.0, form='fp32', config=None):
    """Paired gate/up GEMM and model SwiGLU, BF16 [m, 6144], for 1-16 rows (the same cost at every row count).

    Weight is the preshuffled gate|up MXFP8 layout [12288, 6144]. Activation codes use the 16x16 layout, padded to
    16 rows; scales use AITER's layout. Each dot product rounds to BF16 before the activation. ``form`` is the
    caller's numerical contract (fp32 or bf16_chain), independent of the row count; the K-wave count of ``CONFIG``
    sets the FP32 dot summation order (that of the decode GEMM with waves_m equal to it). ``config``
    (K-waves, lookahead) overrides ``CONFIG`` (sweeps and tests only).
    """
    from amd_kernels.common.tensor_shim import _run_compiled
    assert 1 <= m <= 16 and form in CONFIG
    assert activation.shape == (16, K) and activation.element_size() == 1 and activation.is_contiguous()
    assert weight.shape == (N, K) and weight.element_size() == 1 and weight.is_contiguous()
    assert activation_scale.shape[0] >= 32 and activation_scale.shape[1] == K // 32
    exe = _launcher(form, *(config or CONFIG[form]))
    out = torch.empty((m, I), device=activation.device, dtype=torch.bfloat16)
    u8 = lambda t: t.view(torch.uint8).view(-1)  # noqa: E731
    _run_compiled(exe, u8(activation), u8(weight), u8(out), u8(activation_scale), u8(weight_scale),
                  m, activation_scale.numel(), 1, exe.n_blocks, _bits(alpha), _bits(beta), _bits(limit),
                  torch.cuda.current_stream())
    return out


def warmup_mxfp8_gateup_swiglu_bf16(activation, activation_scale, weight, weight_scale, forms=('fp32', 'bf16_chain')):
    """Compile the reached activation forms before graphs or serving."""
    for form in forms:
        mxfp8_gateup_swiglu_bf16(activation, activation_scale, weight, weight_scale, 8, form=form)
