# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache and entry of the clamped SiLU-mul + MXFP8 quant."""

import functools
import struct

import torch
from amd_kernels.warm_up import entry

from . import kernel as _kernel

I = _kernel.I


@functools.cache
def _kernel_exe():
    return _kernel.compile_clamp_silu_mul_mxfp8()


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _clamp_silu_mul_mxfp8(gate_up, limit=10.0, *, codes=None, scale=None):
    m = gate_up.shape[0]
    assert gate_up.dtype == torch.bfloat16 and gate_up.ndim == 2 and gate_up.shape[1] == 2 * I
    assert gate_up.stride(1) == 1
    dev = gate_up.device
    codes = torch.empty((m, I), dtype=torch.float8_e4m3fn, device=dev) if codes is None else codes
    scale = torch.empty((m, I // 32), dtype=torch.uint8, device=dev) if scale is None else scale
    assert codes.shape == (m, I) and codes.dtype == torch.float8_e4m3fn and codes.is_contiguous()
    assert scale.shape == (m, I // 32) and scale.dtype == torch.uint8 and scale.is_contiguous()
    if m:
        _run_compiled(
            _kernel_exe(),
            gate_up.data_ptr(),
            codes.data_ptr(),
            scale.data_ptr(),
            m,
            gate_up.stride(0),
            struct.unpack("<i", struct.pack("<f", limit))[0],
            torch.cuda.current_stream(),
        )
    return codes, scale


def _mode(gate_up, limit=10.0, **kw):
    return ()


def _sizes(mode):
    return (16,)


def _inputs(mode, n, gate_up, limit=10.0, **kw):
    return (torch.zeros(n, 2 * I, dtype=torch.bfloat16, device=gate_up.device), limit), {}


clamp_silu_mul_mxfp8 = entry("activation/clamp_silu_mul_mxfp8_i2304", mode=_mode, sizes=_sizes, inputs=_inputs)(
    _clamp_silu_mul_mxfp8
)
clamp_silu_mul_mxfp8.__doc__ = """Clamped SiLU-mul of ``gate_up`` [m, 4608] bf16 (gate | up, unit column stride, any
row stride) and its MXFP8 quant in one launch (``reference.py``). Returns (codes e4m3 [m, 2304], scales E8M0 u8
[m, 72] row-major); ``codes`` / ``scale`` may be the caller's buffers. CUDA-graph safe."""


def warmup_clamp_silu_mul_mxfp8(device: torch.device | None = None) -> None:
    """Compile the kernel before graph capture."""
    clamp_silu_mul_mxfp8(torch.zeros(16, 2 * I, dtype=torch.bfloat16, device=device or "cuda"))
