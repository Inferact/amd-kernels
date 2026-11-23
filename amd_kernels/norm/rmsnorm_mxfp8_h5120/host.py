# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache and entry of the RMSNorm + MXFP8 quant of a 5120-column row."""

import functools
import struct

import torch
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from . import kernel as _kernel

H = _kernel.H


@functools.cache
def _kernel_exe():
    return _kernel.compile_rmsnorm_mxfp8()


def _rmsnorm_mxfp8(x, weight, eps, *, codes=None, scale=None):
    m = x.shape[0]
    assert x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape[1] == H and x.stride(1) == 1
    assert weight.dtype == torch.bfloat16 and weight.shape == (H,) and weight.is_contiguous()
    assert weight.device == x.device
    dev = x.device
    codes = torch.empty((m, H), dtype=torch.float8_e4m3fn, device=dev) if codes is None else codes
    scale = torch.empty((m, H // 32), dtype=torch.uint8, device=dev) if scale is None else scale
    assert codes.shape == (m, H) and codes.dtype == torch.float8_e4m3fn and codes.is_contiguous()
    assert scale.shape == (m, H // 32) and scale.dtype == torch.uint8 and scale.is_contiguous()
    if m:
        _run_compiled(_kernel_exe(), x.data_ptr(), weight.data_ptr(), codes.data_ptr(), scale.data_ptr(), m,
                      x.stride(0), struct.unpack("<i", struct.pack("<f", eps))[0], torch.cuda.current_stream())
    return codes, scale


def _mode(x, weight, eps, **kw):
    return ()


def _sizes(mode):
    return (16,)


def _inputs(mode, n, x, weight, eps, **kw):
    return (torch.zeros(n, H, dtype=x.dtype, device=x.device), weight, eps), {}


rmsnorm_mxfp8 = entry("norm/rmsnorm_mxfp8_h5120", mode=_mode, sizes=_sizes, inputs=_inputs)(_rmsnorm_mxfp8)
rmsnorm_mxfp8.__doc__ = """RMSNorm of ``x`` [m, 5120] to MXFP8 (``reference.py``). ``x`` bf16 with unit column stride and any row
stride (a GEMM's output), ``weight`` bf16 contiguous, fp32 ``eps``. Returns (codes e4m3 [m, 5120], scales E8M0 u8
[m, 160], both row-major: a pre-quantized activation of the MXFP8 GEMMs); ``codes`` / ``scale`` may be the caller's
buffers. CUDA-graph safe."""


def warmup_rmsnorm_mxfp8(device: torch.device | None = None) -> None:
    """Compile the kernel before graph capture."""
    dev = device or "cuda"
    rmsnorm_mxfp8(torch.zeros(16, H, dtype=torch.bfloat16, device=dev), torch.zeros(H, dtype=torch.bfloat16, device=dev),
                  1e-20)
