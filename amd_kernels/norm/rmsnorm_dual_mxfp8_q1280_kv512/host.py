# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache and entry of the dual RMSNorm + MXFP8 quant."""

import functools
import struct

import torch
from amd_kernels.warm_up import entry

from . import kernel as _kernel

Q, KV = _kernel.Q, _kernel.KV


@functools.cache
def _kernel_exe():
    return _kernel.compile_rmsnorm_dual_mxfp8()


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _rmsnorm_dual_mxfp8(q, kv, q_weight, kv_weight, eps, *, codes=None, scale=None, kv_out=None):
    m = q.shape[0]
    assert q.dtype == torch.bfloat16 and q.ndim == 2 and q.shape[1] == Q and q.stride(1) == 1
    assert kv.dtype == torch.bfloat16 and kv.shape == (m, KV) and kv.stride(1) == 1 and kv.device == q.device
    for w, n in ((q_weight, Q), (kv_weight, KV)):
        assert w.dtype == torch.bfloat16 and w.shape == (n,) and w.is_contiguous() and w.device == q.device
    dev = q.device
    codes = torch.empty((m, Q), dtype=torch.float8_e4m3fn, device=dev) if codes is None else codes
    scale = torch.empty((m, Q // 32), dtype=torch.uint8, device=dev) if scale is None else scale
    kv_out = torch.empty((m, KV), dtype=torch.bfloat16, device=dev) if kv_out is None else kv_out
    assert codes.shape == (m, Q) and codes.dtype == torch.float8_e4m3fn and codes.is_contiguous()
    assert scale.shape == (m, Q // 32) and scale.dtype == torch.uint8 and scale.is_contiguous()
    assert kv_out.shape == (m, KV) and kv_out.dtype == torch.bfloat16 and kv_out.is_contiguous()
    if m:
        _run_compiled(
            _kernel_exe(),
            q.data_ptr(),
            kv.data_ptr(),
            q_weight.data_ptr(),
            kv_weight.data_ptr(),
            codes.data_ptr(),
            scale.data_ptr(),
            kv_out.data_ptr(),
            m,
            q.stride(0),
            kv.stride(0),
            struct.unpack("<i", struct.pack("<f", eps))[0],
            torch.cuda.current_stream(),
        )
    return codes, scale, kv_out


def _mode(q, kv, q_weight, kv_weight, eps, **kw):
    return ()


def _sizes(mode):
    return (16,)


def _inputs(mode, n, q, kv, q_weight, kv_weight, eps, **kw):
    z = torch.zeros
    return (z(n, Q, dtype=q.dtype, device=q.device), z(n, KV, dtype=q.dtype, device=q.device), q_weight, kv_weight, eps), {}


rmsnorm_dual_mxfp8 = entry("norm/rmsnorm_dual_mxfp8_q1280_kv512", mode=_mode, sizes=_sizes, inputs=_inputs)(
    _rmsnorm_dual_mxfp8
)
rmsnorm_dual_mxfp8.__doc__ = """RMSNorm of ``q`` [m, 1280] to MXFP8 and of ``kv`` [m, 512] to bf16 in one launch
(``reference.py``). Inputs bf16 with unit column stride and any row stride (the two halves of a fused [m, 1792]
projection); weights bf16 contiguous; fp32 ``eps``. Returns (codes e4m3 [m, 1280], scales E8M0 u8 [m, 40] row-major,
kv bf16 [m, 512]); ``codes`` / ``scale`` / ``kv_out`` may be the caller's buffers. CUDA-graph safe."""


def warmup_rmsnorm_dual_mxfp8(device: torch.device | None = None) -> None:
    """Compile the kernel before graph capture."""
    dev = device or "cuda"
    w = torch.zeros(Q, dtype=torch.bfloat16, device=dev)
    rmsnorm_dual_mxfp8(torch.zeros(16, Q, dtype=torch.bfloat16, device=dev),
                       torch.zeros(16, KV, dtype=torch.bfloat16, device=dev), w, w[:KV].contiguous(), 1e-20)
