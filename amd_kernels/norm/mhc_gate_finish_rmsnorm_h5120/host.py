# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache and entry of the mHC seam finisher."""

import functools
import struct

import torch
from amd_kernels.warm_up import entry

from . import kernel as _kernel

H = _kernel.H
PSLOT = _kernel.PSLOT
MAX_ROWS = 4096


@functools.cache
def _kernel_exe(nb, quant, repeat):
    return _kernel.compile_mhc_gate_finish(nb, quant, repeat)


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _f32bits(x):
    return struct.unpack("<i", struct.pack("<f", x))[0]


def _mhc_gate_finish(
    partials, x1, hc_scale, hc_base, norm_weight, *, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, norm_eps,
    sinkhorn_repeat=20, quant=False,
):  # fmt: skip
    m, nb, slot = partials.shape
    dev = partials.device
    assert slot == PSLOT and partials.dtype == torch.float32 and partials.is_contiguous() and 1 <= nb <= 640
    assert x1.dtype == torch.bfloat16 and x1.shape == (m, H) and x1.is_contiguous() and m <= MAX_ROWS
    assert hc_scale.shape == (3,) and hc_base.shape == (24,) and hc_scale.dtype == hc_base.dtype == torch.float32
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.shape == (H,) and norm_weight.is_contiguous()
    post = torch.empty((m, 4, 1), dtype=torch.float32, device=dev)
    comb = torch.empty((m, 4, 4), dtype=torch.float32, device=dev)
    next_pre = torch.empty((m, 4), dtype=torch.float32, device=dev)
    out = torch.empty((m, H), dtype=torch.bfloat16, device=dev)
    codes = torch.empty((m, H), dtype=torch.float8_e4m3fn, device=dev) if quant else out
    scale = torch.empty((m, H // 32), dtype=torch.uint8, device=dev) if quant else out
    if m:
        _run_compiled(
            _kernel_exe(nb, quant, sinkhorn_repeat), partials.data_ptr(), x1.data_ptr(), hc_scale.data_ptr(),
            hc_base.data_ptr(), norm_weight.data_ptr(), post.data_ptr(), comb.data_ptr(), next_pre.data_ptr(),
            out.data_ptr(), codes.data_ptr(), scale.data_ptr(), m, _f32bits(rms_eps), _f32bits(hc_pre_eps),
            _f32bits(hc_sinkhorn_eps), _f32bits(hc_post_mult), _f32bits(norm_eps), torch.cuda.current_stream(),
        )  # fmt: skip
    return (post, comb, out, next_pre) + ((codes, scale) if quant else ())


def _mode(partials, x1, *args, quant=False, **kwargs):
    return (partials.shape[1], bool(quant), kwargs.get("sinkhorn_repeat", 20))


def _sizes(mode):
    return (16,)


def _inputs(mode, n, partials, x1, hc_scale, hc_base, norm_weight, **kw):
    dev = partials.device
    return (torch.zeros(n, mode[0], PSLOT, device=dev), torch.zeros(n, H, dtype=torch.bfloat16, device=dev),
            hc_scale, hc_base, norm_weight), kw  # fmt: skip


mhc_gate_finish = entry("norm/mhc_gate_finish_rmsnorm_h5120", mode=_mode, sizes=_sizes, inputs=_inputs)(_mhc_gate_finish)
mhc_gate_finish.__doc__ = """Gates and RMSNorm of one mHC seam from the producer GEMM's tile partials (``reference.py``):
``partials`` fp32 [m, nb, 32] and ``x1`` bf16 [m, 5120] as ``gemm/mxfp8_postmix_hc4_n5120_k8192`` writes them. Returns
(post [m, 4, 1], comb [m, 4, 4], layer_input bf16 [m, 5120], next_pre [m, 4]) and with ``quant`` also MXFP8 codes e4m3
[m, 5120] and E8M0 bytes u8 [m, 160] of ``layer_input``. m <= 4096; CUDA-graph safe, deterministic."""
