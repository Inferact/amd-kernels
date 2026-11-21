# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache and entry of the step-end mHC collapse + RMSNorm."""

import functools
import struct

import torch
from amd_kernels.warm_up import entry

from . import kernel as _kernel

H = _kernel.H


@functools.cache
def _kernel_exe(has_post, keep_residual, mean):
    return _kernel.compile_mhc_post_collapse_rmsnorm(has_post, keep_residual, mean)


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _mhc_post_collapse_rmsnorm(
    residual,
    pre_mix,
    norm_weight,
    *,
    eps,
    sublayer_out=None,
    post_mix=None,
    comb_mix=None,
    keep_residual=False,
    aux_mean=False,
):
    m = residual.shape[0]
    dev = residual.device
    assert residual.dtype == torch.bfloat16 and residual.shape == (m, 4, H) and residual.is_contiguous()
    assert pre_mix.dtype == torch.float32 and pre_mix.numel() == m * 4 and pre_mix.is_contiguous()
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.shape == (H,) and norm_weight.is_contiguous()
    has_post = sublayer_out is not None
    if has_post:
        assert sublayer_out.shape == (m, H) and sublayer_out.dtype == torch.bfloat16 and sublayer_out.is_contiguous()
        assert post_mix.numel() == m * 4 and comb_mix.numel() == m * 16 and post_mix.is_contiguous()
        assert comb_mix.is_contiguous() and post_mix.dtype == comb_mix.dtype == torch.float32
    else:
        assert post_mix is None and comb_mix is None and not keep_residual
    out = torch.empty((m, H), dtype=torch.bfloat16, device=dev)
    residual_out = torch.empty_like(residual) if keep_residual else None
    mean = torch.empty((m, H), dtype=torch.bfloat16, device=dev) if aux_mean else None
    if m:
        p = lambda t, alt: (t if t is not None else alt).data_ptr()  # noqa: E731
        _run_compiled(
            _kernel_exe(has_post, keep_residual, aux_mean),
            residual.data_ptr(),
            p(sublayer_out, residual),
            p(post_mix, residual),
            p(comb_mix, residual),
            pre_mix.data_ptr(),
            norm_weight.data_ptr(),
            p(residual_out, out),
            out.data_ptr(),
            p(mean, out),
            m,
            struct.unpack("<i", struct.pack("<f", eps))[0],
            torch.cuda.current_stream(),
        )
    return out, residual_out, mean


def _mode(residual, pre_mix, norm_weight, *, sublayer_out=None, keep_residual=False, aux_mean=False, **kw):
    return (sublayer_out is not None, bool(keep_residual), bool(aux_mean))


def _sizes(mode):
    return (16,)


def _inputs(mode, n, residual, pre_mix, norm_weight, *, eps, **kw):
    dev = residual.device
    z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
    extra = {}
    if mode[0]:
        extra = dict(sublayer_out=z(n, H, dt=torch.bfloat16), post_mix=z(n, 4, 1), comb_mix=z(n, 4, 4))
    return (z(n, 4, H, dt=torch.bfloat16), z(n, 4), norm_weight), dict(
        eps=eps, keep_residual=mode[1], aux_mean=mode[2], **extra
    )


mhc_post_collapse_rmsnorm = entry("norm/mhc_post_collapse_rmsnorm_h5120", mode=_mode, sizes=_sizes, inputs=_inputs)(
    _mhc_post_collapse_rmsnorm
)
mhc_post_collapse_rmsnorm.__doc__ = """Step-end tail of the mHC residual (``reference.py``): post-mix of the last
sub-layer (``sublayer_out`` / ``post_mix`` / ``comb_mix``, absent: ``residual`` is already the new residual), collapse
with ``pre_mix`` and RMSNorm. ``residual`` bf16 [m, 4, 5120], ``pre_mix`` fp32 [m, 4], ``norm_weight`` bf16 [5120].
Returns (layer_input bf16 [m, 5120], new residual bf16 [m, 4, 5120] if ``keep_residual`` else None, mean over the four
streams bf16 [m, 5120] if ``aux_mean`` else None). CUDA-graph safe."""


def warmup_mhc_post_collapse_rmsnorm(device: torch.device | None = None) -> None:
    """Compile every variant before graph capture."""
    dev = device or "cuda"
    w = torch.zeros(H, dtype=torch.bfloat16, device=dev)
    for post in (False, True):
        for keep in (False, True) if post else (False,):
            for mean in (False, True):
                kw = {}
                if post:
                    kw = dict(sublayer_out=torch.zeros(16, H, dtype=torch.bfloat16, device=dev),
                              post_mix=torch.zeros(16, 4, device=dev), comb_mix=torch.zeros(16, 4, 4, device=dev))
                mhc_post_collapse_rmsnorm(torch.zeros(16, 4, H, dtype=torch.bfloat16, device=dev),
                                          torch.zeros(16, 4, device=dev), w, eps=1e-6, keep_residual=keep,
                                          aux_mean=mean, **kw)
