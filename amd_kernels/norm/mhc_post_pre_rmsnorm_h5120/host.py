# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache, scratch, dispatch and entry of the mHC seam."""

import functools
import struct

import torch
from amd_kernels.warm_up import entry

from . import kernel as _kernel

H, CHUNKS = _kernel.H, _kernel.CHUNKS
MAX_ROWS = 4096
N_MIX = 24
_CU = 256  # grid bound: every split of a block must be resident while it waits for the last one
_PART_BYTES = 8 << 20
_scratch: dict = {}


def pack_gate_weight(fn: torch.Tensor) -> torch.Tensor:
    """fp32 ``fn`` [24, 4 * 5120] -> the kernel's bf16 hi / lo B operands (once, at load time)."""
    assert fn.shape == (N_MIX, 4 * H) and fn.dtype == torch.float32 and fn.is_contiguous()
    with torch.no_grad():
        hi = fn.to(torch.bfloat16)
        lo = (fn - hi.float()).to(torch.bfloat16)
        hi = hi.view(N_MIX, 4, CHUNKS, 4, 8)
        lo = lo.view(N_MIX, 4, CHUNKS, 4, 8)
        ops = torch.stack([hi[:16], lo[:16], torch.cat([hi[16:], lo[16:]])])  # [3, 16, s, c, g, j]
        return ops.permute(2, 3, 0, 4, 1, 5).contiguous()  # [s, c, op, g, n, j]


_TUNED = ((8, 20, 8), (16, 16, 10), (24, 10, 16), (32, 8, 5), (51, 5, 8))  # (row blocks <=, splits, waves)


def _config(m: int) -> tuple[int, int, int]:
    """(row tiles of 16 per workgroup, splits, waves), a grid within the CU count; from the m = 16 ... 768 sweeps in
    ``benchmarks/norm/mhc_post_pre_rmsnorm_h5120`` (two or four row tiles per workgroup measured slower at every m)."""
    blocks = -(-m // 16)
    for bound, ks, waves in _TUNED:
        if blocks <= bound:
            return 1, ks, waves
    ks = max(k for k in (1, 2, 4) if blocks * k <= _CU or k == 1)
    return 1, ks, 8 if ks > 1 else 16


@functools.cache
def _kernel_exe(rt, ks, waves, has_post, ident_pre, quant, mean, repeat):
    return _kernel.compile_mhc_seam(rt, ks, waves, has_post, ident_pre, quant, mean, repeat)


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _scratch_for(device, rt):
    s = _scratch.get(device)
    if s is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mhc_post_pre_rmsnorm: warm up every row-tile variant outside graph capture")
        s = _scratch[device] = (torch.zeros(_PART_BYTES // 4, dtype=torch.float32, device=device), {})
    part, ctrls = s
    if rt not in ctrls:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mhc_post_pre_rmsnorm: warm up every row-tile variant outside graph capture")
        ctrls[rt] = torch.zeros(MAX_ROWS // 16, 16 + 32 * rt, dtype=torch.int32, device=device)
    return part, ctrls[rt]


def _f32bits(x):
    return struct.unpack("<i", struct.pack("<f", x))[0]


def _mhc_post_pre_rmsnorm(
    residual,
    fn_packed,
    hc_scale,
    hc_base,
    norm_weight,
    *,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult,
    norm_eps,
    sinkhorn_repeat=20,
    pre_mix=None,
    sublayer_out=None,
    post_mix=None,
    comb_mix=None,
    residual_out=None,
    quant=False,
    aux_mean=False,
):
    m = residual.shape[0]
    dev = residual.device
    assert residual.dtype == torch.bfloat16 and residual.shape == (m, 4, H) and residual.is_contiguous()
    assert fn_packed.dtype == torch.bfloat16 and fn_packed.shape == (4, CHUNKS, 3, 4, 16, 8)
    assert hc_scale.shape == (3,) and hc_base.shape == (N_MIX,) and hc_scale.dtype == hc_base.dtype == torch.float32
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.shape == (H,) and norm_weight.is_contiguous()
    assert m <= MAX_ROWS
    has_post = sublayer_out is not None
    if has_post:
        assert sublayer_out.shape == (m, H) and sublayer_out.dtype == torch.bfloat16 and sublayer_out.is_contiguous()
        assert post_mix.numel() == m * 4 and comb_mix.numel() == m * 16 and post_mix.is_contiguous()
        assert comb_mix.is_contiguous() and post_mix.dtype == comb_mix.dtype == torch.float32
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        assert residual_out.shape == residual.shape and residual_out.is_contiguous()
        assert residual_out.untyped_storage().data_ptr() != residual.untyped_storage().data_ptr()
    else:
        assert residual_out is None and post_mix is None and comb_mix is None
        residual_out = residual
    ident = pre_mix is None
    if not ident:
        assert pre_mix.numel() == m * 4 and pre_mix.dtype == torch.float32 and pre_mix.is_contiguous()
    post = torch.empty((m, 4, 1), dtype=torch.float32, device=dev)
    comb = torch.empty((m, 4, 4), dtype=torch.float32, device=dev)
    next_pre = torch.empty((m, 4), dtype=torch.float32, device=dev)
    out = torch.empty((m, H), dtype=torch.bfloat16, device=dev)
    codes = torch.empty((m, H), dtype=torch.float8_e4m3fn, device=dev) if quant else out
    scale = torch.empty((m, H // 32), dtype=torch.uint8, device=dev) if quant else out
    mean = torch.empty((m, H), dtype=torch.bfloat16, device=dev) if aux_mean else None
    if m:
        rt, ks, waves = _config(m)
        part, ctrl = _scratch_for(dev, rt)
        p = lambda t: t.data_ptr()  # noqa: E731
        _run_compiled(
            _kernel_exe(rt, ks, waves, has_post, ident, quant, mean is not None, sinkhorn_repeat),
            p(residual),
            p(fn_packed),
            p(sublayer_out) if has_post else p(residual),
            p(post_mix) if has_post else p(residual),
            p(comb_mix) if has_post else p(residual),
            p(pre_mix) if not ident else p(residual),
            p(hc_scale),
            p(hc_base),
            p(norm_weight),
            p(residual_out),
            p(post),
            p(comb),
            p(next_pre),
            p(out),
            p(codes),
            p(scale),
            p(mean) if mean is not None else p(out),
            p(part),
            p(ctrl),
            m,
            _f32bits(rms_eps),
            _f32bits(hc_pre_eps),
            _f32bits(hc_sinkhorn_eps),
            _f32bits(hc_post_mult),
            _f32bits(norm_eps),
            torch.cuda.current_stream(),
        )
    res = (residual_out, post, comb, out, next_pre)
    res = res + (codes, scale) if quant else res
    return res + (mean,) if aux_mean else res


def _mode(residual, fn_packed, hc_scale, hc_base, norm_weight, *, sublayer_out=None, pre_mix=None, quant=False,
          aux_mean=False, sinkhorn_repeat=20, **kw):
    return (sublayer_out is not None, pre_mix is None, bool(quant), bool(aux_mean), sinkhorn_repeat)


def _sizes(mode):
    seen = {}
    for m in range(16, MAX_ROWS + 1, 16):
        seen.setdefault(_config(m), m)
    return tuple(sorted(seen.values()))


def _inputs(mode, n, residual, fn_packed, hc_scale, hc_base, norm_weight, *, sublayer_out=None, pre_mix=None,
            post_mix=None, comb_mix=None, **kw):
    z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=residual.device)  # noqa: E731
    kw = dict(kw)
    kw.pop("residual_out", None)
    if mode[0]:
        kw.update(sublayer_out=torch.zeros(n, H, dtype=torch.bfloat16, device=residual.device),
                  post_mix=z(n, 4, 1), comb_mix=z(n, 4, 4))
    if not mode[1]:
        kw.update(pre_mix=z(n, 4))
    return (torch.zeros(n, 4, H, dtype=torch.bfloat16, device=residual.device), fn_packed, hc_scale, hc_base,
            norm_weight), kw


mhc_post_pre_rmsnorm = entry("norm/mhc_post_pre_rmsnorm_h5120", mode=_mode, sizes=_sizes, inputs=_inputs)(
    _mhc_post_pre_rmsnorm
)
mhc_post_pre_rmsnorm.__doc__ = """mHC seam (``reference.py``): post-mix of the previous sub-layer, gate projection,
Sinkhorn gates, collapse with the carried ``pre_mix`` and RMSNorm, in one launch. ``residual`` bf16 [m, 4, 5120],
``fn_packed`` from ``pack_gate_weight``; ``sublayer_out`` / ``post_mix`` / ``comb_mix`` / ``residual_out`` as one group
(absent: no post-mix), ``pre_mix`` absent: collapse is stream 0. Returns (residual_out, post [m, 4, 1], comb
[m, 4, 4], layer_input bf16 [m, 5120], next_pre [m, 4]); with ``quant`` also MXFP8 codes e4m3 [m, 5120] and E8M0 bytes
u8 [m, 160] row-major of ``layer_input``. m <= 4096. Calls share one scratch per device: serial per device, CUDA-graph
safe after one call outside capture."""


def warmup_mhc_post_pre_rmsnorm(device: torch.device | None = None) -> None:
    """Compile every variant before graph capture."""
    dev = device or "cuda"
    fn = pack_gate_weight(torch.zeros(N_MIX, 4 * H, dtype=torch.float32, device=dev))
    sc, base = torch.zeros(3, device=dev), torch.zeros(N_MIX, device=dev)
    w = torch.zeros(H, dtype=torch.bfloat16, device=dev)
    for quant in (False, True):
        for post in (False, True):
            for ident in (False, True):
                m = 16
                kw = {}
                if post:
                    kw.update(sublayer_out=torch.zeros(m, H, dtype=torch.bfloat16, device=dev),
                              post_mix=torch.zeros(m, 4, device=dev), comb_mix=torch.zeros(m, 4, 4, device=dev))
                if not ident:
                    kw.update(pre_mix=torch.zeros(m, 4, device=dev))
                mhc_post_pre_rmsnorm(torch.zeros(m, 4, H, dtype=torch.bfloat16, device=dev), fn, sc, base, w,
                                     rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult=1.0,
                                     norm_eps=1e-6, quant=quant, **kw)
