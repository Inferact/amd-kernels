# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP4 pieces of the indexer shared by both kernels: group scale, E2M1 packing and the key LayerNorm + insert of
one wave (two columns a lane)."""

import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr.typing import T

from .common import f32, maxnum, pk_bf16, pow2_exponent, pow2_pair, row_max, wave_sum

D = 128
PAGE_BYTES = 4352  # 64 slots x 68 bytes: 4096 payload, 256 scales


def e2m1x2(a, b, old, sel):
    """E2M1 codes of a, b (v_cvt_scalef32_pk_fp4_f32, scale 1) into byte ``sel`` of ``old``, a the low nibble."""
    return fx.Int32(
        _rocdl_ops.cvt_scalef32_pk_fp4_f32(
            T.i32,
            fx.as_ir_value(fx.Int32(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def unsign_zero(w):
    """Codes 0b1000 (-0) of the packed nibbles of ``w`` -> 0, as the reference keeps no sign on a zero code."""
    low = w & 0x77777777
    nonzero = (low + 0x77777777) & 0x88888888
    return w & (low | nonzero)


def group_scale(amax):
    """(biased exponent, 2^(127 - e)) of the group scale from its amax; amax * (1/6) picks the same exponent as
    amax / 6 for every bf16 amax (checked over all of them)."""
    s = maxnum(amax * f32(0x3E2AAAAB), f32(0x38D1B717))  # 1/6, 1e-4
    e = pow2_exponent(s)
    return e, pow2_pair(e)[1]


def ld16(rsrc, off):
    v = buffer_ops.buffer_load(rsrc, off, vec_width=1, dtype=T.i16)
    return fx.Int16(v).to(fx.Int32)


def key_load(lane, kr, lw, lb):
    """The wave's operands: the key pair of ``lane`` (``kr`` at the row), LayerNorm weight and bias pairs."""
    ld = lambda r: fx.Int32(buffer_ops.buffer_load(r, lane, vec_width=1, dtype=fx.Int32))  # noqa: E731
    return ld(kr), ld(lw), ld(lb)


def key_store(lane, ops, eps, slot, ok, cache_addr, page_b):
    """LayerNorm (fp32 statistics, bf16 weight / bias) of the wave's 128-column key, MXFP4, and the stores into page
    ``slot // 64`` at ``slot % 64`` of the cache at ``cache_addr`` (``ok`` masks them)."""
    xw, ww, bw = ops
    x0, x1 = (xw << 16).bitcast(fx.Float32), (xw & -65536).bitcast(fx.Float32)
    mean = wave_sum(x0 + x1) * fx.Float32(1.0 / D)
    c0, c1 = x0 - mean, x1 - mean
    var = wave_sum(fx.math.fma(c0, c0, c1 * c1)) * fx.Float32(1.0 / D)
    rstd = fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(var + eps)))
    w0, w1 = (ww << 16).bitcast(fx.Float32), (ww & -65536).bitcast(fx.Float32)
    b0, b1 = (bw << 16).bitcast(fx.Float32), (bw & -65536).bitcast(fx.Float32)
    yw = pk_bf16(fx.math.fma(w0, rstd * c0, b0), fx.math.fma(w1, rstd * c1, b1))
    y0, y1 = (yw << 16).bitcast(fx.Float32), (yw & -65536).bitcast(fx.Float32)
    amax = row_max(maxnum(fx.math.absf(y0), fx.math.absf(y1)))
    e, rescale = group_scale(amax)
    byte = unsign_zero(e2m1x2(y0 * rescale, y1 * rescale, fx.Int32(0), 0) & 0xFF)
    off = slot % 64
    cr = buffer_ops.create_buffer_resource_from_addr(
        cache_addr + fx.Int64(slot // 64) * page_b, num_records_bytes=PAGE_BYTES
    )
    grp = lane // 16
    buffer_ops.buffer_store(byte.to(fx.Int8), cr, grp * 1024 + off * 16 + lane % 16, mask=ok, offset_is_bytes=True)
    buffer_ops.buffer_store(
        e.to(fx.Int8),
        cr,
        4096 + grp * 64 + (off % 16) * 4 + off // 16,
        mask=ok & ((lane % 16) == 0),
        offset_is_bytes=True,
    )
