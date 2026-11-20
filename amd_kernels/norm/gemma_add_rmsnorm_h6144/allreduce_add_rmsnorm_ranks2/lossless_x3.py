# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``lossless_bf16``'s exact 12-bit code in 12-byte lane records, for readers across the link.

The code is ``allreduce_add_rmsnorm/lossless_bf16``'s (per 32 values a group exponent, per value sign, 7 mantissa bits
and a 4-bit exponent delta; zeros, exceptions and raw virtual waves as there); only the layout changes. Lane t's 8
sign/mantissa bytes and its 8 delta nibbles are one record at byte 12 t, then 12 headers of 32 B at 9216 (16 group
exponent bytes, the count word, 6 exception values): 9600 B = 0.781x the bf16 row. A reader fetches a record with one
dwordx3 and all 12 headers with one wave's 384-byte load, shared through LDS: whole-line reads (per-wave 32-byte header
reads cost ~10% of the link, which is request-bound). Raw virtual waves go to the caller's raw slot (1 KB at 1024 v).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..allreduce_add_rmsnorm.lossless_bf16 import RAWF, SPECF, _dec_fast, _dec_fix, enc
from .common import ROWB, SYS

HDR = 9216  # header bytes of a coded row
HDR_DW = 96  # header dwords of a row


@flyc.jit
def enc_store(value, base, raw, tid, mask):
    """Write one lane's 8 values of a coded row (or its virtual wave's raw fallback), without publishing."""
    lane, wave = tid % 64, tid // 64
    a0, a1, nib, gmax, cwv, ev, idx, has, rawi = enc(value, lane)
    rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
    rr = buffer_ops.create_buffer_resource_from_addr(raw, num_records_bytes=ROWB)
    buffer_ops.buffer_store(value, rr, tid * 4, mask=mask & (rawi != 0), cache_modifier=SYS)
    buffer_ops.buffer_store(Vec.from_elements([a0, a1, nib], fx.Int32), rs, tid * 3, mask=mask, cache_modifier=SYS)
    buffer_ops.buffer_store(gmax.to(fx.Int8), rs, HDR + 32 * wave + lane // 4, mask=mask & (lane % 4 == 0),
                            cache_modifier=SYS, offset_is_bytes=True)
    buffer_ops.buffer_store(cwv, rs, HDR // 4 + 8 * wave + 4, mask=mask & (lane == 0), cache_modifier=SYS)
    buffer_ops.buffer_store(ev.to(fx.Int16), rs, HDR + 32 * wave + 20 + 2 * (idx - 1),
                            mask=mask & has & (rawi == 0), cache_modifier=SYS, offset_is_bytes=True)


def fetch(base, tid, mask):
    """A lane's record (3 dwords); issue every row's before decoding any."""
    rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
    rec = Vec(buffer_ops.buffer_load(rs, tid * 3, vec_width=3, dtype=fx.Int32, mask=mask, cache_modifier=SYS))
    return [fx.Int32(rec[t]) for t in range_constexpr(3)]


def fetch_headers(base, lane, mask):
    """Lane l < 24's 4 header dwords 4 l .. 4 l + 3 of a row (``mask``: the loading wave's lanes)."""
    rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
    hv = Vec(buffer_ops.buffer_load(rs, HDR // 4 + 4 * lane, vec_width=4, dtype=fx.Int32, mask=mask & (lane < 24),
                                    cache_modifier=SYS))
    return [fx.Int32(hv[t]) for t in range_constexpr(4)]


@flyc.jit
def finish(rec, hw, raw, tid, mask):
    """Decode a lane's four packed bf16 pairs from its record and its virtual wave's 8 header dwords ``hw``; a raw
    virtual wave reads the caller's raw slot. Masked lanes return zeros (their ``hw`` may be any LDS words)."""
    lane = tid % 64
    pw, pb = rec[:2], rec[2]
    cw = fx.Int32(fx.rocdl.readfirstlane(T.i32, hw[4]))
    d0, d1, d2, d3 = _dec_fast(pw, pb, hw, lane)
    if (cw & SPECF) != 0:
        d0, d1, d2, d3 = _dec_fix([d0, d1, d2, d3], pw, pb, hw)
    if (cw & RAWF) != 0:
        rr = buffer_ops.create_buffer_resource_from_addr(raw, num_records_bytes=ROWB)
        r = Vec(buffer_ops.buffer_load(rr, tid * 4, vec_width=4, dtype=fx.Int32, mask=mask, cache_modifier=SYS))
        d0, d1, d2, d3 = [fx.Int32(r[t]) for t in range_constexpr(4)]
    return Vec.from_elements([mask.select(d, fx.Int32(0)) for d in (d0, d1, d2, d3)], fx.Int32)
