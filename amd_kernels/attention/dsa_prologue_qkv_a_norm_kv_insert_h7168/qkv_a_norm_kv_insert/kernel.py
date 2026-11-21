# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DSA decode prologue, FlyDSL (gfx950): the q_a and kv_a RMS norms of the qkv_a projection row and the fp8_ds_mla
KV-cache insert, one launch.

One workgroup of 4 waves per token. A row is [q 1536 | kv 512 | k_pe 64] bf16 and the first 2048 columns map one to
one onto 256 lanes of 8 columns: waves 0-2 hold q, wave 3 holds kv. The squares are summed in fp32 (a DPP fold per
wave, the three q waves added through LDS); ``y = RN_bf16(x * rsqrt(sum / n + eps) * w)``, one rounding as AITER's
norm does. Wave 3 then takes the max of ``|y|`` over each 16-lane group (one 128-column tile), the tile scale is the
smallest power of two >= max(amax / 448, 1e-4), and its 8 columns go into the cache record as e4m3 of ``y / scale``:
[512 e4m3 | 4 fp32 scales | 64 bf16 k_pe] = 656 bytes at ``slot``. A slot < 0 (a token this DCP rank does not
own, or padding) writes no cache byte; the norm outputs are written for every row.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..index_key import key_load, key_store, ld16
from ..common import PRELOAD_HINTS, bf16x8, f32, maxnum, pk_bf16, pow2_exponent, pow2_pair, row_max, tree_absmax, wave_allsum as wave_allsum_replicated

Q = 1536
KV = 512
PE = 64
ROW = Q + KV + PE
RECORD = 656
D = 128
HEADS = 32
NT = 256  # waves 0-2: q columns 512 each, wave 3: kv


def _cvt_pk_fp8(old, a, b, hi: bool):
    """v_cvt_scalef32_pk_fp8_f32 (scale 1): a, b -> two e4m3 bytes in the low / high half of ``old``."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def _dot2_bf16(a, b, acc):
    """acc + a.lo * b.lo + a.hi * b.hi of two dwords of bf16 pairs (v_dot2_f32_bf16)."""
    bf2 = _ir.VectorType.get([2], _ir.BF16Type.get())
    return fx.Float32(
        _rocdl_ops.fdot2_f32_bf16_(
            T.f32,
            _llvm.bitcast(bf2, fx.as_ir_value(a)),
            _llvm.bitcast(bf2, fx.as_ir_value(b)),
            fx.as_ir_value(acc),
        )
    )


def compile_qkv_a_norm_kv_insert(insert: bool, index: bool = False):
    name = "dsa_qkv_a_norm_kv_insert_h7168" + ("" if insert else "_noinsert") + ("_index" if index else "")
    nthreads = NT + (128 if index else 0)

    @flyc.kernel(name=name, known_block_size=[nthreads, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_nsl: fx.Int32,
        i32_row_st: fx.Int32,
        i32_tok_b: fx.Int32,
        i32_eps: fx.Int32,
        i64_page_b: fx.Int64,
        arg_x: fx.Int64,  # [n, row stride] bf16, rows of ROW
        arg_qw: fx.Int64,  # [1536] bf16
        arg_kw: fx.Int64,  # [512] bf16
        arg_q: fx.Int64,  # q_out [n, 1536] bf16
        arg_kv: fx.Int64,  # kv_out [n, 512] bf16
        arg_cache: fx.Int64,  # fp8_ds_mla pages, 64-token blocks of 656-byte records
        arg_slot: fx.Int64,  # slots [nsl] i64, < 0 skips
        i32_wscale: fx.Int32,
        i32_nisl: fx.Int32,
        i64_ipage_b: fx.Int64,
        arg_lw: fx.Int64,  # indexer key LayerNorm weight [128] bf16
        arg_lb: fx.Int64,
        arg_icache: fx.Int64,  # indexer pages [pages, 64, 68] u8
        arg_islot: fx.Int64,  # indexer slots [nisl] i64, < 0 skips
        arg_wo: fx.Int64,  # index head weights [n, 32] fp32
    ):
        lane = fx.thread_idx.x % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.thread_idx.x // 64))
        row = fx.block_idx.x
        if wave < 4:
            is_kv = wave == 3
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x + fx.Int64(row) * fx.Int64(i32_row_st * 2), num_records_bytes=ROW * 2
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                is_kv.select(arg_kw, arg_qw), num_records_bytes=Q * 2
            )
            if const_expr(insert):
                sl_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_slot, num_records_bytes=fx.Int64(i32_nsl) * 8
                )
                slot = fx.Int32(
                    buffer_ops.buffer_load(sl_r, row * 2, vec_width=1, dtype=fx.Int32, is_scalar=True)
                )
                pe = buffer_ops.buffer_load(
                    xr, (Q + KV) // 2 + lane * 4, vec_width=4, dtype=fx.Int32, mask=is_kv & (lane < 8)
                )
            u = [
                Vec(
                    buffer_ops.buffer_load(
                        xr,
                        (is_kv.select(fx.Int32(Q // 8 + 0), fx.Int32(64 * i)) + lane) * 4 if i == 0 else (64 * i + lane) * 4,
                        vec_width=4,
                        dtype=fx.Int32,
                        mask=None if i == 0 else wave < 3,
                    )
                )
                for i in range_constexpr(3)
            ]
            wv = Vec(buffer_ops.buffer_load(wr, is_kv.select(lane, wave * 64 + lane) * 4, vec_width=4, dtype=fx.Int32))
            acc = fx.Float32(0.0)
            for i in range_constexpr(3):
                for j in range_constexpr(4):
                    acc = _dot2_bf16(u[i][j], u[i][j], acc)
            ss = wave_allsum_replicated(acc)
            inv_n = is_kv.select(fx.Float32(1.0 / KV), fx.Float32(1.0 / Q))
            rr = fx.Float32(fx.rocdl.rsq(T.f32, fx.as_ir_value(ss * inv_n + i32_eps.bitcast(fx.Float32))))
            sel1, sel2 = wave == 1, wave == 2
            xv = bf16x8(
                Vec.from_elements(
                    [sel1.select(u[1][j], sel2.select(u[2][j], u[0][j])) for j in range_constexpr(4)], fx.Int32
                )
            )
            y = [(xv[j] * rr) * w_ for j, w_ in enumerate(bf16x8(wv))]
            packed = [pk_bf16(y[2 * j], y[2 * j + 1]) for j in range_constexpr(4)]
            out = buffer_ops.create_buffer_resource_from_addr(
                is_kv.select(arg_kv + fx.Int64(row) * (KV * 2), arg_q + fx.Int64(row) * (Q * 2)),
                num_records_bytes=Q * 2,
            )
            buffer_ops.buffer_store(
                Vec.from_elements(packed, fx.Int32), out, is_kv.select(lane, wave * 64 + lane) * 4
            )
            if const_expr(insert):
                yb = bf16x8(Vec.from_elements(packed, fx.Int32))
                amax = row_max(tree_absmax(yb))
                s = maxnum(amax * f32(0x3B124925), f32(0x38D1B717))  # 1/448, 1e-4
                scale, rescale = pow2_pair(pow2_exponent(s))
                q8 = [v * rescale for v in yb]
                lo = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q8[0], q8[1], False), q8[2], q8[3], True)
                hi = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q8[4], q8[5], False), q8[6], q8[7], True)
                ok = (slot >= 0) & (row < i32_nsl) & is_kv
                rec = (
                    arg_cache
                    + fx.Int64(slot // 64) * i64_page_b
                    + fx.Int64((slot % 64) * i32_tok_b)
                )
                cr = buffer_ops.create_buffer_resource_from_addr(rec, num_records_bytes=RECORD)
                buffer_ops.buffer_store(Vec.from_elements([lo, hi], fx.Int32), cr, lane * 2, mask=ok)
                buffer_ops.buffer_store(scale, cr, KV // 4 + lane // 16, mask=ok & ((lane % 16) == 0))
                buffer_ops.buffer_store(pe, cr, (KV + 16) // 4 + lane * 4, mask=ok & (lane < 8))

        if const_expr(index):
            if wave == 4:
                isl_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_islot, num_records_bytes=fx.Int64(i32_nisl) * 8
                )
                islot = fx.Int32(
                    buffer_ops.buffer_load(isl_r, row * 2, vec_width=1, dtype=fx.Int32, is_scalar=True)
                )
                kr = buffer_ops.create_buffer_resource_from_addr(
                    arg_x + fx.Int64(row) * fx.Int64(i32_row_st * 2) + ROW * 2, num_records_bytes=D * 2
                )
                lw = buffer_ops.create_buffer_resource_from_addr(arg_lw, num_records_bytes=D * 2)
                lb = buffer_ops.create_buffer_resource_from_addr(arg_lb, num_records_bytes=D * 2)
                key_store(
                    lane, key_load(lane, kr, lw, lb), i32_eps.bitcast(fx.Float32), islot,
                    (islot >= 0) & (row < i32_nisl), arg_icache, i64_ipage_b,
                )
            if wave == 5:
                if lane < 32:
                    wr2 = buffer_ops.create_buffer_resource_from_addr(
                        arg_x + fx.Int64(row) * fx.Int64(i32_row_st * 2) + (ROW + D) * 2, num_records_bytes=HEADS * 2
                    )
                    wv2 = (ld16(wr2, lane) << 16).bitcast(fx.Float32)
                    wo = buffer_ops.create_buffer_resource_from_addr(
                        arg_wo, num_records_bytes=fx.Int64(i32_n) * (HEADS * 4)
                    )
                    buffer_ops.buffer_store(wv2 * i32_wscale.bitcast(fx.Float32), wo, row * HEADS + lane)

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_cache: fx.Int64,
        arg_slot: fx.Int64,
        i32_n: fx.Int32,
        i32_nsl: fx.Int32,
        i32_row_st: fx.Int32,
        i32_tok_b: fx.Int32,
        i32_eps: fx.Int32,
        i64_page_b: fx.Int64,
        arg_lw: fx.Int64,
        arg_lb: fx.Int64,
        arg_icache: fx.Int64,
        arg_islot: fx.Int64,
        arg_wo: fx.Int64,
        i32_wscale: fx.Int32,
        i32_nisl: fx.Int32,
        i64_ipage_b: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            i32_n, i32_nsl, i32_row_st, i32_tok_b, i32_eps, i64_page_b,
            arg_x, arg_qw, arg_kw, arg_q, arg_kv, arg_cache, arg_slot,
            i32_wscale, i32_nisl, i64_ipage_b, arg_lw, arg_lb, arg_icache, arg_islot, arg_wo,
        ).launch(grid=(i32_n, 1, 1), block=(nthreads, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
