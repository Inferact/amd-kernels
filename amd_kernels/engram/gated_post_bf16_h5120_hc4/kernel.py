# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gated Engram update of the residual streams, one workgroup of 128 lanes (2 waves) per (token, hc copy), 40 columns
per lane as 5 vectors of 8 (lane l holds columns 8 l + 1024 v of vector v).

Per (token, hc): fp32 sums over the 5120 columns of hidden^2, key^2 and hidden * q * k * key (a lane's columns in
order, DPP wave folds, then the two waves through LDS), the signed-sqrt gate, and ``hidden + gate * value`` rounded
once to bf16 (an fma). The value is shared by the four copies; a workgroup reads and writes only its own (token, hc)
row of ``hidden``, so the output may alias it. 5 vectors per lane beat 1, 2 and 10 at 16-768 rows (measured).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

H = 5120
HC = 4
NT = H // 8  # 640 lanes
NW = NT // 64
KV_DW = (HC + 1) * H // 2  # dwords of a kv row
ROW_DW = H // 2
INV_SQRT_H = 1.0 / H**0.5
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _dpp_f(v, ctrl, row_mask=0xF, old=None):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(_bits(old))
    r = _rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(_bits(v)), ctrl, row_mask, 0xF, True)
    return fx.Int32(r).bitcast(fx.Float32)


def wave_sum_lane63(acc):
    """DPP wave sum, valid in lane 63: row_shr 8/4/2/1, row_bcast 15 and 31."""
    for sh in (8, 4, 2, 1):
        acc = acc + _dpp_f(acc, 0x110 + sh)
    acc = _dpp_f(acc, 0x142, row_mask=0xA, old=acc) + acc
    return acc + _dpp_f(acc, 0x143)


def bf16x8(v):
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (v_cvt_pk_bf16_f32, RNE), ``a`` low."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def xcd_row(bid, n):
    """(index, live) of workgroup ``bid``: it runs on XCD bid % 8, and XCD x takes the contiguous run of indices
    [x * (n // 8) + min(x, n % 8), ...), so the four copies of a token share an L2 (their value row)."""
    x, k = bid % 8, bid // 8
    q8, r8 = n // 8, n % 8
    extra = x < r8
    return x * q8 + extra.select(x, r8) + k, k < q8 + extra.select(fx.Int32(1), fx.Int32(0))


def compile_gated_post(has_mask, nv=1):
    """``nv`` vectors of 8 columns per lane: 640 / nv lanes per workgroup (``host.NV``)."""
    nt = NT // nv
    nw = nt // 64
    assert nt % 64 == 0
    name = "gated_post_bf16_h5120_hc4" + ("_mask" if has_mask else "") + f"_v{nv}"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, nw * 3]

    @flyc.kernel(name=name, known_block_size=[nt, 1, 1])
    def kernel(arg_h: fx.Int64, arg_kv: fx.Int64, arg_q: fx.Int64, arg_k: fx.Int64, arg_mask: fx.Int64,
               arg_out: fx.Int64, i32_t: fx.Int32, i32_eps: fx.Int32, i32_clamp: fx.Int32):
        tx = fx.thread_idx.x
        bid, live = xcd_row(fx.block_idx.x, i32_t * HC)
        t, hc = bid // HC, bid % HC
        if live:
            lane = tx % 64
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            hr = buffer_ops.create_buffer_resource_from_addr(arg_h, num_records_bytes=i32_t * (HC * H * 2))
            kvr = buffer_ops.create_buffer_resource_from_addr(arg_kv, num_records_bytes=i32_t * (KV_DW * 4))
            wq = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=HC * H * 2)
            wk = buffer_ops.create_buffer_resource_from_addr(arg_k, num_records_bytes=HC * H * 2)
            out = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i32_t * (HC * H * 2))

            def load(rsrc, base, v):
                return bf16x8(Vec(buffer_ops.buffer_load(rsrc, base + (tx + v * nt) * 4, vec_width=4, dtype=fx.Int32)))

            h = [load(hr, bid * ROW_DW, v) for v in range_constexpr(nv)]
            key = [load(kvr, t * KV_DW + hc * ROW_DW, v) for v in range_constexpr(nv)]
            q = [load(wq, hc * ROW_DW, v) for v in range_constexpr(nv)]
            k = [load(wk, hc * ROW_DW, v) for v in range_constexpr(nv)]
            val = [load(kvr, t * KV_DW + HC * ROW_DW, v) for v in range_constexpr(nv)]
            ss_h = h[0][0] * h[0][0]
            ss_k = key[0][0] * key[0][0]
            dot = h[0][0] * q[0][0] * k[0][0] * key[0][0]
            for v in range_constexpr(nv):
                for i in range_constexpr(8):
                    if const_expr(v > 0 or i > 0):
                        ss_h = fx.math.fma(h[v][i], h[v][i], ss_h)
                        ss_k = fx.math.fma(key[v][i], key[v][i], ss_k)
                        dot = fx.math.fma(h[v][i] * q[v][i], k[v][i] * key[v][i], dot)
            sums = [wave_sum_lane63(x) for x in (ss_h, ss_k, dot)]
            if const_expr(nw == 1):
                tot = [fx.Int32(fx.rocdl.readlane(T.i32, fx.as_ir_value(_bits(x)), 63)).bitcast(fx.Float32) for x in sums]
            else:
                wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
                if lane == 63:
                    for j in range_constexpr(3):
                        part[wave * 3 + j] = sums[j]
                fx.gpu.barrier()
                tot = []
                for j in range_constexpr(3):
                    acc = part[j]
                    for w in range_constexpr(1, nw):
                        acc = acc + part[w * 3 + j]
                    tot.append(acc)
            eps = i32_eps.bitcast(fx.Float32)
            h_rms = fx.math.rsqrt(tot[0] / fx.Float32(float(H)) + eps)
            k_rms = fx.math.rsqrt(tot[1] / fx.Float32(float(H)) + eps)
            d = tot[2] * h_rms * k_rms * fx.Float32(INV_SQRT_H)
            mag = fx.math.sqrt(_maxnum(fx.math.absf(d), i32_clamp.bitcast(fx.Float32)))
            gi = (d < fx.Float32(0.0)).select(fx.Float32(0.0) - mag, mag)
            gate = fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(fx.Float32(0.0) - gi))
            if const_expr(has_mask):
                mr = buffer_ops.create_buffer_resource_from_addr(arg_mask, num_records_bytes=i32_t)
                active = fx.Int32(buffer_ops.buffer_load(mr, t, vec_width=1, dtype=fx.Int8)) != 0
                gate = active.select(gate, fx.Float32(0.0))
            for v in range_constexpr(nv):
                y = [fx.math.fma(gate, val[v][i], h[v][i]) for i in range_constexpr(8)]
                buffer_ops.buffer_store(
                    Vec.from_elements([pk_bf16(y[2 * i], y[2 * i + 1]) for i in range_constexpr(4)], fx.Int32),
                    out, bid * ROW_DW + (tx + v * nt) * 4)

    @flyc.jit
    def launch_gated_post_bf16_h5120_hc4(arg_h: fx.Int64, arg_kv: fx.Int64, arg_q: fx.Int64, arg_k: fx.Int64,
                                         arg_mask: fx.Int64, arg_out: fx.Int64, i32_t: fx.Int32, i32_eps: fx.Int32,
                                         i32_clamp: fx.Int32, stream: fx.Stream):
        kernel(arg_h, arg_kv, arg_q, arg_k, arg_mask, arg_out, i32_t, i32_eps, i32_clamp).launch(
            grid=(fx.Int64((i32_t * HC + 7) // 8 * 8), 1, 1), block=(nt, 1, 1), stream=stream)

    launch_gated_post_bf16_h5120_hc4.kernel_name = name
    launch_gated_post_bf16_h5120_hc4.compile_hints = PRELOAD_HINTS
    return launch_gated_post_bf16_h5120_hc4
