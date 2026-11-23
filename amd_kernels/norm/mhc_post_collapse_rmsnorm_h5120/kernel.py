# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Post-mix, stream collapse and RMSNorm of one row per workgroup, optionally with the stream mean.

640 lanes (10 waves) own 8 of the 5120 columns each, of all four streams: the post-mix, the collapse, the square sum
and the weighted normalization are lane-local except the sum, which is folded by DPP inside a wave and in a fixed order
across the waves through LDS. The bf16 roundings are those of the seam kernel of ``norm/mhc_post_pre_rmsnorm_h5120``:
the new residual ``R'`` and the collapse ``x1`` are rounded to bf16 before they are used.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

H = 5120
NS = 4
ROW_DW = H // 2
NTHR = H // 8
WAVES = NTHR // 64
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _dpp_f(v, ctrl, row_mask=0xF, old=None):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(_bits(old))
    r = _rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(_bits(v)), ctrl, row_mask, 0xF, True)
    return fx.Int32(r).bitcast(fx.Float32)


def _wave_sum_lane63(acc):
    """Row_shr 8/4/2/1 adds, row_bcast 15 / 31: the wave sum is valid in lane 63."""
    for sh in (8, 4, 2, 1):
        acc = acc + _dpp_f(acc, 0x110 + sh)
    acc = _dpp_f(acc, 0x142, row_mask=0xA, old=acc) + acc
    return acc + _dpp_f(acc, 0x143)


def _unpack8(v):
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _pack8(vals):
    return [
        fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(vals[2 * i])),
                                          fx.as_ir_value(fx.Float32(vals[2 * i + 1]))))
        for i in range(4)
    ]


def compile_mhc_post_collapse_rmsnorm(has_post, keep_residual, mean):
    name = "mhc_post_collapse_rmsnorm_h5120" + "".join(
        t for t, c in (("_post", has_post), ("_keep", keep_residual), ("_mean", mean)) if c
    )

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, WAVES]

    @flyc.kernel(name=name, known_block_size=[NTHR, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_eps: fx.Int32,
        arg_res: fx.Int64,  # [m, 4, H] bf16
        arg_y: fx.Int64,  # [m, H] bf16
        arg_pm: fx.Int64,  # [m, 4] f32
        arg_cm: fx.Int64,  # [m, 4, 4] f32
        arg_pre: fx.Int64,  # [m, 4] f32
        arg_nw: fx.Int64,  # [H] bf16
        arg_rout: fx.Int64,  # [m, 4, H] bf16
        arg_out: fx.Int64,  # [m, H] bf16
        arg_mean: fx.Int64,  # [m, H] bf16
    ):
        tx = fx.thread_idx.x
        row = fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        part = smem.part.ptr
        m64 = fx.Int64(i32_m)

        def rsrc(addr, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)

        def ldf(r, off, n):
            v = buffer_ops.buffer_load(r, off, vec_width=n, dtype=fx.Float32)
            return Vec(v) if n > 1 else fx.Float32(v)

        rr = rsrc(arg_res, m64 * (NS * H * 2))
        nwr = rsrc(arg_nw, H * 2)
        outr = rsrc(arg_out, m64 * (H * 2))
        qv = ldf(rsrc(arg_pre, m64 * 16), row * 4, 4)
        rs = [Vec(buffer_ops.buffer_load(rr, row * (NS * ROW_DW) + s * ROW_DW + tx * 4, vec_width=4, dtype=fx.Int32))
              for s in range_constexpr(NS)]
        w = _unpack8(Vec(buffer_ops.buffer_load(nwr, tx * 4, vec_width=4, dtype=fx.Int32)))
        if fx.const_expr(has_post):
            yv = _unpack8(Vec(buffer_ops.buffer_load(rsrc(arg_y, m64 * (H * 2)), row * ROW_DW + tx * 4, vec_width=4,
                                                     dtype=fx.Int32)))
            pv = ldf(rsrc(arg_pm, m64 * 16), row * 4, 4)
            cmr = rsrc(arg_cm, m64 * 64)
            cv = [ldf(cmr, row * 16 + 4 * h, 4) for h in range_constexpr(NS)]
            f = [_unpack8(r) for r in rs]
            pk = []
            for j in range_constexpr(NS):
                vals = []
                for e in range_constexpr(8):
                    a = pv[j] * yv[e]
                    for h in range_constexpr(NS):
                        a = fx.math.fma(cv[h][j], f[h][e], a)
                    vals.append(a)
                pk.append(_pack8(vals))
            if fx.const_expr(keep_residual):
                ror = rsrc(arg_rout, m64 * (NS * H * 2))
                for j in range_constexpr(NS):
                    buffer_ops.buffer_store(Vec.from_elements(pk[j], fx.Int32), ror,
                                            row * (NS * ROW_DW) + j * ROW_DW + tx * 4)
        else:
            pk = [[fx.Int32(r[k]) for k in range(4)] for r in rs]
        gf = [_unpack8(p) for p in pk]
        if fx.const_expr(mean):
            mv = [((gf[0][e] + gf[1][e]) + gf[2][e] + gf[3][e]) * fx.Float32(0.25) for e in range_constexpr(8)]
            buffer_ops.buffer_store(Vec.from_elements(_pack8(mv), fx.Int32), rsrc(arg_mean, m64 * (H * 2)),
                                    row * ROW_DW + tx * 4)
        x1v = []
        for e in range_constexpr(8):
            a = qv[0] * gf[0][e]
            for s in range_constexpr(1, NS):
                a = fx.math.fma(qv[s], gf[s][e], a)
            x1v.append(a)
        x1 = _unpack8(_pack8(x1v))
        acc = fx.Float32(0.0)
        for e in range_constexpr(8):
            acc = fx.math.fma(x1[e], x1[e], acc)
        acc = _wave_sum_lane63(acc)
        if lane == 63:
            part[wave] = acc
        gpu.barrier()
        ss = part[0]
        for i in range_constexpr(1, WAVES):
            ss = ss + part[i]
        eps = i32_eps.bitcast(fx.Float32)
        rstd = fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(ss * fx.Float32(1.0 / H) + eps)))
        yp = _pack8([(x1[e] * rstd) * w[e] for e in range_constexpr(8)])
        buffer_ops.buffer_store(Vec.from_elements(yp, fx.Int32), outr, row * ROW_DW + tx * 4)

    @flyc.jit
    def launch(
        arg_res: fx.Int64,
        arg_y: fx.Int64,
        arg_pm: fx.Int64,
        arg_cm: fx.Int64,
        arg_pre: fx.Int64,
        arg_nw: fx.Int64,
        arg_rout: fx.Int64,
        arg_out: fx.Int64,
        arg_mean: fx.Int64,
        i32_m: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(i32_m, i32_eps, arg_res, arg_y, arg_pm, arg_cm, arg_pre, arg_nw, arg_rout, arg_out, arg_mean).launch(
            grid=(fx.Int64(i32_m), 1, 1), block=(NTHR, 1, 1), stream=stream
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
