# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""BF16 [M, 3584] -> MXFP4 codes [M, 1792] + e8m0 scales [M, 112] (row-major), the activation format of GEMM1: one
thread per (row, 32-column group), aiter's rounding (ceil exponent of amax / 6 floored at 92, RNE e2m1, saturating)."""
import functools
import flydsl.compiler as flyc
from amd_kernels.warm_up import launcher
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from .common import H, cvt_fp4, e8m0_roundup

G = H // 32  # 112 groups per row
E8M0_FLOOR = 92  # aiter's activation quant never goes below 2^-35 (all-zero groups)


@functools.cache
@launcher
def compile_quant(*, name_suffix=""):
    name = f"moe_quant_bf16_mxfp4_h{H}{name_suffix}"

    @flyc.kernel(name=name, known_block_size=[G * 2, 1, 1])
    def kernel(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, m: fx.Int32):
        tx = fx.thread_idx.x
        row = fx.Int32(fx.block_idx.x) * 2 + tx // G
        g = tx % G
        if row < m:
            xr = buffer_ops.create_buffer_resource(X, max_size=False, num_records_bytes=m * H * 2)
            qr = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=m * (H // 2))
            sr = buffer_ops.create_buffer_resource(S, max_size=False, num_records_bytes=m * G)
            vs = [Vec(buffer_ops.buffer_load(xr, (row * H + g * 32) // 2 + c * 4, vec_width=4, dtype=fx.Int32))
                  for c in range_constexpr(4)]
            v = []
            for c in range_constexpr(4):
                for i in range_constexpr(4):
                    w = fx.Int32(vs[c][i])
                    v.append((w << 16).bitcast(fx.Float32))
                    v.append((w & -65536).bitcast(fx.Float32))
            amax = fx.math.absf(v[0])
            for i in range_constexpr(1, 32):
                z = fx.math.absf(v[i])
                amax = (amax > z).select(amax, z)
            e = e8m0_roundup(amax)
            e = (e > E8M0_FLOOR).select(e, fx.Int32(E8M0_FLOOR))
            scale = (e << 23).bitcast(fx.Float32)
            pk = []
            for w in range_constexpr(4):
                a = cvt_fp4(fx.Int32(0), v[8 * w], v[8 * w + 1], scale, 0)
                a = cvt_fp4(a, v[8 * w + 2], v[8 * w + 3], scale, 1)
                a = cvt_fp4(a, v[8 * w + 4], v[8 * w + 5], scale, 2)
                a = cvt_fp4(a, v[8 * w + 6], v[8 * w + 7], scale, 3)
                pk.append(a)
            buffer_ops.buffer_store(Vec.from_elements(pk, fx.Int32), qr, (row * (H // 2) + g * 16) // 4)
            buffer_ops.buffer_store(e.to(fx.Int8), sr, row * G + g, offset_is_bytes=True)

    @flyc.jit
    def launch(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, m: fx.Int32, stream: fx.Stream):
        kernel(X, Q, S, m).launch(grid=((m + 1) // 2, 1, 1), block=(G * 2, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
