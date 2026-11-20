# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""65-128 packed rows: the 64-row kernel's tile (four waves, 128 vocabulary columns per block) run over two 64-row
halves with the weight registers loaded once. Holding all 128 rows' activations would take 256 VGPRs per lane, so
each half loads its activations, runs its MFMAs and writes through the LDS transpose before the next half."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T, Vector as Vec
from .reference import N as V

TN = 128
HALVES = 2
MH = 4  # 16-row fragments per half


def compile_bf16_gemm_packed_rows_m128():
    NR = TN // 64

    @fx.struct
    class Shared:
        out: fx.Array[fx.Int32, MH * 16 * TN // 2, 16]

    @flyc.kernel(name=f'bf16_gemm_packed_rows_n200064_k256_mt8_tn{TN}', known_block_size=[256, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, i32_n: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n0 = bid * TN + wave * (TN // 4)
        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=HALVES * MH * 8192)
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=V * 256 * 2)
        yr = buffer_ops.create_buffer_resource_from_addr(arg_y, num_records_bytes=i32_n * V * 2)
        wv = [[Vec(buffer_ops.buffer_load(wr, (n0 + rn * 16 + lane % 16) * 128 + k * 16 + (lane // 16) * 4,
                                          vec_width=4, dtype=fx.Int32)) for k in range_constexpr(8)]
              for rn in range_constexpr(NR)]
        lds16 = fx.logical_divide(fx.make_view(fx.recast_iter(fx.Int32, smem.out.ptr),
                                               fx.make_layout(MH * 16 * TN // 2, 1)), fx.make_layout(4, 1))
        atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        for h in range_constexpr(HALVES):
            xv = [[Vec(buffer_ops.buffer_load(xr, (h * MH + m) * 2048 + k * 256 + lane * 4, vec_width=4,
                                              dtype=fx.Int32, mask=(lane % 16 + (h * MH + m) * 16) < i32_n))
                   for k in range_constexpr(8)] for m in range_constexpr(MH)]
            acc = [[Vec.filled(4, 0.0, fx.Float32) for m in range_constexpr(MH)] for rn in range_constexpr(NR)]
            for k in range_constexpr(8):
                for rn in range_constexpr(NR):
                    for m in range_constexpr(MH):
                        acc[rn][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                            Vec.make_type(4, fx.Float32),
                            [wv[rn][k].bitcast(fx.BFloat16), xv[m][k].bitcast(fx.BFloat16), acc[rn][m], 0, 0, 0]))
            if h:
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()  # every wave has read the first half's tile
            for rn in range_constexpr(NR):
                for m in range_constexpr(MH):
                    r = lane % 16 + m * 16
                    for e in range_constexpr(2):
                        pk = fx.Int32(Vec.from_elements([fx.Float32(acc[rn][m][2 * e]),
                                                         fx.Float32(acc[rn][m][2 * e + 1])],
                                                        fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0])
                        c = wave * (TN // 4) + rn * 16 + (lane // 16) * 4 + e * 2
                        phys = (r * (TN // 2) + c // 2) ^ ((r % 8) * 4)
                        smem.out.ptr[phys] = pk
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            for it in range_constexpr((MH * 16 * TN // 8 + 255) // 256):
                v = tx + it * 256
                row = v // (TN // 8)
                col = (v % (TN // 8)) * 8
                if v < MH * 16 * TN // 8:
                    phys = (row * (TN // 2) + col // 2) ^ ((row % 8) * 4)
                    rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(atom, fx.slice(lds16, (None, phys // 4)), rm)
                    val = Vec(fx.memref_load_vec(rm))
                    c = bid * TN + col
                    rr = row + h * MH * 16
                    buffer_ops.buffer_store(val, yr, rr * (V // 2) + c // 2, mask=(rr < i32_n) & (c < V))

    @flyc.jit
    def launch_bf16_gemm_packed_rows_m128(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, i32_n: fx.Int32,
                                          stream: fx.Stream):
        kernel(arg_x, arg_w, arg_y, i32_n).launch(grid=((V + TN - 1) // TN, 1, 1), block=(256, 1, 1), stream=stream)
    return launch_bf16_gemm_packed_rows_m128
