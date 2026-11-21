# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""16-row MXFP4 x MXFP4 gate/up GEMM, SiTUv2 and FP4 quant of the intermediate (aiter weight and scale layouts).

One workgroup = 16 sorted rows x ``tn`` intermediate columns; a wave owns 32 columns of both halves for ``kw`` K slices
(k128 step s goes to K-wave s % kw), the K waves are summed in LDS in rank order. Weights are read straight from the
vLLM / aiter preshuffled tensors: W13 [E, 48 n16 tiles (gate 0-23, up 24-47), K/128, 4, 16, 16 B], scales
e8m0_shuffle'd [E * 768, 112].
"""
import functools
import flydsl.compiler as flyc
from amd_kernels.warm_up import launcher
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import E, H, I, cvt_fp4, e8m0_roundup, hold, lds_ptr, mfma, raw, situ
from .sync import READY_STRIDE, SC1, signal

KS = H // 128  # 28 k128 steps
W_EXPERT = 2 * I * (H // 2)  # bytes of one expert's W13
W_TILE = KS * 1024  # bytes of one n16 tile
GROUPS = 2 * I // 32  # 24 scale row groups per expert
KBLK = H // 32 // 8  # 14 scale k-blocks of 8 k32


@functools.cache
def compile_gemm1_tile(*, kw=4, pf=3, bm=16, signal_ready=False):
    """The tile function of GEMM1 (called inside a kernel with 256 threads) and its geometry.

    ``signal_ready``: store the rows write-through and count the tile in its m-block's ready counter (``ready``, the
    device address of the counters; ``READY_STRIDE`` bytes per block)."""
    assert kw in (2, 4) and bm in (16, 32) and KS % kw == 0
    nw = 4 // kw
    tn = nw * 32
    kt = KS // kw
    mr = bm // 16
    nblk = I // tn
    qkw = {"cache_modifier": SC1} if signal_ready else {}

    @flyc.jit
    def tile(base, rb, A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows, ready):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        nk, kk = wave % nw, wave // nw
        mb, nb = rb // nblk, rb % nblk
        nr = fx.Int32(NV[0])
        if mb * bm < nr:
            expert = fx.Int32(EIDS[mb])
            ng = nb * nw + nk  # 32-column group of this wave
            ar = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=m * (H // 2))
            sr = buffer_ops.create_buffer_resource(AS, max_size=False, num_records_bytes=m * (H // 32))
            wr = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=W_EXPERT,
                                                   base_byte_offset=fx.Int64(expert) * W_EXPERT)
            ws = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=E * GROUPS * KBLK * 256)
            qr = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=rows * (I // 2))
            ss = buffer_ops.create_buffer_resource(S, max_size=False, num_records_bytes=rows * (I // 32))
            tok = [fx.Int32(IDS[mb * bm + mi * 16 + l16]) & 0xFFFFFF for mi in range_constexpr(mr)]
            ad = [(tok[mi] * (H // 2) + kk * 64 + q16 * 16) // 4 for mi in range_constexpr(mr)]
            sd = [tok[mi] * (H // 32) // 4 + kk for mi in range_constexpr(mr)]
            # tile index of (ni, gate/up): gate ng * 2 + ni, up 24 + ng * 2 + ni
            wd = [[((gu * (I // 16) + ng * 2 + ni) * W_TILE + kk * 1024 + q16 * 256 + l16 * 16) // 4
                   for gu in range_constexpr(2)] for ni in range_constexpr(2)]
            swd = [((expert * GROUPS + gu * (I // 32) + ng) * KBLK + kk // 2) * 64 + q16 * 16 + l16
                   for gu in range_constexpr(2)]

            def load(k):
                aa = [buffer_ops.buffer_load(ar, ad[mi], vec_width=4, dtype=fx.Int32, soffset_bytes=k * kw * 64)
                      for mi in range_constexpr(mr)]
                sa = [(fx.Int32(buffer_ops.buffer_load(sr, sd[mi], vec_width=1, dtype=fx.Int32,
                                                    soffset_bytes=k * kw * 4)) >> (q16 * 8)) & 255
                      for mi in range_constexpr(mr)]
                bb = [[buffer_ops.buffer_load(wr, wd[ni][gu], vec_width=4, dtype=fx.Int32,
                                             soffset_bytes=k * kw * 1024, cache_modifier=2)
                       for gu in range_constexpr(2)] for ni in range_constexpr(2)]
                sb = [fx.Int32(buffer_ops.buffer_load(ws, swd[gu], vec_width=1, dtype=fx.Int32,
                                                      soffset_bytes=k * (kw // 2) * 256)) >> ((kk % 2) * 16)
                      for gu in range_constexpr(2)]
                return aa, sa, bb, sb

            acc = [[[None] * 2 for _ in range(2)] for _ in range(mr)]
            ring = [load(k) for k in range_constexpr(pf)]
            for k in range_constexpr(kt):
                if const_expr(k + pf < kt):
                    ring.append(load(k + pf))
                aa, sa, bb, sb = ring.pop(0)
                for mi in range_constexpr(mr):
                    for ni in range_constexpr(2):
                        for gu in range_constexpr(2):
                            acc[mi][ni][gu] = mfma(aa[mi], bb[ni][gu], acc[mi][ni][gu], sa[mi], sb[gu], ni, True)
            flat = hold([acc[mi][ni][gu] for mi in range(mr) for ni in range(2) for gu in range(2)])
            for j in range_constexpr(mr * 4):
                llvm.StoreOp(raw(flat[j]), lds_ptr(base, ((wave * mr * 4 + j) * 64 + lane) * 16), alignment=16)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            if kk == 0:
                vals = [Vec(v) for v in flat]
                for k in range_constexpr(1, kw):
                    for j in range_constexpr(mr * 4):
                        other = Vec(llvm.load(T.vec(4, T.f32),
                                              lds_ptr(base, (((nk + k * nw) * mr * 4 + j) * 64 + lane) * 16),
                                              alignment=16))
                        vals[j] = vals[j] + other
                for mi in range_constexpr(mr):
                    y = [[situ(fx.Float32(vals[mi * 4 + ni * 2][v]), fx.Float32(vals[mi * 4 + ni * 2 + 1][v]))
                          for v in range_constexpr(4)] for ni in range_constexpr(2)]
                    amax = fx.math.absf(y[0][0])
                    for ni in range_constexpr(2):
                        for v in range_constexpr(4):
                            z = fx.math.absf(y[ni][v])
                            amax = (amax > z).select(amax, z)
                    for sh in range_constexpr(2):
                        z = amax.shuffle_xor(16 << sh, 64)
                        amax = (amax > z).select(amax, z)
                    e = e8m0_roundup(amax)
                    scale = (e << 23).bitcast(fx.Float32)
                    row = mb * bm + mi * 16 + l16
                    for ni in range_constexpr(2):
                        pk = cvt_fp4(fx.Int32(0), y[ni][0], y[ni][1], scale, 0)
                        pk = cvt_fp4(pk, y[ni][2], y[ni][3], scale, 1)
                        off = row * (I // 2) + ng * 16 + ni * 8 + q16 * 2
                        buffer_ops.buffer_store(pk.to(fx.Int16), qr, off, offset_is_bytes=True, mask=tok[mi] < m, **qkw)
                    buffer_ops.buffer_store(e.to(fx.Int8), ss, row * (I // 32) + ng, offset_is_bytes=True,
                                            mask=(tok[mi] < m) & (q16 == 0), **qkw)
            if signal_ready:
                signal(ready + fx.Int64(mb * READY_STRIDE))

    return tile, nblk


@functools.cache
@launcher
def compile_gemm1(*, kw=4, pf=3, bm=16, name_suffix=""):
    tile, nblk = compile_gemm1_tile(kw=kw, pf=pf, bm=bm)
    mr = bm // 16
    name = f"moe_gemm1_a4w4_situv2_h{H}_i{I}_bm{bm}_kw{kw}_pf{pf}{name_suffix}"

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, 4 * mr * 4 * 64 * 16, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, Q: fx.Tensor, S: fx.Tensor,
               IDS: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor, m: fx.Int32, rows: fx.Int32):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        tile(base, fx.Int32(fx.block_idx.x), A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows, fx.Int64(0))

    @flyc.jit
    def launch(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, Q: fx.Tensor, S: fx.Tensor,
               IDS: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor, m: fx.Int32, rows: fx.Int32, stream: fx.Stream):
        kernel(A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows).launch(
            grid=(rows // bm * nblk, 1, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
