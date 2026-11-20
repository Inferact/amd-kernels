# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two-rank all-reduce + residual add + Gemma RMSNorm (bf16 out, or MXFP8 with tiled / row-major codes), one-shot,
eager or in graphs.

Rows go in batches of ``rpb``. The first ``ns`` workgroups are stagers: batch k by stager k % ns, in order, each wave
writing its columns of the batch's partial rows into this rank's buffer, then the batch's flag into the peer's
(``common``; one per wave, coded rows one per batch). The other ``nr`` are reducers: batch k by reducer k % nr; each
wave loads its own partial and residual rows, polls the flag, reads the peer's staged rows over the link, sums them in
fp32 per row and runs ``add_rmsnorm``'s norm (12 waves, bf16 out) or ``add_rmsnorm_mxfp8``'s norm + quant (8 waves).
Stagers dispatch first and never wait on the peer, so the grid needs no residency guarantee, and a reducer's rows cross
the link while earlier batches are reduced. ``slots``: the partial is the MoE combine
``bf16((((s0 + s1) + s2) + s3) + shared)`` of this rank's [rows, 4, H] slots and [rows, H] shared output (stagers and
reducers both compute it: the same bits). The inputs are read where they are: no registration.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..add_rmsnorm import add_rmsnorm as _bf16_norm
from ..add_rmsnorm import add_rmsnorm_mxfp8 as _mx_norm
from ..add_rmsnorm.common import (
    PRELOAD_HINTS,
    H,
    bf16x8,
    pk_bf16,
    rsqrt_rn,
    seq_sum_sq,
    wave_sum_lane63,
)
from . import lossless_x3 as _x3
from .common import (
    COUNT_PAD,
    MAX_ROWS,
    ROWB,
    SYS,
    data_addr,
    flag_addr,
    flag_store,
    local_sum,
    spin,
    store_rowmajor,
    store_tiled,
    sum2,
)

SLOTS = 4
COUNT_WORDS = MAX_ROWS + COUNT_PAD  # per count array: stagers at 0, reducers after


def compile_oneshot_ar_add_rmsnorm(out: str = "bf16", slots: bool = False, rpb: int = 1, coded: bool = False):
    """``out``: "bf16", "tiled" or "rowmajor" (MXFP8 codes); ``rpb`` rows per batch (one flag per batch and wave);
    ``coded``: rows cross the link in ``lossless_x3``'s exact 12-bit code (0.781x the bytes, one flag per batch; a
    virtual wave of 512 values is the 8 columns of 64 lanes, so the MXFP8 lane map's second chunk is virtual waves
    8-11)."""
    B = rpb
    quant = out != "bf16"
    NW = 8 if quant else 12
    NT = 64 * NW
    NPK = 2 if quant else 1  # 16-byte packs per lane and row
    block_sum = _mx_norm._block_sum if quant else _bf16_norm._block_sum
    name = (
        ("sum_slots4_shared_" if slots else "")
        + "allreduce_oneshot_ranks2_gemma_add_rmsnorm"
        + (f"_mxfp8_{out}" if quant else "")
        + f"_h6144_b{B}"
        + ("_coded" if coded else "")
    )

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]
        hdr: fx.Array[fx.Int32, B * _x3.HDR_DW]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,  # stagers | reducers << 16
        i32_sx: fx.Int32,  # partial row stride (elements, a multiple of 8)
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_x: fx.Int64,  # [rows, H] bf16 partial, or [rows, 4, H] bf16 slots
        arg_sh: fx.Int64,  # [rows, H] bf16 shared-expert output (slots)
        arg_mine: fx.Int64,  # this rank's buffer
        arg_peer: fx.Int64,  # the peer's buffer, mapped here
        arg_r: fx.Int64,  # residual [rows, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_out: fx.Int64,  # [rows, H] bf16 out, or the e4m3 codes
        arg_s: fx.Int64,  # scales u8, AITER layout (MXFP8)
        arg_res: fx.Int64,  # residual out [rows, H] bf16
        arg_cnt: fx.Int64,  # int32 [2, COUNT_WORDS] call counts: stagers, reducers
        i32_eps: fx.Int32,  # fp32 bits
    ):
        if const_expr(quant):
            _mx_norm._set_fp16_ovfl()
        tx, b = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        rows = i32_rows
        nstage = i32_grid & 0xFFFF
        nred = i32_grid >> 16
        nb = (rows + (B - 1)) // B
        cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=2 * COUNT_WORDS * 4)
        has1 = tx < 256  # MXFP8: second chunk (columns 4096 + 8 t) inside the row
        offs = [tx * 4, tx * 4 + 2048]
        always = fx.Boolean(True)

        def ld(base, cm, ok):
            rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
            return [
                Vec(buffer_ops.buffer_load(rs, offs[h], vec_width=4, dtype=fx.Int32,
                                           mask=ok if h == 0 else ok & has1, cache_modifier=cm))
                for h in range_constexpr(NPK)
            ]

        def st(vs, base, ok, cm):
            rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
            for h in range_constexpr(NPK):
                buffer_ops.buffer_store(vs[h], rs, offs[h], mask=ok if h == 0 else ok & has1, cache_modifier=cm)

        def own_row(r, ok):
            if const_expr(slots):
                sbase = arg_x + fx.Int64(r) * (SLOTS * ROWB)
                sl = [ld(sbase + k * ROWB, 0, ok) for k in range_constexpr(SLOTS)]
                sh = ld(arg_sh + fx.Int64(r) * ROWB, 0, ok)
                return [local_sum([sl[k][h] for k in range_constexpr(SLOTS)], sh[h]) for h in range_constexpr(NPK)]
            return ld(arg_x + fx.Int64(r) * fx.Int64(i32_sx * 2), 0, ok)

        def count(base, r):
            return fx.Int32(buffer_ops.buffer_load(cr, base + r, vec_width=1, dtype=fx.Int32, is_scalar=True)) + 1

        def advance(base, first, step):
            """After the workgroup's last read: count + 1 for every row of its batches."""
            fx.gpu.barrier()
            for k in range(first, nb, step):
                r = k * B + tx
                m = (tx < B) & (r < rows)
                cur = fx.Int32(buffer_ops.buffer_load(cr, base + r, vec_width=1, dtype=fx.Int32, mask=m))
                buffer_ops.buffer_store(cur + 1, cr, base + r, mask=m)

        if b < nstage:
            for k in range(b, nb, nstage):
                r0 = k * B
                for i in range_constexpr(B):
                    r = r0 + i
                    ok = r < rows
                    e = count(0, r)
                    if const_expr(coded):
                        v = own_row(r, ok)
                        for h in range_constexpr(NPK):
                            _x3.enc_store(v[h], data_addr(arg_mine, r, e), data_addr(arg_mine, r, e, True),
                                          tx + 512 * h, ok if h == 0 else ok & has1)
                    else:
                        st(own_row(r, ok), data_addr(arg_mine, r, e), ok, SYS)
                fx.rocdl.s_waitcnt(vmcnt=0)
                if const_expr(coded):
                    # one flag per batch: a reader wave decodes with every wave's header
                    fx.gpu.barrier()
                    if tx == 0:
                        flag_store(flag_addr(arg_peer, r0, 0), count(0, r0))
                else:
                    # the wave's own stores landed: its flag for this batch
                    if lane == 0:
                        flag_store(flag_addr(arg_peer, r0, wave), count(0, r0))
            advance(0, b, nstage)
        else:
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            hdr = smem.hdr.ptr
            wv = [bf16x8(p) for p in ld(arg_w, 0, always)]
            for k in range(b - nstage, nb, nred):
                r0 = k * B
                oks = [(r0 + i) < rows for i in range_constexpr(B)]
                masks = [[oks[i] if h == 0 else oks[i] & has1 for h in range_constexpr(NPK)]
                         for i in range_constexpr(B)]
                owns = [own_row(r0 + i, oks[i]) for i in range_constexpr(B)]
                rvs = [ld(arg_r + fx.Int64(r0 + i) * fx.Int64(i32_sr * 2), 0, oks[i]) for i in range_constexpr(B)]
                # a poll waits for the wave's outstanding vector loads: the batch's own rows and residuals land first
                if const_expr(coded):
                    if tx == 0:
                        spin(flag_addr(arg_mine, r0, 0), count(COUNT_WORDS, r0))
                    fx.gpu.barrier()
                elif lane == 0:
                    spin(flag_addr(arg_mine, r0, wave), count(COUNT_WORDS, r0))
                es = [count(COUNT_WORDS, r0 + i) for i in range_constexpr(B)]
                if const_expr(coded):
                    recs = [[_x3.fetch(data_addr(arg_peer, r0 + i, es[i]), tx + 512 * h, masks[i][h])
                             for h in range_constexpr(NPK)] for i in range_constexpr(B)]
                    hvs = [_x3.fetch_headers(data_addr(arg_peer, r0 + i, es[i]), lane, oks[i] & (wave == 0))
                           for i in range_constexpr(B)]
                    if (wave == 0) & (lane < 24):
                        for i in range_constexpr(B):
                            for t in range_constexpr(4):
                                hdr[i * _x3.HDR_DW + 4 * lane + t] = hvs[i][t]
                    fx.gpu.barrier()
                    peers = [[_x3.finish(recs[i][h], [hdr[i * _x3.HDR_DW + 8 * (wave + 8 * h) + t]
                                                      for t in range_constexpr(8)],
                                         data_addr(arg_peer, r0 + i, es[i], True), tx + 512 * h, masks[i][h])
                              for h in range_constexpr(NPK)] for i in range_constexpr(B)]
                else:
                    peers = [ld(data_addr(arg_peer, r0 + i, es[i]), SYS, oks[i]) for i in range_constexpr(B)]
                for i in range_constexpr(B):
                    r = r0 + i
                    if r < rows:
                        own, peer, rv = owns[i], peers[i], rvs[i]
                        xv = [bf16x8(sum2(own[h], peer[h])) for h in range_constexpr(NPK)]
                        rvf = [bf16x8(rv[h]) for h in range_constexpr(NPK)]
                        s = [[xv[h][n] + rvf[h][n] for n in range_constexpr(8)] for h in range_constexpr(NPK)]
                        res = [
                            Vec.from_elements([pk_bf16(s[h][2 * n], s[h][2 * n + 1]) for n in range_constexpr(4)],
                                              fx.Int32)
                            for h in range_constexpr(NPK)
                        ]
                        st(res, arg_res + fx.Int64(r) * ROWB, always, 0)
                        sq = s[0] + s[1] if const_expr(quant) else s[0]
                        acc = wave_sum_lane63(seq_sum_sq(sq))
                        if lane == 63:
                            part[wave] = acc
                        fx.gpu.barrier()
                        rstd = rsqrt_rn(block_sum([part[n] for n in range_constexpr(NW)]),
                                        i32_eps.bitcast(fx.Float32))
                        ys = [[(s[h][n] * rstd) * (wv[h][n] + fx.Float32(1.0)) for n in range_constexpr(8)]
                              for h in range_constexpr(NPK)]
                        if const_expr(quant):
                            codes, sbs = [], []
                            for h in range_constexpr(2):
                                sb, rescale = _mx_norm._e8m0(ys[h])
                                codes.append(_mx_norm._codes(ys[h], rescale))
                                sbs.append(fx.Int32(sb.to(fx.Int32)))
                            q_bytes = (rows + 15) // 16 * 16 * H
                            s_bytes = (rows + 31) // 32 * 32 * (H // 32)
                            qr = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=q_bytes)
                            sr = buffer_ops.create_buffer_resource_from_addr(arg_s, num_records_bytes=s_bytes)
                            (store_tiled if out == "tiled" else store_rowmajor)(tx, r, codes, sbs, has1, always,
                                                                                qr, sr)
                        else:
                            y = Vec.from_elements([pk_bf16(ys[0][2 * n], ys[0][2 * n + 1])
                                                   for n in range_constexpr(4)], fx.Int32)
                            st([y], arg_out + fx.Int64(r) * ROWB, always, 0)
                        fx.gpu.barrier()
            advance(COUNT_WORDS, b - nstage, nred)

    @flyc.jit
    def launch_oneshot_ar_add_rmsnorm(
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
        i32_sx: fx.Int32,
        i32_sr: fx.Int32,
        arg_x: fx.Int64,
        arg_sh: fx.Int64,
        arg_mine: fx.Int64,
        arg_peer: fx.Int64,
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_out: fx.Int64,
        arg_s: fx.Int64,
        arg_res: fx.Int64,
        arg_cnt: fx.Int64,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows, i32_grid, i32_sx, i32_sr, arg_x, arg_sh, arg_mine, arg_peer, arg_r, arg_w, arg_out, arg_s,
            arg_res, arg_cnt, i32_eps,
        ).launch(grid=(fx.Int64((i32_grid & 0xFFFF) + (i32_grid >> 16)), 1, 1), block=(NT, 1, 1), stream=stream)

    launch_oneshot_ar_add_rmsnorm.kernel_name = name
    launch_oneshot_ar_add_rmsnorm.compile_hints = PRELOAD_HINTS
    return launch_oneshot_ar_add_rmsnorm
