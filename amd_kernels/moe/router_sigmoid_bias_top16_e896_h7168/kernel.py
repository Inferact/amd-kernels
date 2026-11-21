# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 [M, 7168] x bf16 [896, 7168]^T -> fp32 split-K partials [KS, M, 896] of the MoE router logits, 1-192 rows.

Tiles: 14 column tiles of 64 weight rows (wave w owns the 16-row group w) x KS K slices of 112 / KS blocks of 64
(grid 14 KS). A workgroup covers all M rows: both operands go HBM / L2 -> LDS by asynchronous `buffer_load ... lds`
into a ring of R slots (one slot = the 4 weight fragments + MT activation fragments, 2 KB each), with the XOR-swizzled
fragment reads and the half-block software pipeline of gemm/bf16_n50016_k6144 (v_mfma_f32_16x16x32_bf16, K ascending).
Weight loads are non-temporal: the weight streams through.

Epilogue: each workgroup stores its fp32 accumulators to the partial slab [KS, M, 896] (plain stores: the
top-k launch that follows sums the KS slabs in slice order, so the logits' bits do not depend on which slice ends
last). No tickets, no atomics.

Fused (the whole router in one launch): each slab word's lowest mantissa bit carries the call's parity (no arrival
flag: the previous call's words, never an older one's, carry the other parity). Workgroup t < M takes token t: its four
waves load the token's slab words (one 256-expert group each) until every word has this call's parity, then the
top-16 (``topk.py``) and, with ``aux``, the sort (``aux.py``) follow.

Weight layouts: ROW (the checkpoint's [896, 7168]) or SHUF ([56 groups, 112 K blocks, 16 rows, 64 K]: one DMA reads
1 KB in one piece, a bit-exact permutation made once at load).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

from .reference import E as N, H as K
from .aux import (EPOCH_BASE, IDS_WORDS, NE, aux_counts, aux_expert_blocks, aux_max_blocks, aux_pair_row, aux_prefill, aux_publish_cum,
                  aux_zero_masks, lds_or, mask_words)
from .topk import (LDS_WORDS, MAXC, NQ, RANK_UNROLL, TOPK, _ballot, _popcount, topk_fast_write, topk_new, topk_pivot, topk_q_keys,
                   topk_rank_start, topk_rank_step, topk_slow_out, topk_slow_select,
                   topk_finish, topk_tie_masks)

SC1 = 0x10  # agent-scope (cross-XCD) stores and loads of the slabs and counters
FLAG_WORDS = 256  # one flag word per workgroup (a call advances each by one)
POLL_SLEEP = 1  # s_sleep units (64 clocks) between two polls of the slabs
NB = K // 64  # 112 K blocks
NWV = 4
W_CM = 2  # nt
MT_MAX = 12
RING_BYTES = 140 * 1024


def ring_depth(frags: int) -> int:
    """Ring slots that fit the LDS next to the top-k arrays (one slot = ``frags`` fragments of 2 KB)."""
    return 4 if frags * 2048 * 4 <= RING_BYTES else 3


def tiles(nf: int) -> int:
    """Column tiles of ``nf`` 16-column fragments."""
    return N // 16 // nf


def row_fragments(mt: int, wn: int, ms: int) -> int:
    """Row fragments of one workgroup: the ``mt`` fragments split over ``ms`` workgroups, rounded up to the row waves."""
    wm = NWV // wn
    return (-(-mt // ms) + wm - 1) // wm * wm


def slab_rows(mt: int, wn: int, ms: int = 1) -> int:
    """Rows of a partial slab: every row of every workgroup's row fragments is stored (rows past M repeat the last token),
    so a call overwrites every word of its slabs."""
    return 16 * ms * row_fragments(mt, wn, ms)


def partial_bytes(mt: int, ks: int, wn: int, ms: int = 1) -> int:
    return ks * slab_rows(mt, wn, ms) * N * 4


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


@launcher
def compile_router_e896_h7168(mt: int, ks: int, shuf: int, fused: int = 1, wn: int = 4, nrw: int = 1, aux: int = 0, ts: int = 0, skel: int = 0, ms: int = 1):
    """``fused`` 0: the GEMM launch alone (partials out); 1: the whole router (the top-16 follows in the same launch).
    ``ms`` row splits: the row fragments are shared by ``ms`` workgroups per (column tile, K slice), each streams the same
    weights (an L2 hit: the pair sits on one XCD) and the slabs of fewer K splits carry the same bytes per row.
    The workgroup tile is ``wn nrw`` 16-column weight fragments; its ``wn`` x ``4 / wn`` waves split the columns
    (``nrw`` fragments each) and the row fragments (``MTP / (4 / wn)`` each). ``aux`` 1: the launch also writes the BM=16
    MoE sort of the top-16 and zeroes the moe buffer (``aux.py``)."""
    MT, KS, SHUF, FUSED, WN, NRW, AUX, TS, SKEL, MS = mt, ks, shuf, fused, wn, nrw, aux, ts, skel, ms
    # ``skel`` bits (floor measurements, results meaningless): 1 no LDS reads / MFMAs; 2 no top-16 after the slab gather;
    # 4 no activation DMAs; 8 no consumer; 16 no slab stores
    # ``ts``: the reverse_sorted (moe_buf with ``aux``) argument is a buffer of per-workgroup realtime stamps (16 int64 each)
    assert not AUX or FUSED
    WM = NWV // WN
    NF = WN * NRW  # weight fragments of the tile
    TN = 16 * NF
    TILES = tiles(NF)
    MTP = row_fragments(MT, WN, MS)  # row fragments of a workgroup, a multiple of the row waves
    WCNT = TILES * KS  # workgroups of one row split
    GRID = WCNT * MS
    MW = MTP // WM  # row fragments of a wave
    assert NWV % WN == 0 and N // 16 % NF == 0 and (NF * 2) % NWV == 0
    assert 1 <= MT <= MT_MAX and NB % KS == 0 and (not FUSED or GRID >= 16 * MT) and (MS == 1 or WCNT % 8 == 0) and GRID <= FLAG_WORDS  # a token a workgroup; a pair of row splits on one XCD
    NBK = NB // KS  # K blocks per workgroup
    MR_ALLOC = slab_rows(MT, WN, MS)
    IDS_VECS = (192 * TOPK + 255) // 256  # vectors of 4 ids a lane loads
    MWORDS = mask_words(16 * MT)  # words of an expert's token mask
    MASK_B = NE * MWORDS * 4
    SLOT_FRAGS = NF + MTP
    WF = NF  # weight positions per slot
    SLOT = SLOT_FRAGS * 2048
    XD = MTP * 2  # activation DMAs per block
    XPW = 0 if SKEL & 4 else (XD + NWV - 1) // NWV
    WPW = NF * 2 // NWV  # weight DMAs per wave per block
    PER = WPW + XPW  # DMAs per wave per block
    # as many ring slots as the LDS holds (every block of the K slice: the whole stream is in flight from the start)
    R = max(2, min(NBK, RING_BYTES // SLOT, 63 // PER + 1))
    LDS = R * SLOT
    assert LDS <= RING_BYTES and (R - 1) * PER <= 63, (LDS, R, PER)
    assert not AUX or LDS >= MASK_B + 4096 + 256, LDS  # the masks and the first rows of the sort live in the dead ring

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]
        xch: fx.Array[fx.Int32, NWV * 64 * 8, 16]
        ent: fx.Array[fx.Int32, LDS_WORDS["ent"], 16]
        osig: fx.Array[fx.Float32, LDS_WORDS["osig"], 16]
        oid: fx.Array[fx.Int32, LDS_WORDS["oid"], 16]
        tie: fx.Array[fx.Int32, LDS_WORDS["tie"], 16]
        tex: fx.Array[fx.Int32, LDS_WORDS["tex"], 16]
        tmask: fx.Array[fx.Int32, LDS_WORDS["tmask"], 16]

    name = f"router_{'fused' if FUSED else 'partials'}_e896_h7168_mt{MT}_ks{KS}_w{WN}n{NRW}" + ("_aux" if AUX else "") + ("_ts" if TS else "") + (f"_skel{SKEL}" if SKEL else "") + ("_shuf" if SHUF else "_row")

    @flyc.kernel(name=name, known_block_size=[64 * NWV, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_p: fx.Int64, arg_c: fx.Int64, arg_bias: fx.Int64,
               arg_ow: fx.Int64, arg_oid: fx.Int64, arg_aux: fx.Int64, arg_sti: fx.Int64, arg_sw: fx.Int64,
               arg_sei: fx.Int64, arg_nv: fx.Int64, arg_mi: fx.Int64, arg_rs: fx.Int64, arg_mb: fx.Int64,
               i32_m: fx.Int32, i32_os: fx.Int32, i32_scale: fx.Int32, i32_cap: fx.Int32, i32_zvecs: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wid = bid % WCNT
        row0 = (bid // WCNT) * (MTP * 16)  # first row of the workgroup's row split
        kz = wid % KS
        tile = wid // KS
        n0 = tile * TN
        wn = wave % WN  # the wave's column fragments wn NRW ..
        wm = wave // WN  # and row fragments wm MW ..
        b0 = kz * NBK
        g = lane // 16
        l16 = lane % 16
        r8 = lane // 8

        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_m * (K * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K * 2)
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))
        xch_base = fx.Int32(fx.ptrtoint(smem.xch.ptr))

        def stamp(k_, val=None):
            if const_expr(TS):  # wave 0 / thread 0 of the workgroup: s_memrealtime (100 MHz) of event k_
                fx.rocdl.sched_barrier(0)
                t_ = fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])) if val is None else fx.Int64(val)
                tsr = buffer_ops.create_buffer_resource_from_addr(arg_mb if AUX else arg_rs, num_records_bytes=4096 * 16 * 8)
                buffer_ops.buffer_store(t_, tsr, (bid * 16 + k_) * 8, offset_is_bytes=True, mask=tx == 0)
                fx.rocdl.sched_barrier(0)

        stamp(0)
        if const_expr(FUSED):
            # this call's tag: every workgroup's flag advances by one a call, so the flag read now plus one is the value all
            # workgroups publish when their slabs are stored
            flags = buffer_ops.create_buffer_resource_from_addr(arg_c, num_records_bytes=FLAG_WORDS * 4)
            tag = fx.Int32(buffer_ops.buffer_load(flags, bid, vec_width=1, dtype=fx.Int32,
                                                  cache_modifier=SC1 | 1)) + 1

        # DMA jj of a fragment: rows 8 jj + r8, 16-byte piece (lane % 8) ^ ((r8 >> 1) | (jj << 2)) of the line
        w_src, w_dst = [], []
        for i in range_constexpr(WPW):
            d = wave + i * NWV  # weight DMA 2 f + jj: rows 8 jj + r8 of fragment f
            f = d // 2
            jj = d % 2
            pc = (lane % 8) ^ ((r8 // 2) + jj * 4)
            if const_expr(SHUF):
                off = (tile * NF + f) * (K * 32) + (jj * 8 + r8) * 128 + pc * 16
            else:
                off = (n0 + f * 16 + jj * 8 + r8) * (K * 2) + pc * 16
            w_src.append(off)
            w_dst.append(d * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave + i * NWV
            d = (d < XD).select(d, fx.Int32(XD - 1))  # spare DMAs repeat the last one (same bytes, same place)
            jx = d % 2
            row = row0 + (d // 2) * 16 + jx * 8 + r8
            row = (row < i32_m).select(row, i32_m - 1)  # rows past M read the last row; never stored
            px = (lane % 8) ^ ((r8 // 2) + jx * 4)
            x_src.append(row * (K * 2) + px * 16)
            x_dst.append((WF * 2 + d) * 1024)
        aux_w = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), W_CM)
        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def issue(rs, dst, src, b, soff, aux):
            d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + (b % R) * SLOT + dst))
            lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
            _rocdl_ods.raw_ptr_buffer_load_async_lds(
                rs, lp, fx.Int32(16).ir_value(), fx.Int32(src).ir_value(),
                fx.Int32(soff).ir_value(), fx.Int32(0).ir_value(), aux=aux)

        def dma(b):
            bg = b0 + b
            for q in range_constexpr(WPW):
                issue(wr, w_dst[q], w_src[q], b, bg * (2048 if SHUF else 128), aux_w)
            for q in range_constexpr(XPW):
                issue(xr, x_dst[q], x_src[q], b, bg * 128, aux_x)

        r8r = l16 % 8
        hh = l16 // 8
        rd_off = [hh * 1024 + (r8r * 8 + ((kk * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16 for kk in range_constexpr(2)]

        def rd(b, pos, kk):
            p = _lds_ptr(lds_base + (b % R) * SLOT + pos * 2048 + rd_off[kk])
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MW)] for _ in range_constexpr(NRW)]

        def read_ops(b, kk):
            if const_expr(SKEL & 1):
                return None
            return ([rd(b, wn * NRW + i, kk) for i in range_constexpr(NRW)],
                    [rd(b, WF + wm * MW + m, kk) for m in range_constexpr(MW)])

        def mfmas(ops):
            if const_expr(SKEL & 1):
                return
            wv, xv = ops
            for i in range_constexpr(NRW):
                for m in range_constexpr(MW):
                    acc[i][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [wv[i], xv[m], acc[i][m], 0, 0, 0]))

        def barrier():
            llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)

        if const_expr(AUX):
            # the sort's pads and the zeroed moe buffer, issued before the weight stream: the loop's last wait (vmcnt 0)
            # has them acknowledged before this workgroup's slabs are stored, so a consumer wave, which goes on once every
            # slab is in, writes its valid rows after every pad
            pre_bytes = fx.Int64(i32_cap) * 4
            aux_prefill(bid * (64 * NWV) + tx, GRID * 64 * NWV, i32_m, i32_cap, i32_zvecs,
                        buffer_ops.create_buffer_resource_from_addr(arg_sti, num_records_bytes=pre_bytes),
                        buffer_ops.create_buffer_resource_from_addr(arg_sw, num_records_bytes=pre_bytes),
                        buffer_ops.create_buffer_resource_from_addr(arg_mi, num_records_bytes=pre_bytes),
                        buffer_ops.create_buffer_resource_from_addr(arg_mb, num_records_bytes=fx.Int64(i32_zvecs) * 16))
        for b in range_constexpr(R):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=(R - 1) * PER)
        barrier()
        _sb()
        ops_a = read_ops(0, 0)
        for b in range_constexpr(NBK):
            _sb()
            ops_b = read_ops(b, 1)
            _sb()
            mfmas(ops_a)
            _sb()
            if b + 1 < NBK:
                fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=(min(b + R - 1, NBK - 1) - (b + 1)) * PER)
                barrier()
                _sb()
                ops_a = read_ops(b + 1, 0)
                _sb()
                if b + R < NBK:
                    dma(b + R)
                _sb()
            mfmas(ops_b)

        stamp(1)
        pr = buffer_ops.create_buffer_resource_from_addr(arg_p, num_records_bytes=KS * MR_ALLOC * (N * 4))
        for i in range_constexpr(NRW):
            for m in range_constexpr(MW):
                row = row0 + (wm * MW + m) * 16 + l16
                v = acc[i][m].bitcast(fx.Int32)
                if const_expr(FUSED):
                    # the call's parity in each word's lowest mantissa bit: the consumer takes a word as this call's only when
                    # the bit matches (the words of the previous call, never of an older one, carry the other parity)
                    v = Vec.from_elements([(fx.Int32(v[e]) & fx.Int32(-2)) | (tag & 1) for e in range_constexpr(4)], fx.Int32)
                if const_expr(not SKEL & 16):
                    buffer_ops.buffer_store(v, pr, ((kz * MR_ALLOC + row) * N + n0 + (wn * NRW + i) * 16 + g * 4) * 4,
                                            offset_is_bytes=True, cache_modifier=SC1 if FUSED else 0)
        if const_expr(FUSED):
            # no arrival: the slabs carry the call's parity. The epoch word advances (this workgroup's own copy only, read
            # at the start); workgroup ``t`` (t < M) takes token t in its wave 0: it loads the token's slab words of all
            # tiles at once and loads again until every word has this call's parity (a token's 50 KB of slabs go through
            # one CU's load path), then sums, sigmoids and keys them
            buffer_ops.buffer_store(tag, flags, bid, mask=tx == 0, cache_modifier=SC1)
            stamp(2)
            if bid < (i32_m * 0 if SKEL & 8 else i32_m):
                tok = bid
                stamp(3)
                if const_expr(AUX):  # this call's id parity (the sort workspace's own epoch: it serves one M)
                    ep_res = buffer_ops.create_buffer_resource_from_addr(arg_aux, num_records_bytes=(IDS_WORDS + 256) * 4)
                    ep = fx.Int32(buffer_ops.buffer_load(ep_res, EPOCH_BASE + bid, vec_width=1, dtype=fx.Int32,
                                                         cache_modifier=SC1 | 1)) + 1
                out_bytes = fx.Int64(i32_m) * fx.Int64(i32_os) * 4
                owr = buffer_ops.create_buffer_resource_from_addr(arg_ow, num_records_bytes=out_bytes)
                oir = buffer_ops.create_buffer_resource_from_addr(arg_oid, num_records_bytes=out_bytes)
                br = buffer_ops.create_buffer_resource_from_addr(arg_bias, num_records_bytes=N * 4)
                # wave ``wave`` takes the experts 256 wave + 4 lane + j (its 14 slab loads: one wave's load path moves
                # about 25 GB/s, so the four waves share the token's 50 KB) and loads again until all its words carry
                # this call's parity
                qv = wave
                qvalid = (qv != NQ - 1) | (lane < 32)
                qsum = Vec.filled(4, 0.0, fx.Float32)
                qbias = Vec(buffer_ops.buffer_load(br, 256 * qv + 4 * lane, vec_width=4, dtype=fx.Float32, mask=qvalid))
                stale = fx.Int32(1)
                iters = fx.Int32(0)
                while stale != 0:
                    qparts = [Vec(buffer_ops.buffer_load(pr, (s * MR_ALLOC + tok) * N + 256 * qv + 4 * lane,
                                                         vec_width=4, dtype=fx.Int32, mask=qvalid,
                                                         cache_modifier=SC1 | 1)) for s in range_constexpr(KS)]
                    qdiff = fx.Int32(0)
                    qacc = None
                    for s in range_constexpr(KS):
                        for ee in range_constexpr(4):
                            qdiff = qdiff | (fx.Int32(qparts[s][ee]) ^ tag)
                        qw = Vec.from_elements([fx.Int32(qparts[s][ee]) & fx.Int32(-2) for ee in range_constexpr(4)],
                                               fx.Int32).bitcast(fx.Float32)
                        qacc = qw if qacc is None else qacc + qw
                    qsum = qacc
                    stale = _popcount(_ballot(qvalid.select(qdiff & 1, fx.Int32(0)) != 0))
                    iters = iters + 1
                    fx.rocdl.s_sleep(POLL_SLEEP)
                stamp(4)
                stamp(5, iters)
                mine = topk_q_keys(qsum, br, lane, qv, qbias)
                xaddr = xch_base + (wave * 64 + lane) * 32
                llvm.store(_raw(Vec.from_elements([sg.bitcast(fx.Int32) for sg, _k in mine], fx.Int32)), _lds_ptr(xaddr),
                           alignment=16)
                llvm.store(_raw(Vec.from_elements([fx.Int32(key) for _s, key in mine], fx.Int32)), _lds_ptr(xaddr + 16),
                           alignment=16)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                if wave == 0:
                    if const_expr(SKEL & 2):
                        sgv = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(xch_base + lane * 32), alignment=16))
                        buffer_ops.buffer_store(fx.Int32(sgv[0]), owr, tok * i32_os + lane, mask=lane < TOPK)
                    else:
                        st = topk_new(lane)
                        for q in range_constexpr(NQ):
                            sgv = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(xch_base + (q * 64 + lane) * 32), alignment=16))
                            kyv = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(xch_base + (q * 64 + lane) * 32 + 16), alignment=16))
                            for j in range_constexpr(4):
                                st.sig.append(fx.Int32(sgv[j]).bitcast(fx.Float32))
                                st.key.append(fx.Uint32(kyv[j]))
                        stamp(8)
                        topk_pivot(st, smem, wave, lane)
                        stamp(9)
                        out_id = fx.Int32(0)
                        out_w = fx.Float32(0.0)
                        if st.nc <= MAXC:
                            topk_fast_write(st, smem, wave, lane)
                            topk_rank_start(st, smem, wave, lane)
                            stamp(10)
                            rank = fx.Int32(0)
                            e = fx.Int32(0)
                            while e < st.nc:
                                rank = topk_rank_step(rank, e, st, smem, wave)
                                e = e + RANK_UNROLL
                            stamp(11)
                            out_id, out_w = topk_finish(rank, lane < st.nc, st.mysig, st.myid, lane, smem, wave, owr, oir, tok,
                                                        i32_os, i32_scale)
                        else:
                            ss = topk_slow_select(st)
                            if ss.total_ties != ss.need:
                                topk_tie_masks(ss, smem, wave, lane)
                            rank, mysig, myid = topk_slow_out(ss, smem, wave, lane)
                            out_id, out_w = topk_finish(rank, lane < TOPK, mysig, myid, lane, smem, wave, owr, oir, tok, i32_os, i32_scale)
                        stamp(12)
                        if const_expr(AUX):
                            # publish the token's 16 expert ids (tagged with the call's parity), load every token's ids until
                            # all carry the parity, OR their bits into the expert masks (LDS) and write the token's 16 rows
                            ids_res = buffer_ops.create_buffer_resource_from_addr(arg_aux, num_records_bytes=IDS_WORDS * 4)
                            par = fx.Int32(ep & 1)
                            buffer_ops.buffer_store(ep, ep_res, EPOCH_BASE + tok, mask=lane == 0, cache_modifier=SC1)
                            buffer_ops.buffer_store(out_id | (par << 31), ids_res, tok * TOPK + lane, mask=lane < TOPK,
                                                    cache_modifier=SC1)
                            aux_zero_masks(lds_base, lane, MWORDS)
                            nwords = i32_m * TOPK
                            idw = [Vec.filled(4, 0, fx.Int32) for _ in range_constexpr(IDS_VECS)]
                            stale = fx.Int32(1)
                            while stale != 0:
                                bad = fx.Int32(0)
                                loaded = []
                                for it in range_constexpr(IDS_VECS):
                                    wi = (it * 64 + lane) * 4
                                    inside = wi < nwords
                                    v = Vec(buffer_ops.buffer_load(ids_res, wi, vec_width=4, dtype=fx.Int32, mask=inside,
                                                                   cache_modifier=SC1 | 1))
                                    for ee in range_constexpr(4):
                                        bad = bad | inside.select(((fx.Int32(v[ee]) >> 31) & 1) ^ par, fx.Int32(0))
                                    loaded.append(v)
                                idw = loaded
                                stale = _popcount(_ballot(bad != 0))
                                fx.rocdl.s_sleep(POLL_SLEEP)
                            stamp(13)
                            for it in range_constexpr(IDS_VECS):
                                if it * 256 < nwords:  # a vector row with ids (uniform): the others have none to OR
                                    for ee in range_constexpr(4):
                                        wi = (it * 64 + lane) * 4 + ee
                                        inside = wi < nwords
                                        t_ = wi >> 4
                                        ex = fx.Int32(idw[it][ee]) & 1023
                                        # lanes past the last id OR zero into a slot of their own (one address for all of
                                        # them would serialise the atomic)
                                        lds_or(inside.select(lds_base + (ex * MWORDS + (t_ >> 5)) * 4,
                                                             lds_base + MASK_B + 4096 + lane * 4),
                                               inside.select(fx.Int32(1) << (t_ & 31), fx.Int32(0)))
                            fx.rocdl.s_waitcnt(lgkmcnt=0)
                            fx.rocdl.sched_barrier(0)
                            stamp(14)
                            ac = aux_counts(lds_base, lane, MWORDS)
                            aux_publish_cum(ac, lds_base, wave, lane, MWORDS)
                            stamp(15)
                            row = aux_pair_row(lds_base, wave, out_id, tok, MWORDS)
                            cap_bytes = fx.Int64(i32_cap) * 4
                            sti_r = buffer_ops.create_buffer_resource_from_addr(arg_sti, num_records_bytes=cap_bytes)
                            sw_r = buffer_ops.create_buffer_resource_from_addr(arg_sw, num_records_bytes=cap_bytes)
                            mi_r = buffer_ops.create_buffer_resource_from_addr(arg_mi, num_records_bytes=cap_bytes)
                            rs_r = buffer_ops.create_buffer_resource_from_addr(arg_rs, num_records_bytes=fx.Int64(i32_m) * 64)
                            keep = lane < TOPK
                            buffer_ops.buffer_store(tok | (lane << 24), sti_r, row, mask=keep, cache_modifier=SC1)
                            buffer_ops.buffer_store(out_w, sw_r, row, mask=keep, cache_modifier=SC1)
                            buffer_ops.buffer_store(tok, mi_r, row, mask=keep, cache_modifier=SC1)
                            buffer_ops.buffer_store(row, rs_r, tok * TOPK + lane, mask=keep, cache_modifier=SC1)
                            stamp(5)
                            if tok == 0:
                                sei_r = buffer_ops.create_buffer_resource_from_addr(arg_sei, num_records_bytes=cap_bytes)
                                nv_r = buffer_ops.create_buffer_resource_from_addr(arg_nv, num_records_bytes=8)
                                buffer_ops.buffer_store(ac.total, nv_r, 0, mask=lane == 0, cache_modifier=SC1)
                                buffer_ops.buffer_store(i32_m, nv_r, 1, mask=lane == 0, cache_modifier=SC1)
                                maxb = aux_max_blocks(ac)
                                blk = fx.Int32(0)
                                while blk < maxb:
                                    aux_expert_blocks(ac, blk, lane, sei_r)
                                    blk = blk + 1

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_p: fx.Int64, arg_c: fx.Int64, arg_bias: fx.Int64,
               arg_ow: fx.Int64, arg_oid: fx.Int64, arg_aux: fx.Int64, arg_sti: fx.Int64, arg_sw: fx.Int64,
               arg_sei: fx.Int64, arg_nv: fx.Int64, arg_mi: fx.Int64, arg_rs: fx.Int64, arg_mb: fx.Int64,
               i32_m: fx.Int32, i32_os: fx.Int32, i32_scale: fx.Int32, i32_cap: fx.Int32, i32_zvecs: fx.Int32,
               stream: fx.Stream):
        kernel(arg_x, arg_w, arg_p, arg_c, arg_bias, arg_ow, arg_oid, arg_aux, arg_sti, arg_sw, arg_sei, arg_nv,
               arg_mi, arg_rs, arg_mb, i32_m, i32_os, i32_scale, i32_cap, i32_zvecs).launch(
            grid=(GRID, 1, 1), block=(64 * NWV, 1, 1), stream=stream)

    return launch
