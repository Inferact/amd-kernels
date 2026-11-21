# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# Block scans and the order-preserving score key follow AITER's FlyDSL DCP top-k merge
# (aiter/ops/flydsl/kernels/dcp_topk_merge.py, MIT, Copyright (C) 2024-2026, Advanced Micro Devices, Inc.).
"""Score all-gather, global top-2048 merge and owned-slot CSR emission in one launch.

One workgroup per row; every rank pushes, the receivers poll the pushed words themselves (no flags: a region is zero
between uses and a pushed word is never zero). Each rank pushes to the seven peers (system-scope stores) one block of
512 keys plus a header. The keys are those of its plane in the bins at or above a cut bin of the 10-bit histogram of
the order key, the cut being the lowest bin whose cumulative count from the top stays within 512; the block is padded
with an impossible key. The header carries the cut bin, the plane's count of real keys and of -inf keys.
Every receiver takes the highest cut over the eight planes: above it all planes are complete, so a histogram of the
received keys finds the bin B of the 2048th key when at least 2048 keys are above it, and two radix passes in LDS
(11 + 11 bits) pick the exact threshold. Otherwise (a skewed row) a second exchange pushes the full histograms, and
when B is still inside a plane's missing part a third pushes the bin-B keys, which the passes stream through LDS in
chunks. With fewer than 2048 real keys the threshold is -inf. The decision is a function of data every rank holds, so
all ranks take the same path. The rank's own plane is then compacted in position order (keys above the threshold, then
ties in rank-major position order up to the remaining room, counting the lower ranks' tied band keys, or their -inf
keys) and mapped to physical KV rows. The row's count is published tagged with the call sequence; each row waits for
the counts of the rows before it and writes its slots at that prefix. Rows are dispatched in index order and a row
waits only on lower rows and on the same row of the peers.

Two receive slots (call parity) bound peer lead and lag: a peer starts call c + 1 only after its call c ended, and a
rank cannot end call c + 1 before every peer pushed its first round of c + 1, which follows that peer's reads of call c.
Receivers zero what they read at the end of the call. The device counter is advanced by the last workgroup to arrive
(ticket) after every workgroup has read it, so graph replays never reuse a host sequence. Waits are atomic loads (a
plain-load wait loop is deleted by the compiler) and bounded by SPIN_LIMIT; DCP8_TIMING=1 builds stamp the phases
into the debug buffer. Outputs use plain stores.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import Array, Int32, arith, const_expr, gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

WORLD = 8
K = 2048
NT = 512
EPT = 4
MAX_ROWS = 256
NB = 1024
HIST_SHIFT = 22
HIST_BYTES = 2176
SPEC_OFF = HIST_BYTES
SPEC_E = 512
SPEC_ELEM_OFF = SPEC_OFF + 16
BAND_OFF = SPEC_OFF + 2176
BAND_ELEM_OFF = BAND_OFF + 16
REGION_BYTES = 12672
SLOT_BYTES = MAX_ROWS * WORLD * REGION_BYTES
SEG = 512
LIST_CAP = WORLD * SEG
NWAVES = NT // 64
GHC = 4
SYS = 0x11
NINF_KEY = 0x007FFFFF
VALID2 = 0x80008000 - (1 << 32)
PACK = 16
DPP = ((0x111, 1), (0x112, 2), (0x114, 4), (0x118, 8))
SPIN_LIMIT = 1 << 28
SLEEP = 1
STAG = 2
STAG_MIN_ROWS = 64
TIMING = bool(__import__("os").environ.get("DCP8_TIMING"))


def _now():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _ptr(addr, dtype):
    return fx.inttoptr(fx.PointerType.get(dtype, address_space=fx.AddressSpace.Global, alignment=8 if dtype == T.i64 else 4), addr)


def _poll(addr, dtype):
    ty = fx.Int64.ir_type if dtype == T.i64 else fx.Int32.ir_type
    return llvm.LoadOp(ty, _ptr(addr, dtype).llvm_ptr, alignment=8 if dtype == T.i64 else 4,
                       ordering=llvm.AtomicOrdering.monotonic, syncscope="").result


def _ticket(addr):
    return fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, _ptr(addr, T.i32).llvm_ptr, fx.Int32(1).ir_value(),
                                     llvm.AtomicOrdering.monotonic, syncscope="", alignment=4).result)


def _sel(p, vals):
    out = vals[-1]
    for i in range(len(vals) - 2, -1, -1):
        out = (p == i).select(vals[i], out)
    return out


def _key(bits):
    ords = bits ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
    nan = fx.Uint32(bits & fx.Int32(0x7FFFFFFF)) > 0x7F800000
    return nan.select(fx.Int32(0), ords ^ fx.Int32(-(1 << 31)))


def _ugt(a, b):
    return fx.Uint32(a) > fx.Uint32(b)


def _barrier():
    fly_rocdl.s_waitcnt(lgkmcnt=0)
    gpu.barrier()


def _unwrap(v):
    return v.ir_value() if hasattr(v, "ir_value") else arith.unwrap(v)


def _make_storage():
    @fx.struct
    class Storage:
        hist: Array[Int32, NB + 64, 16]
        gh: Array[Int32, GHC * (NB + 64), 16]
        bins1: Array[Int32, 2048 + 64, 16]
        bins2: Array[Int32, 2048 + 64, 16]
        lst: Array[Int32, LIST_CAP, 16]
        ownb: Array[Int32, K, 16]
        spk: Array[Int32, SPEC_E + 4, 16]
        scan: Array[Int32, 128, 16]
        state: Array[Int32, 16, 16]
        acc: Array[Int32, 64, 4]
        cnt: Array[Int32, 16, 4]

    return Storage


def compile_topk_gather_merge():
    @flyc.kernel(name="dcp8_topk_gather_merge_top2048", known_block_size=[NT, 1, 1])
    def kernel(arg_scores: fx.Int64, arg_idx: fx.Int64, arg_bt: fx.Int64, arg_slots: fx.Int64, arg_indptr: fx.Int64,
               arg_ctl: fx.Int64, arg_counts: fx.Int64, arg_dbg: fx.Int64,
               p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64, p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64,
               rows: fx.Int32, rank: fx.Int32, scores_stride: fx.Int32, idx_stride: fx.Int32, bt_stride: fx.Int32,
               page_size: fx.Int32, block_stride: fx.Int32):
        tid = fx.thread_idx.x
        row = fx.block_idx.x
        pools = [p0, p1, p2, p3, p4, p5, p6, p7]
        seq = fx.Int32(_ptr(arg_ctl, T.i32)[0]) + 1
        slot = fx.Int64((seq - 1) & 1)
        own_pool = _sel(rank, pools)
        region0 = (slot * MAX_ROWS + row) * WORLD * REGION_BYTES
        t_start = _now()
        peer_base = [_sel((rank + (row + j - 1) % (WORLD - 1) + 1) % WORLD, pools) + region0 + rank * REGION_BYTES for j in range_constexpr(1, WORLD)]
        dst1 = [buffer_ops.create_buffer_resource_from_addr(peer_base[j - 1], num_records_bytes=HIST_BYTES) for j in range_constexpr(1, WORLD)]
        lane = tid % 64
        warp = fx.Int32(fly_rocdl.readfirstlane(T.i32, _unwrap(tid // 64)))

        storage = fx.SharedAllocator().allocate(_make_storage())
        s_hist = storage.hist.peek().view(fx.make_layout(NB + 64, 1))
        s_gh = storage.gh.peek().view(fx.make_layout(GHC * (NB + 64), 1))
        s_b1 = storage.bins1.peek().view(fx.make_layout(2048 + 64, 1))
        s_b2 = storage.bins2.peek().view(fx.make_layout(2048 + 64, 1))
        s_lst = storage.lst.peek().view(fx.make_layout(LIST_CAP, 1))
        s_own = storage.ownb.peek().view(fx.make_layout(K, 1))
        s_spk = storage.spk.peek().view(fx.make_layout(SPEC_E + 4, 1))
        s_scan = storage.scan.peek().view(fx.make_layout(128, 1))
        s_state = storage.state.peek().view(fx.make_layout(16, 1))
        s_acc = storage.acc.peek().view(fx.make_layout(64, 1))
        s_cnt = storage.cnt.peek().view(fx.make_layout(16, 1))

        def lds_add(buf, idx, val):
            return fx.atomic_add(fx.get_iter(buf) + idx, val, syncscope="workgroup")

        def inclusive(val):
            raw = _unwrap(val)
            zero = _unwrap(0)
            for op, thr_ in DPP:
                r = fly_rocdl.update_dpp(T.i32, zero, raw, op, 0xF, 0xF, True)
                val = (lane >= fx.Int32(thr_)).select(val + fx.Int32(r), val)
                raw = _unwrap(val)
            r16 = fly_rocdl.ds_bpermute(T.i32, ((lane & fx.Int32(0x30)) - 1) * fx.Int32(4), val)
            val = (lane >= fx.Int32(16)).select(val + fx.Int32(r16), val)
            r32 = fly_rocdl.ds_bpermute(T.i32, ((lane & fx.Int32(0x30)) - fx.Int32(17)) * fx.Int32(4), val)
            return (lane >= fx.Int32(32)).select(val + fx.Int32(r32), val)

        def block_prefix(val, scan, base):
            inc = inclusive(val)
            if lane == fx.Int32(63):
                scan[base + warp] = inc
            _barrier()
            res = inc - val
            tot = fx.Int32(0)
            for w_ in range_constexpr(NWAVES):
                t_ = scan[base + w_]
                res = res + (warp > w_).select(t_, fx.Int32(0))
                tot = tot + t_
            return res, tot

        def append(hit, cell):
            mask = fx.Int64(fly_rocdl.ballot(T.i64, _unwrap(hit)))
            n = fx.Int32(llvm.call_intrinsic(T.i64, "llvm.ctpop.i64", [_unwrap(mask)], [], []))
            old = fx.Int32(0)
            if lane == fx.Int32(0):
                old = lds_add(s_cnt, cell, n)
            base = fx.Int32(fly_rocdl.readfirstlane(T.i32, _unwrap(old)))
            a = fly_rocdl.mbcnt_lo(T.i32, _unwrap(mask.to(fx.Int32)), _unwrap(fx.Int32(0)))
            return base + fx.Int32(fly_rocdl.mbcnt_hi(T.i32, _unwrap((mask >> 32).to(fx.Int32)), a))

        def accumulate(bins, shift, dmask, prefix):
            for p in range_constexpr(WORLD):
                cp = s_cnt[p]
                inb = tid < cp
                key = s_lst[p * SEG + inb.select(tid, fx.Int32(0))]
                hit = inb & ((key & dmask) == prefix)
                fx.atomic_add(fx.get_iter(bins) + hit.select((key >> fx.Int32(shift)) & fx.Int32(2047), fx.Int32(2048) + lane), fx.Int32(1), syncscope="workgroup")

        def acc_reg(bins, shift, dmask, prefix, keys, oks):
            for k in range_constexpr(len(keys)):
                hit = oks[k] & ((keys[k] & dmask) == prefix)
                if hit:
                    fx.atomic_add(fx.get_iter(bins) + ((keys[k] >> fx.Int32(shift)) & fx.Int32(2047)), fx.Int32(1), syncscope="workgroup")

        def load_chunk(r, cnts, cnt_arr, lst_arr, own_arr):
            if tid == 0:
                for p in range_constexpr(WORLD):
                    rest = cnts[p] - r * SEG
                    cnt_arr[p] = (rest < 0).select(fx.Int32(0), (rest > SEG).select(fx.Int32(SEG), rest))
            _barrier()
            for p in range_constexpr(WORLD):
                okp = tid < cnt_arr[p]
                if fx.Int32(p) == rank:
                    if okp:
                        lst_arr[p * SEG + tid] = own_arr[r * SEG + tid]
                else:
                    addr_p = own_pool + region0 + p * REGION_BYTES + BAND_ELEM_OFF + fx.Int64(r * SEG + tid) * 4
                    word = fx.Int32(_poll(addr_p, T.i32))
                    spin5 = fx.Int32(0)
                    while ((word == fx.Int32(0)) & okp) & (spin5 < SPIN_LIMIT):
                        spin5 = spin5 + 1
                        word = fx.Int32(_poll(addr_p, T.i32))
                    if okp:
                        lst_arr[p * SEG + tid] = word
            _barrier()

        def reset_words(rr, off, cnt):
            for e in range(fx.Int32(0), (cnt + NT - 1) // NT, fx.Int32(1)):
                if tid + e * NT < cnt:
                    buffer_ops.buffer_store(fx.Int32(0), rr, off + tid + e * NT, cache_modifier=SYS)

        def stamp(dbg, row_, tid_, k):
            if const_expr(TIMING):
                if tid_ == 0:
                    _ptr(dbg + fx.Int64(row_ * 32 + k) * 8, T.i64)[0] = _now()

        stamp(arg_dbg, row, tid, 0)
        if tid == 0:
            for i in range_constexpr(48):
                s_acc[i] = 0
            for i in range_constexpr(16):
                s_state[i] = 0
                s_cnt[i] = 0
        for b in range_constexpr((NB + 64 + NT - 1) // NT):
            if tid + b * NT < NB + 64:
                s_hist[tid + b * NT] = 0
        for b in range_constexpr((GHC * (NB + 64) + NT - 1) // NT):
            if tid + b * NT < GHC * (NB + 64):
                s_gh[tid + b * NT] = 0
        s_spk[tid] = fx.Int32(-1)
        for b in range_constexpr((2048 + 64 + NT - 1) // NT):
            if tid + b * NT < 2048 + 64:
                s_b1[tid + b * NT] = 0
                s_b2[tid + b * NT] = 0

        sr = buffer_ops.create_buffer_resource_from_addr(arg_scores + fx.Int64(row * scores_stride) * 4, num_records_bytes=K * 4)
        mine = Vec(buffer_ops.buffer_load(sr, tid * EPT, vec_width=EPT, dtype=fx.Int32))
        u = [_key(mine[i]) for i in range_constexpr(EPT)]
        _barrier()
        stamp(arg_dbg, row, tid, 26)
        real = [_ugt(u[i], fx.Int32(NINF_KEY)) for i in range_constexpr(EPT)]
        bins_i = [(u[i] >> fx.Int32(HIST_SHIFT)) & fx.Int32(NB - 1) for i in range_constexpr(EPT)]
        ninf = fx.Int32(0)
        for i in range_constexpr(EPT):
            lds_add(s_hist, real[i].select(bins_i[i], fx.Int32(NB) + lane), fx.Int32(1))
            ninf = ninf + (u[i] == fx.Int32(NINF_KEY)).select(fx.Int32(1), fx.Int32(0))
        incn = inclusive(ninf)
        if lane == fx.Int32(63):
            lds_add(s_acc, 8, incn)
        _barrier()
        c0 = s_hist[2 * tid]
        c1 = s_hist[2 * tid + 1]
        hd = (c0 | fx.Int32(0x8000)) | ((c1 | fx.Int32(0x8000)) << fx.Int32(16))
        ninf_word = s_acc[8] | fx.Int32(0x8000)
        sumt_o = c0 + c1
        res_o, tot_o = block_prefix(sumt_o, s_scan, 80)
        above_o = tot_o - res_o - sumt_o
        ge_hi = above_o + c1
        ge_lo = ge_hi + c0
        below = s_hist[(2 * tid - 1) & 1023]
        low_lo = (ge_lo <= SPEC_E) & ((ge_lo + below > SPEC_E) | (tid == 0))
        low_hi = (ge_hi <= SPEC_E) & (ge_lo > SPEC_E)
        if low_lo | low_hi:
            s_acc[9] = low_lo.select(2 * tid, 2 * tid + 1)
            s_acc[10] = low_lo.select(ge_lo, ge_hi)
        _barrier()
        stamp(arg_dbg, row, tid, 27)
        cut = s_acc[9]
        n_e = s_acc[10]
        hit_s = [real[i] & (bins_i[i] >= cut) for i in range_constexpr(EPT)]
        masks = [fx.Int64(fly_rocdl.ballot(T.i64, _unwrap(hit_s[i]))) for i in range_constexpr(EPT)]
        cnts_w = [fx.Int32(llvm.call_intrinsic(T.i64, "llvm.ctpop.i64", [_unwrap(masks[i])], [], [])) for i in range_constexpr(EPT)]
        old_s = fx.Int32(0)
        if lane == fx.Int32(0):
            old_s = lds_add(s_cnt, 8, cnts_w[0] + cnts_w[1] + cnts_w[2] + cnts_w[3])
        base_s = fx.Int32(fly_rocdl.readfirstlane(T.i32, _unwrap(old_s)))
        pos_s = []
        run_s = base_s
        for i in range_constexpr(EPT):
            a_ = fly_rocdl.mbcnt_lo(T.i32, _unwrap(masks[i].to(fx.Int32)), _unwrap(fx.Int32(0)))
            pos_s.append(run_s + fx.Int32(fly_rocdl.mbcnt_hi(T.i32, _unwrap((masks[i] >> 32).to(fx.Int32)), a_)))
            run_s = run_s + cnts_w[i]
        for i in range_constexpr(EPT):
            if hit_s[i]:
                s_spk[pos_s[i]] = u[i]
        _barrier()
        if const_expr(STAG > 0):
            t_go = t_start + fx.Int64(row) * STAG
            spin0 = fx.Int32(0)
            while (rows > STAG_MIN_ROWS) & (_now() < t_go) & (spin0 < SPIN_LIMIT):
                spin0 = spin0 + 1
                fly_rocdl.s_sleep(1)
        kv = Vec.from_elements([s_spk[4 * (tid & 127) + i] for i in range_constexpr(4)], fx.Int32)
        hdr2 = Vec.from_elements([(cut << fx.Int32(16)) | (n_e + 1), ((s_acc[8] + 1) << fx.Int32(16)) | (tot_o + 1)], fx.Int32)
        for j in range_constexpr(1, WORLD):
            dst = buffer_ops.create_buffer_resource_from_addr(peer_base[j - 1] + SPEC_OFF, num_records_bytes=16 + SPEC_E * 4)
            buffer_ops.buffer_store(kv, dst, 16 + 16 * tid, mask=tid < SPEC_E // 4, cache_modifier=SYS, offset_is_bytes=True)
            buffer_ops.buffer_store(hdr2, dst, 0, mask=tid == 0, cache_modifier=SYS)
        stamp(arg_dbg, row, tid, 1)

        ir_ = buffer_ops.create_buffer_resource_from_addr(arg_idx + fx.Int64(row * idx_stride) * 4, num_records_bytes=K * 4)
        local = Vec(buffer_ops.buffer_load(ir_, tid * EPT, vec_width=EPT, dtype=fx.Int32))
        btr = buffer_ops.create_buffer_resource_from_addr(arg_bt + fx.Int64(row * bt_stride) * 4, num_records_bytes=bt_stride * 4)
        live, slots = [], []
        for i in range_constexpr(EPT):
            j = local[i]
            ok = j >= 0
            jj = ok.select(j, fx.Int32(0))
            page = fx.Int32(buffer_ops.buffer_load(btr, jj // page_size, vec_width=1, dtype=fx.Int32))
            live.append(ok)
            slots.append(page * block_stride + jj % page_size)

        se0 = [fx.Int32(_poll(own_pool + region0 + ((rank + j) % WORLD) * REGION_BYTES + SPEC_ELEM_OFF + fx.Int64(tid) * 4, T.i32)) for j in range_constexpr(1, WORLD)]
        hb = own_pool + region0 + ((rank + 1 + tid % (WORLD - 1)) % WORLD) * REGION_BYTES + SPEC_OFF
        hv = fx.Int64(_poll(hb, T.i64))
        spin4 = fx.Int32(0)
        pend = fx.Int32(0)
        for j in range_constexpr(1, WORLD):
            pend = pend + (se0[j - 1] == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
        while (pend > 0) & (spin4 < SPIN_LIMIT):
            spin4 = spin4 + 1
            pend = fx.Int32(0)
            for j in range_constexpr(1, WORLD):
                w_ = fx.Int32(_poll(own_pool + region0 + ((rank + j) % WORLD) * REGION_BYTES + SPEC_ELEM_OFF + fx.Int64(tid) * 4, T.i32))
                se0[j - 1] = (se0[j - 1] == fx.Int32(0)).select(w_, se0[j - 1])
                pend = pend + (se0[j - 1] == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
        stamp(arg_dbg, row, tid, 21)
        if tid < WORLD - 1:
            h_lo = hv.to(fx.Int32)
            h_hi = (hv >> 32).to(fx.Int32)
            spin1 = fx.Int32(0)
            while ((h_lo == fx.Int32(0)) | (h_hi == fx.Int32(0))) & (spin1 < SPIN_LIMIT):
                spin1 = spin1 + 1
                hv = fx.Int64(_poll(hb, T.i64))
                h_lo = hv.to(fx.Int32)
                h_hi = (hv >> 32).to(fx.Int32)
            s_acc[17 + tid] = h_lo
            s_acc[48 + tid] = (h_hi & fx.Int32(0xFFFF)) - 1
            s_acc[40 + tid] = (h_hi >> fx.Int32(16)) - 1
        _barrier()
        stamp(arg_dbg, row, tid, 2)
        lob = cut
        valid_t = tot_o
        prior_ninf = fx.Int32(0)
        for j in range_constexpr(1, WORLD):
            h = s_acc[16 + j]
            hc = h >> fx.Int32(16)
            lob = (hc > lob).select(hc, lob)
            valid_t = valid_t + s_acc[47 + j]
            prior_ninf = prior_ninf + (rank + j >= WORLD).select(s_acc[39 + j], fx.Int32(0))
        keys = [se0[j - 1] for j in range_constexpr(1, WORLD)] + [u[i] for i in range_constexpr(EPT)]
        oks = [se0[j - 1] != fx.Int32(-1) for j in range_constexpr(1, WORLD)] + [real[i] for i in range_constexpr(EPT)]
        for k in range_constexpr(len(keys)):
            bk = (keys[k] >> fx.Int32(HIST_SHIFT)) & fx.Int32(NB - 1)
            hit = oks[k] & (bk >= lob)
            lds_add(s_gh, hit.select(bk, fx.Int32(NB) + lane) + (lane & fx.Int32(GHC - 1)) * (NB + 64), fx.Int32(1))
        _barrier()
        stamp(arg_dbg, row, tid, 22)
        g0 = s_gh[2 * tid]
        g1 = s_gh[2 * tid + 1]
        for cpy in range_constexpr(1, GHC):
            g0 = g0 + s_gh[cpy * (NB + 64) + 2 * tid]
            g1 = g1 + s_gh[cpy * (NB + 64) + 2 * tid + 1]
        sumt = g0 + g1
        res0, valid = block_prefix(sumt, s_scan, 0)
        above_t = valid - res0 - sumt
        if (above_t < K) & (above_t + g1 >= K):
            s_acc[3] = 2 * tid + 1
            s_acc[4] = fx.Int32(K) - above_t
        if (above_t + g1 < K) & (above_t + sumt >= K):
            s_acc[3] = 2 * tid
            s_acc[4] = fx.Int32(K) - above_t - g1
        _barrier()
        stamp(arg_dbg, row, tid, 3)

        if (valid_t >= K) & (valid < K):
            hd = (c0 | fx.Int32(0x8000)) | ((c1 | fx.Int32(0x8000)) << fx.Int32(16))
            ninf_word = s_acc[8] | fx.Int32(0x8000)
            for j in range_constexpr(1, WORLD):
                buffer_ops.buffer_store(hd, dst1[j - 1], tid, cache_modifier=SYS)
                buffer_ops.buffer_store(ninf_word, dst1[j - 1], NB // 2, mask=tid == 0, cache_modifier=SYS)
            if tid < WORLD - 1:
                addr_f = own_pool + region0 + ((rank + 1 + tid) % WORLD) * REGION_BYTES
                val_f = fx.Int32(_poll(addr_f, T.i32))
                spin_f = fx.Int32(0)
                while ((val_f & fx.Int32(VALID2)) != fx.Int32(VALID2)) & (spin_f < SPIN_LIMIT):
                    spin_f = spin_f + 1
                    fly_rocdl.s_sleep(SLEEP)
                    val_f = fx.Int32(_poll(addr_f, T.i32))
            _barrier()
            f_g0 = c0
            f_g1 = c1
            for j in range_constexpr(1, WORLD):
                a_f = own_pool + region0 + ((rank + j) % WORLD) * REGION_BYTES + fx.Int64(tid) * 4
                w_f = fx.Int32(_poll(a_f, T.i32))
                spin_g = fx.Int32(0)
                while ((w_f & fx.Int32(VALID2)) != fx.Int32(VALID2)) & (spin_g < SPIN_LIMIT):
                    spin_g = spin_g + 1
                    w_f = fx.Int32(_poll(a_f, T.i32))
                f_g0 = f_g0 + (w_f & fx.Int32(0x7FFF))
                f_g1 = f_g1 + ((w_f >> fx.Int32(16)) & fx.Int32(0x7FFF))
            f_sum = f_g0 + f_g1
            f_res, f_valid = block_prefix(f_sum, s_scan, 0)
            f_above = f_valid - f_res - f_sum
            if (f_above < K) & (f_above + f_g1 >= K):
                s_acc[3] = 2 * tid + 1
                s_acc[4] = fx.Int32(K) - f_above
            if (f_above + f_g1 < K) & (f_above + f_sum >= K):
                s_acc[3] = 2 * tid
                s_acc[4] = fx.Int32(K) - f_above - f_g1
            if tid == 0:
                s_acc[15] = 1
            _barrier()
        bsel = s_acc[3]
        rem0 = s_acc[4]
        covered = bsel >= lob

        if valid_t < K:
            if tid == 0:
                s_state[12] = NINF_KEY
                s_state[13] = fx.Int32(K) - valid_t
                s_acc[1] = prior_ninf
        else:
            cnts = [fx.Int32(0) for p in range_constexpr(WORLD)]
            if covered:
                if tid == 0:
                    s_acc[12] = 1
            else:
                hits = [real[i] & (bins_i[i] == bsel) for i in range_constexpr(EPT)]
                nh = fx.Int32(0)
                for i in range_constexpr(EPT):
                    nh = nh + hits[i].select(fx.Int32(1), fx.Int32(0))
                ex, n_own = block_prefix(nh, s_scan, 16)
                pos = ex
                for j in range_constexpr(1, WORLD):
                    dst2 = buffer_ops.create_buffer_resource_from_addr(_sel((rank + (row + j - 1) % (WORLD - 1) + 1) % WORLD, pools) + region0 + rank * REGION_BYTES + BAND_OFF, num_records_bytes=16 + K * 4)
                    pj = ex
                    for i in range_constexpr(EPT):
                        buffer_ops.buffer_store(u[i], dst2, 4 + pj, mask=hits[i], cache_modifier=SYS)
                        pj = pj + hits[i].select(fx.Int32(1), fx.Int32(0))
                    buffer_ops.buffer_store(n_own + 1, dst2, 0, mask=tid == 0, cache_modifier=SYS)
                for i in range_constexpr(EPT):
                    if hits[i]:
                        s_own[pos] = u[i]
                    pos = pos + hits[i].select(fx.Int32(1), fx.Int32(0))
                if tid < WORLD - 1:
                    pq = (rank + 1 + tid) % WORLD
                    addr = own_pool + region0 + pq * REGION_BYTES + BAND_OFF
                    val = fx.Int32(_poll(addr, T.i32))
                    spin6 = fx.Int32(0)
                    while (val == fx.Int32(0)) & (spin6 < SPIN_LIMIT):
                        spin6 = spin6 + 1
                        fly_rocdl.s_sleep(SLEEP)
                        val = fx.Int32(_poll(addr, T.i32))
                _barrier()
                mx = fx.Int32(0)
                for p in range_constexpr(WORLD):
                    hdr = fx.Int32(_poll(own_pool + region0 + p * REGION_BYTES + BAND_OFF, T.i32))
                    spin9 = fx.Int32(0)
                    while ((hdr == fx.Int32(0)) & (fx.Int32(p) != rank)) & (spin9 < SPIN_LIMIT):
                        spin9 = spin9 + 1
                        hdr = fx.Int32(_poll(own_pool + region0 + p * REGION_BYTES + BAND_OFF, T.i32))
                    c_ = (fx.Int32(p) == rank).select(n_own, hdr - 1)
                    cnts[p] = c_
                    mx = (c_ > mx).select(c_, mx)
                if tid == 0:
                    for p in range_constexpr(WORLD):
                        s_acc[32 + p] = cnts[p]
                    s_acc[13] = 1
                    s_acc[14] = (mx + SEG - 1) // SEG
                _barrier()
            stamp(arg_dbg, row, tid, 5)
            cl = [s_acc[32 + p] for p in range_constexpr(WORLD)]
            nr = s_acc[14]
            prefix = bsel << fx.Int32(HIST_SHIFT)
            dmask = fx.Int32(((NB - 1) << HIST_SHIFT) - (1 << 32))
            rem = rem0
            for pi in range_constexpr(2):
                shift = 11 if pi == 0 else 0
                bins = s_b1 if pi == 0 else s_b2
                sb = 4 * pi
                scan_base = 32 + 16 * pi
                if covered:
                    acc_reg(bins, shift, dmask, prefix, keys, oks)
                else:
                    for r in range(fx.Int32(0), nr, fx.Int32(1)):
                        load_chunk(r, cl, s_cnt, s_lst, s_own)
                        accumulate(bins, shift, dmask, prefix)
                        _barrier()
                _barrier()
                lo = fx.Int32(2048) - (tid + 1) * 4
                vals = [bins[lo + i] for i in range_constexpr(4)]
                tsum = vals[0] + vals[1] + vals[2] + vals[3]
                above, _tot = block_prefix(tsum, s_scan, scan_base)
                if (above < rem) & (above + tsum >= rem):
                    run = above
                    for i in range_constexpr(3, -1, -1):
                        c = vals[i]
                        if (run < rem) & (run + c >= rem):
                            s_state[sb + 4] = prefix | ((lo + i) << fx.Int32(shift))
                            s_state[sb + 5] = dmask | fx.Int32(2047 << shift)
                            s_state[sb + 6] = rem - run
                        run = run + c
                _barrier()
                stamp(arg_dbg, row, tid, 23 + pi)
                prefix = s_state[sb + 4]
                dmask = s_state[sb + 5]
                rem = s_state[sb + 6]
            pr = fx.Int32(0)
            if covered:
                for j in range_constexpr(1, WORLD):
                    pr = pr + ((rank + j >= WORLD) & oks[j - 1] & (keys[j - 1] == prefix)).select(fx.Int32(1), fx.Int32(0))
            else:
                for r in range(fx.Int32(0), nr, fx.Int32(1)):
                    load_chunk(r, cl, s_cnt, s_lst, s_own)
                    for p in range_constexpr(WORLD):
                        cp = s_cnt[p]
                        key = s_lst[p * SEG + (tid < cp).select(tid, fx.Int32(0))]
                        pr = pr + ((fx.Int32(p) < rank) & (tid < cp) & (key == prefix)).select(fx.Int32(1), fx.Int32(0))
                    _barrier()
            incp = inclusive(pr)
            if lane == fx.Int32(63):
                lds_add(s_acc, 1, incp)
            stamp(arg_dbg, row, tid, 12)
            if tid == 0:
                s_state[12] = prefix
                s_state[13] = rem
        _barrier()
        stamp(arg_dbg, row, tid, 6)

        thr = s_state[12]
        room_all = s_state[13]
        own = u
        abv = [(live[i] & _ugt(own[i], thr)).select(fx.Int32(1), fx.Int32(0)) for i in range_constexpr(EPT)]
        eql = [(live[i] & (own[i] == thr)).select(fx.Int32(1), fx.Int32(0)) for i in range_constexpr(EPT)]
        na = abv[0]
        ne = eql[0]
        for i in range_constexpr(1, EPT):
            na = na + abv[i]
            ne = ne + eql[i]
        pack = (na << fx.Int32(PACK)) + ne
        bef, total = block_prefix(pack, s_scan, 64)
        room = room_all - s_acc[1]
        room = (room < 0).select(fx.Int32(0), room)
        tot_a = total >> fx.Int32(PACK)
        tot_e = total & fx.Int32((1 << PACK) - 1)
        count = tot_a + (tot_e < room).select(tot_e, room)
        offs_o, keeps = [], []
        ab = bef >> fx.Int32(PACK)
        eb = bef & fx.Int32((1 << PACK) - 1)
        for i in range_constexpr(EPT):
            keeps.append((abv[i] != 0) | ((eql[i] != 0) & (eb < room)))
            offs_o.append(ab + (eb < room).select(eb, room))
            ab = ab + abv[i]
            eb = eb + eql[i]

        if tid == NT - 1:
            old = _ticket(arg_ctl + 4)
            if old == rows - 1:
                _ptr(arg_ctl, T.i32)[0] = seq
                _ptr(arg_ctl + 4, T.i32)[0] = fx.Int32(0)
        if tid == 0:
            cr = buffer_ops.create_buffer_resource_from_addr(arg_counts + fx.Int64(row) * 8, num_records_bytes=8)
            buffer_ops.buffer_store(Vec.from_elements([count, seq], fx.Int32), cr, 0, cache_modifier=SYS)
        cntv = fx.Int32(0)
        if tid < row:
            addr = arg_counts + fx.Int64(tid) * 8
            v = fx.Int64(_poll(addr, T.i64))
            spin8 = fx.Int32(0)
            while (((v >> 32).to(fx.Int32) - seq) < 0) & (spin8 < SPIN_LIMIT):
                spin8 = spin8 + 1
                fly_rocdl.s_sleep(SLEEP)
                v = fx.Int64(_poll(addr, T.i64))
            cntv = v.to(fx.Int32)
        incc = inclusive(cntv)
        if lane == fx.Int32(63):
            lds_add(s_acc, 2, incc)
        _barrier()
        stamp(arg_dbg, row, tid, 7)
        base = s_acc[2]
        sl = buffer_ops.create_buffer_resource_from_addr(arg_slots, num_records_bytes=rows * K * 4)
        for i in range_constexpr(EPT):
            buffer_ops.buffer_store(slots[i], sl, base + offs_o[i], mask=keeps[i])
        if tid == 0:
            ip = buffer_ops.create_buffer_resource_from_addr(arg_indptr, num_records_bytes=(rows + 1) * 4)
            buffer_ops.buffer_store(base + count, ip, row + 1)
            if row == 0:
                buffer_ops.buffer_store(fx.Int32(0), ip, 0)
        for j in range_constexpr(1, WORLD):
            rrs = buffer_ops.create_buffer_resource_from_addr(own_pool + region0 + ((rank + j) % WORLD) * REGION_BYTES + SPEC_OFF, num_records_bytes=16 + SPEC_E * 4)
            if tid == 0:
                buffer_ops.buffer_store(Vec.from_elements([fx.Int32(0), fx.Int32(0)], fx.Int32), rrs, 0, cache_modifier=SYS)
            buffer_ops.buffer_store(fx.Int32(0), rrs, 4 + tid, cache_modifier=SYS)
        if s_acc[15] == 1:
            for j in range_constexpr(1, WORLD):
                rrj = buffer_ops.create_buffer_resource_from_addr(own_pool + region0 + ((rank + j) % WORLD) * REGION_BYTES, num_records_bytes=HIST_BYTES)
                buffer_ops.buffer_store(fx.Int32(0), rrj, tid, cache_modifier=SYS)
                if tid == 0:
                    buffer_ops.buffer_store(fx.Int32(0), rrj, NB // 2, cache_modifier=SYS)
        if s_acc[13] == 1:
            for p in range_constexpr(WORLD):
                rr_b = buffer_ops.create_buffer_resource_from_addr(own_pool + region0 + p * REGION_BYTES + BAND_OFF, num_records_bytes=16 + K * 4)
                if fx.Int32(p) != rank:
                    if tid == 0:
                        buffer_ops.buffer_store(fx.Int32(0), rr_b, 0, cache_modifier=SYS)
                    reset_words(rr_b, 4, s_acc[32 + p])

    @flyc.jit
    def launch_dcp8_topk_gather_merge_top2048(
            arg_scores: fx.Int64, arg_idx: fx.Int64, arg_bt: fx.Int64, arg_slots: fx.Int64, arg_indptr: fx.Int64,
            arg_ctl: fx.Int64, arg_counts: fx.Int64, arg_dbg: fx.Int64,
            p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64, p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64,
            rows: fx.Int32, rank: fx.Int32, scores_stride: fx.Int32, idx_stride: fx.Int32, bt_stride: fx.Int32,
            page_size: fx.Int32, block_stride: fx.Int32, stream: fx.Stream):
        kernel(arg_scores, arg_idx, arg_bt, arg_slots, arg_indptr, arg_ctl, arg_counts, arg_dbg,
               p0, p1, p2, p3, p4, p5, p6, p7,
               rows, rank, scores_stride, idx_stride, bt_stride, page_size, block_stride).launch(
            grid=(rows, 1, 1), block=(NT, 1, 1), stream=stream)
    return launch_dcp8_topk_gather_merge_top2048
