# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Per-row top-2048 of the prefill indexer's logits over each row's window ``[ks, ke)`` (gfx950, FlyDSL): threshold and stream.

``threshold``: one workgroup per row estimates the value above which about 4.6K (``TMID``) of the window's keys lie from
a weighted sample (the window's last 4096 and first 512 keys exactly, plus ~1% stratified 64-key blocks of the middle) and
writes it as a float.
``stream``: one workgroup per row streams its window once (nontemporal 16-byte loads, one chunk prefetched) and keeps every value
>= the threshold in an LDS list that is ascending in position by construction (each wave owns a contiguous slice of every chunk,
the waves' counts are exchanged through LDS once per chunk), then copies the list out with the count.
``retry``: a row whose count is outside [2048, CAP] is streamed again, up to twice, with a threshold re-ranked from the sample by
the observed ratio and the counts of two more thresholds, which let a second failure continue with a threshold whose exact count
lies in range. None of 9600 random rows retries; clustered data does more often.
``select.py`` takes the exact 2048th largest of each list and writes the positions; rows that all passes failed on are selected by
an exact radix over the window.

Windows of at most CAP keys skip the threshold (the list is the window).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm  # noqa: F401
from flydsl.expr import rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import T, as_ir_value

from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher

I, F, I64 = fx.Int32, fx.Float32, fx.Int64
INT_MIN = -2147483648
KMAXV = 0x7FFFFFFF
NUM_REC = 0x7FFFF000
K = 2048
CAP = 8192
NBIN = 4096
HSZA = 4608  # histogram words (4096 bins + 64 dummy, padded one word per 64, a multiple of 512)
HEADB = 8
TAILB = 64
NBMID = 152
NBALL = NBMID + HEADB + TAILB
SDIV = 80
TMID, TA, TC = 4608, 16128, 3456


def _update_dpp(old, src, ctrl, row_mask, bank_mask):
    return rocdl.update_dpp(I.ir_type, I(old).ir_value(), I(src).ir_value(), ctrl, row_mask, bank_mask, False)


def _readlane(v, lane):
    return I(rocdl.readlane(T.i32, I(v), I(lane)))


def _uniform(v):
    return I(rocdl.readfirstlane(T.i32, I(v)))


def _ballot(pred):
    return I64(rocdl.ballot(T.i64, pred))


def _popc(mask):
    return I(fx.math.ctpop(mask).to(I))


def _lane_prefix(mask):
    lo = mask.to(I)
    hi = (mask >> I64(32)).to(I)
    return I(rocdl.mbcnt_hi(T.i32, as_ir_value(hi), as_ir_value(I(rocdl.mbcnt_lo(T.i32, as_ir_value(lo), as_ir_value(I(0)))))))


def _key_to_float(key):
    return (key < 0).select(key ^ I(0x7FFFFFFF), key).bitcast(F)


def _float_to_key(x):
    bits = (F(x) + 0.0).bitcast(I)
    return (bits < 0).select(bits ^ I(0x7FFFFFFF), bits)


def _sts(field, idx, v):
    field.ptr[idx] = v


def _wave_scan(v, op):
    v = I(v)
    if op == "add":
        old = I(0)
    for kk in (1, 2, 4, 8):
        o = _update_dpp(old if op == "add" else v, v, 0x110 + kk, 0xF, 0xF)
        v = v + I(o) if op == "add" else (fx.max(v, I(o)) if op == "max" else fx.min(v, I(o)))
    for ctrl, rmask in ((0x142, 0xA), (0x143, 0xC)):
        o = _update_dpp(old if op == "add" else v, v, ctrl, rmask, 0xF)
        v = v + I(o) if op == "add" else (fx.max(v, I(o)) if op == "max" else fx.min(v, I(o)))
    return v


def _lsr(x, s):
    if isinstance(s, int):
        return x if s == 0 else (x >> s) & ((1 << (32 - s)) - 1)
    return (s == 0).select(x, (x >> s) & ((I(1) << (I(32) - s)) - 1))


def _pad(b):
    return b + (b >> 6)


def _zero_hist(smem, tx, nt):
    for i in range_constexpr(HSZA // nt):
        _sts(smem.hist, tx + nt * i, I(0))


def _hist_add(smem, idx, w):
    fx.atomic_add(fx.add_offset(smem.hist.ptr, _pad(idx)), w, syncscope="workgroup")


def _scan_hist(smem, lane, need_):
    """(bin, weight above, weight of the bin) of the bin holding the need_-th largest unit of weight."""
    ssum = I(smem.hist.ptr[(63 - lane) * 65])
    for t in range_constexpr(1, 64):
        ssum = ssum + I(smem.hist.ptr[(63 - lane) * 65 + t])
    incl = _wave_scan(ssum, "add")
    hit = _ballot(incl >= need_)
    lstar = I(fx.math.cttz(hit).to(I))
    excl = _readlane(incl - ssum, lstar)
    cbase = (63 - lstar) * 64
    c = I(smem.hist.ptr[cbase + (63 - lane) + (63 - lstar)])
    incl2 = _wave_scan(c, "add")
    cum = excl + incl2
    hit2 = _ballot(cum >= need_)
    l2 = I(fx.math.cttz(hit2).to(I))
    return cbase + 63 - l2, _readlane(cum - c, l2), _readlane(c, l2)


def _blk_reduce(smem, wave, nw, vals, ops):
    for q in range_constexpr(len(vals)):
        _sts(smem.wred, q * nw + wave, _readlane(_wave_scan(vals[q], ops[q]), 63))
    fx.gpu.barrier()
    outs = []
    for q in range_constexpr(len(vals)):
        acc = I(smem.wred.ptr[q * nw])
        for w in range_constexpr(1, nw):
            x = I(smem.wred.ptr[q * nw + w])
            acc = acc + x if ops[q] == "add" else (fx.max(acc, x) if ops[q] == "max" else fx.min(acc, x))
        outs.append(_uniform(acc))
    fx.gpu.barrier()
    return outs


def _sample_row(rsrc, row, ks, ke, ks_al, L, wave, g, l16, nw):
    """Keys (ordered ints, INT_MIN = absent) and weights of the row sample: the last TAILB and first HEADB blocks of the window at
    weight 1 and stratified jittered blocks of the middle at the weight of the blocks each stands for."""
    span = (ke - ks_al + 63) // 64
    mid_n = fx.max(span - HEADB - TAILB, I(1))
    ns = fx.max(I(2048), L // SDIV)
    nb_mid = fx.min(fx.min((ns + 63) // 64, I(NBMID)), mid_n)
    wmid = fx.max(mid_n // nb_mid, I(1))
    nlive = nb_mid + HEADB + TAILB
    nls = (NBALL + nw * 4 - 1) // (nw * 4)
    keys, wts = [], []
    for t in range_constexpr(nls):
        j = t * (nw * 4) + wave * 4 + g
        u0 = HEADB + (j * mid_n) // nb_mid
        wb = HEADB + ((j + 1) * mid_n) // nb_mid - u0
        h = row * 0x45D9F3B + j * 0x2C1B3C6D + 12345
        h = ((h >> 16) ^ h) * 0x45D9F3B
        h = ((h >> 16) ^ h) * 0x45D9F3B
        h = ((h >> 16) ^ h) & 0x7FFFFFFF
        is_mid = j < nb_mid
        is_head = (j >= nb_mid) & (j < nb_mid + HEADB)
        blk = is_mid.select(u0 + h % fx.max(wb, I(1)), is_head.select(j - nb_mid, span - TAILB + (j - nb_mid - HEADB)))
        wj = is_mid.select(wmid, I(1))
        live = j < nlive
        start = ks_al + blk * 64
        v = fx.Vector(buffer_ops.buffer_load(rsrc, start + l16 * 4, vec_width=4, dtype=F, mask=live))
        for k in range_constexpr(4):
            e = start + l16 * 4 + k
            ok = live & (e >= ks) & (e < ke)
            keys.append(ok.select(_float_to_key(v[k]), I(INT_MIN)))
        wts.append(wj)
    return keys, wts


def _thresholds(smem, tx, lane, wave, nt, keys, wts, t_mid, t_a, t_c, full=True):
    """(thr_mid, thr_a, thr_c) floats whose sampled weight above them is t_mid / t_a / t_c; one 12-bit round over the sample's
    value range for all three, a second one resolving the middle one."""
    nw = nt // 64
    nk = len(keys)
    kmax = keys[0]
    kmin = (keys[0] == INT_MIN).select(I(KMAXV), keys[0])
    for t in range_constexpr(1, nk):
        kmax = fx.max(kmax, keys[t])
        kmin = fx.min(kmin, (keys[t] == INT_MIN).select(I(KMAXV), keys[t]))
    wsum = I(0)
    for t in range_constexpr(nk):
        wsum = wsum + (keys[t] != INT_MIN).select(wts[t // 4], I(0))
    pmax, pmin, wtot = _blk_reduce(smem, wave, nw, [kmax, kmin, wsum], ["max", "min", "add"])
    t_mid = fx.max(fx.min(t_mid, wtot), I(1))
    t_a = fx.max(fx.min(t_a, wtot), I(1))
    t_c = fx.max(fx.min(t_c, wtot), I(1))
    dmax = pmax - pmin
    bl = I(32) - I(fx.math.ctlz(dmax | 1).to(I))
    sh = fx.max(bl - 12, I(0))
    _zero_hist(smem, tx, nt)
    fx.gpu.barrier()
    for t in range_constexpr(nk):
        kk = keys[t]
        _hist_add(smem, (kk != INT_MIN).select(_lsr(kk - pmin, sh), I(NBIN) + lane), wts[t // 4])
    fx.gpu.barrier()
    b0, ab0, _ = _scan_hist(smem, lane, t_mid)
    if full:
        ba, _, _ = _scan_hist(smem, lane, t_a)
        bc, _, _ = _scan_hist(smem, lane, t_c)
    fx.gpu.barrier()
    sh1 = fx.max(sh - 12, I(0))
    _zero_hist(smem, tx, nt)
    fx.gpu.barrier()
    for t in range_constexpr(nk):
        kk = keys[t]
        mem = (kk != INT_MIN) & (_lsr(kk - pmin, sh) == b0)
        _hist_add(smem, mem.select(_lsr(kk - pmin, sh1) - (b0 << (sh - sh1)), I(NBIN) + lane), wts[t // 4])
    fx.gpu.barrier()
    b1, _, _ = _scan_hist(smem, lane, t_mid - ab0)
    fx.gpu.barrier()
    thr_mid = _key_to_float(pmin + (((b0 << (sh - sh1)) + b1) << sh1))
    if full:
        return thr_mid, _key_to_float(pmin + (ba << sh)), _key_to_float(pmin + (bc << sh))
    return thr_mid, thr_mid, thr_mid


# =====================================================================================================================
# K1: per-row thresholds
# =====================================================================================================================
@launcher
def compile_threshold(NT=512):
    NW = NT // 64

    @fx.struct
    class Shared:
        hist: fx.Array[fx.Int32, HSZA]
        wred: fx.Array[fx.Int32, 64]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def k1(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_meta: fx.Int64, i64_stride: fx.Int64):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = _uniform(tx // 64)
        g = lane // 16
        l16 = lane % 16
        row = I(fx.block_idx.x)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        ks_r = buffer_ops.create_buffer_resource_from_addr(arg_ks)
        ke_r = buffer_ops.create_buffer_resource_from_addr(arg_ke)
        meta_r = buffer_ops.create_buffer_resource_from_addr(arg_meta)
        ks = _uniform(buffer_ops.buffer_load(ks_r, row, vec_width=1, dtype=I))
        ke = _uniform(buffer_ops.buffer_load(ke_r, row, vec_width=1, dtype=I))
        L = ke - ks
        ks_al = ks & I(-64)
        exact = L <= CAP
        ninf = F(float("-inf")).bitcast(I)
        if exact:
            if tx == 0:
                buffer_ops.buffer_store(fx.Vector.from_elements([ninf, ninf, ninf, I(0)], dtype=I), meta_r, row * 4)
        else:
            rsrc = buffer_ops.create_buffer_resource_from_addr(arg_lg + I64(row) * i64_stride * 4, num_records_bytes=i64_stride * 4)
            keys, wts = _sample_row(rsrc, row, ks, ke, ks_al, L, wave, g, l16, NW)
            thr_mid, thr_a, thr_c = _thresholds(smem, tx, lane, wave, NT, keys, wts, I(TMID), I(TA), I(TC), False)
            if tx == 0:
                buffer_ops.buffer_store(fx.Vector.from_elements([thr_mid.bitcast(I), thr_a.bitcast(I), thr_c.bitcast(I), I(0)], dtype=I), meta_r, row * 4)

    @flyc.jit
    def launch(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_meta: fx.Int64, i64_stride: fx.Int64,
               rows: fx.Int32, stream: fx.Stream):
        k1(arg_lg, arg_ks, arg_ke, arg_meta, i64_stride).launch(grid=(I64(rows), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


# =====================================================================================================================
# K2: one stream pass at the threshold of K1
# =====================================================================================================================
@launcher
def compile_stream(NT=512, V=4, NC_ST=2):
    NW = NT // 64
    CH = NT * 4 * V
    SW = V * 256
    LSZ = CAP + 64

    @fx.struct
    class Shared:
        lkey: fx.Array[fx.Int32, LSZ]
        lidx: fx.Array[fx.Int32, LSZ]
        cbuf: fx.Array[fx.Int32, 32]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def k2(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_meta: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64,
           i64_stride: fx.Int64, i32_cstride: fx.Int32, arg_order: fx.Int64):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = _uniform(tx // 64)
        row = I(fx.block_idx.x)
        if arg_order != I64(0):
            row = _uniform(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_order), row, vec_width=1, dtype=I))
        smem = fx.SharedAllocator().allocate(Shared).peek()

        ks_r = buffer_ops.create_buffer_resource_from_addr(arg_ks)
        ke_r = buffer_ops.create_buffer_resource_from_addr(arg_ke)
        meta_r = buffer_ops.create_buffer_resource_from_addr(arg_meta)
        ks = _uniform(buffer_ops.buffer_load(ks_r, row, vec_width=1, dtype=I))
        ke = _uniform(buffer_ops.buffer_load(ke_r, row, vec_width=1, dtype=I))
        thr = _uniform(buffer_ops.buffer_load(meta_r, row * 4, vec_width=1, dtype=I)).bitcast(F)
        L = ke - ks
        ks_al = ks & I(-64)
        rsrc = buffer_ops.create_buffer_resource_from_addr(arg_lg + I64(row) * i64_stride * 4, num_records_bytes=i64_stride * 4)
        nit = (ke - ks_al + (CH - 1)) // CH
        wlen = L.bitcast(fx.Uint32)

        def load_chunk(i, ok):
            return [buffer_ops.buffer_load(rsrc, ks_al + i * CH + wave * SW + (v * 64 + lane) * 4, vec_width=4, dtype=F,
                                           cache_modifier=NC_ST, mask=ok) for v in range_constexpr(V)]
        for i, state in range(0, nit, 1, init=load_chunk(I(0), nit > 0) + [I(0)]):
            cur = [fx.Vector(state[v]) for v in range_constexpr(V)]
            run0 = _uniform(state[V])
            nxt = load_chunk(I(i) + 1, (I(i) + 1) < nit)
            base = ks_al + I(i) * CH + wave * SW
            xs = [F(cur[v][k]) for v in range_constexpr(V) for k in range_constexpr(4)]
            if (I(i) == 0) | (I(i) == nit - 1):
                xs = [((base + (q // 4 * 64 + lane) * 4 + q % 4 - ks).bitcast(fx.Uint32) < wlen).select(xs[q], F(float("nan")))
                      for q in range_constexpr(V * 4)]
            hv, incls, tots = [], [], []
            wtot = I(0)
            for v in range_constexpr(V):
                hk = []
                cl = I(0)
                for k in range_constexpr(4):
                    hh = xs[v * 4 + k] >= thr
                    hk.append(hh)
                    cl = cl + hh.select(I(1), I(0))
                incl = _wave_scan(cl, "add")
                tv = _readlane(incl, 63)
                hv.append(hk)
                incls.append(incl - cl)
                tots.append(tv)
                wtot = wtot + tv
            par = (I(i) & 1) * 16
            if lane == 0:
                _sts(smem.cbuf, par + wave, wtot)
            fx.gpu.barrier()
            pref = I(0)
            itot = I(0)
            for w in range_constexpr(NW):
                cw = _uniform(smem.cbuf.ptr[par + w])
                pref = pref + (wave > w).select(cw, I(0))
                itot = itot + cw
            runv = run0 + pref
            for v in range_constexpr(V):
                pos = runv + incls[v]
                for k in range_constexpr(4):
                    wp = (hv[v][k] & (pos < CAP)).select(pos, I(CAP) + lane)
                    _sts(smem.lidx, wp, base + (v * 64 + lane) * 4 + k)
                    _sts(smem.lkey, wp, xs[v * 4 + k].bitcast(I))
                    pos = pos + hv[v][k].select(I(1), I(0))
                runv = runv + tots[v]
            res = yield [nxt[v] for v in range_constexpr(V)] + [run0 + itot]
        fx.gpu.barrier()
        count = _uniform(res[V])
        cand_r = buffer_ops.create_buffer_resource_from_addr(arg_cand, num_records_bytes=NUM_REC)
        if tx == 0:
            buffer_ops.buffer_store(count, buffer_ops.create_buffer_resource_from_addr(arg_cnt), row)
        if (count >= K) & (count <= CAP) | (L <= CAP):
            ncopy = fx.min(fx.min(count, I(CAP)), i32_cstride)
            for c in range(0, (ncopy + (NT - 1)) // NT, 1):
                j = tx + NT * I(c)
                buffer_ops.buffer_store(fx.Vector.from_elements([I(smem.lidx.ptr[j]), I(smem.lkey.ptr[j])], dtype=I), cand_r,
                                        (row * i32_cstride + j) * 2, mask=j < ncopy)

    @flyc.jit
    def launch(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_meta: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64,
               i64_stride: fx.Int64, i32_cstride: fx.Int32, arg_order: fx.Int64, rows: fx.Int32, stream: fx.Stream):
        k2(arg_lg, arg_ks, arg_ke, arg_meta, arg_cnt, arg_cand, i64_stride, i32_cstride, arg_order).launch(
            grid=(I64(rows), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


# =====================================================================================================================
# K2b: retry passes of the rows whose first pass count fell outside [2048, CAP]
# =====================================================================================================================
@launcher
def compile_retry(NT=512, V=4, NC_ST=2):
    NW = NT // 64
    CH = NT * 4 * V
    SW = V * 256
    LSZ = CAP + 64

    @fx.struct
    class Shared:
        lkey: fx.Array[fx.Int32, LSZ]
        lidx: fx.Array[fx.Int32, LSZ]
        hist: fx.Array[fx.Int32, HSZA]
        wred: fx.Array[fx.Int32, 64]
        cbuf: fx.Array[fx.Int32, 32]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def k2b(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64, i64_stride: fx.Int64,
            i32_cstride: fx.Int32):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = _uniform(tx // 64)
        g = lane // 16
        l16 = lane % 16
        row = I(fx.block_idx.x)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        ks_r = buffer_ops.create_buffer_resource_from_addr(arg_ks)
        ke_r = buffer_ops.create_buffer_resource_from_addr(arg_ke)
        cnt_r = buffer_ops.create_buffer_resource_from_addr(arg_cnt)
        ks = _uniform(buffer_ops.buffer_load(ks_r, row, vec_width=1, dtype=I))
        ke = _uniform(buffer_ops.buffer_load(ke_r, row, vec_width=1, dtype=I))
        cnt0 = _uniform(buffer_ops.buffer_load(cnt_r, row, vec_width=1, dtype=I))
        L = ke - ks
        bad = (L > CAP) & ((cnt0 < K) | (cnt0 > CAP))
        if bad:
            cand_r = buffer_ops.create_buffer_resource_from_addr(arg_cand, num_records_bytes=NUM_REC)
            ks_al = ks & I(-64)
            rsrc = buffer_ops.create_buffer_resource_from_addr(arg_lg + I64(row) * i64_stride * 4, num_records_bytes=i64_stride * 4)
            nit = (ke - ks_al + (CH - 1)) // CH
            wlen = L.bitcast(fx.Uint32)

            def blk_sum(vals):
                for q in range_constexpr(len(vals)):
                    _sts(smem.wred, q * NW + wave, vals[q])
                fx.gpu.barrier()
                outs = []
                for q in range_constexpr(len(vals)):
                    acc = I(smem.wred.ptr[q * NW])
                    for w in range_constexpr(1, NW):
                        acc = acc + I(smem.wred.ptr[q * NW + w])
                    outs.append(_uniform(acc))
                fx.gpu.barrier()
                return outs

            def load_chunk(i, ok):
                return [buffer_ops.buffer_load(rsrc, ks_al + i * CH + wave * SW + (v * 64 + lane) * 4, vec_width=4, dtype=F,
                                               cache_modifier=NC_ST, mask=ok) for v in range_constexpr(V)]

            def resample_thresholds(t_mid):
                keys, wts = _sample_row(rsrc, row, ks, ke, ks_al, L, wave, g, l16, NW)
                return _thresholds(smem, tx, lane, wave, NT, keys, wts, t_mid, (t_mid * TA) // TMID, (t_mid * TC) // TMID)

            def stream_pass(first_c, thr, thr_a, thr_c):
                for i, state in range(0, nit, 1, init=list(first_c) + [I(0), I(0), I(0)]):
                    cur = [fx.Vector(state[v]) for v in range_constexpr(V)]
                    run0 = _uniform(state[V])
                    na0 = _uniform(state[V + 1])
                    nc0 = _uniform(state[V + 2])
                    nxt = load_chunk(I(i) + 1, (I(i) + 1) < nit)
                    base = ks_al + I(i) * CH + wave * SW
                    hv, incls, tots = [], [], []
                    wtot = I(0)
                    ia = I(0)
                    ic = I(0)
                    for v in range_constexpr(V):
                        hk = []
                        cl = I(0)
                        for k in range_constexpr(4):
                            e = base + (v * 64 + lane) * 4 + k
                            inr = (e - ks).bitcast(fx.Uint32) < wlen
                            xv = F(cur[v][k])
                            hh = (xv >= thr) & inr
                            hk.append(hh)
                            cl = cl + hh.select(I(1), I(0))
                            ia = ia + _popc(_ballot((xv >= thr_a) & inr))
                            ic = ic + _popc(_ballot((xv >= thr_c) & inr))
                        incl = _wave_scan(cl, "add")
                        tv = _readlane(incl, 63)
                        hv.append(hk)
                        incls.append(incl - cl)
                        tots.append(tv)
                        wtot = wtot + tv
                    par = (I(i) & 1) * 16
                    if lane == 0:
                        _sts(smem.cbuf, par + wave, wtot)
                    fx.gpu.barrier()
                    pref = I(0)
                    itot = I(0)
                    for w in range_constexpr(NW):
                        cw = _uniform(smem.cbuf.ptr[par + w])
                        pref = pref + (wave > w).select(cw, I(0))
                        itot = itot + cw
                    runv = run0 + pref
                    for v in range_constexpr(V):
                        pos = runv + incls[v]
                        for k in range_constexpr(4):
                            wp = (hv[v][k] & (pos < CAP)).select(pos, I(CAP) + lane)
                            _sts(smem.lidx, wp, base + (v * 64 + lane) * 4 + k)
                            _sts(smem.lkey, wp, F(cur[v][k]).bitcast(I))
                            pos = pos + hv[v][k].select(I(1), I(0))
                        runv = runv + tots[v]
                    res = yield [nxt[v] for v in range_constexpr(V)] + [run0 + itot, na0 + ia, nc0 + ic]
                return res[V], res[V + 1], res[V + 2]

            r0 = fx.max(fx.min((I(TMID) * TMID) // fx.max(cnt0, I(1)), L), I(64))
            ft0 = resample_thresholds(r0)
            state0 = [r0, I(0), cnt0, ft0[0].bitcast(I), ft0[1].bitcast(I), ft0[2].bitcast(I)] + load_chunk(I(0), nit > 0)
            for att, st in range(0, 2, 1, init=state0):
                r_c = _uniform(st[0])
                done_c = _uniform(st[1])
                cnt_c = _uniform(st[2])
                tm_c = _uniform(st[3])
                ta_c = _uniform(st[4])
                tc_c = _uniform(st[5])
                first_c = [st[6 + q] for q in range_constexpr(V)]
                r_n = r_c
                done_n = done_c
                cnt_n = cnt_c
                tm_n = tm_c
                ta_n = ta_c
                tc_n = tc_c
                nf = list(first_c)
                if done_c == 0:
                    cnt_n, na_w, nc_w = stream_pass(first_c, tm_c.bitcast(F), ta_c.bitcast(F), tc_c.bitcast(F))
                    fx.gpu.barrier()
                    cnt_n = _uniform(cnt_n)
                    n_a, n_c = blk_sum([na_w, nc_w])
                    ok_n = (cnt_n >= K) & (cnt_n <= CAP)
                    done_n = ok_n.select(I(1), I(0))
                    use_c = (cnt_n > CAP) & (n_c >= K) & (n_c <= CAP)
                    use_a = (cnt_n < K) & (n_a >= K) & (n_a <= CAP)
                    tm_n = use_c.select(tc_c, use_a.select(ta_c, tm_c))
                    ta_n = tm_n
                    tc_n = tm_n
                    r_n = fx.max(fx.min((r_c * TMID) // fx.max(cnt_n, I(1)), L), I(64))
                    if (~ok_n) & (~(use_c | use_a)):
                        ft = resample_thresholds(r_n)
                        tm_n = ft[0].bitcast(I)
                        ta_n = ft[1].bitcast(I)
                        tc_n = ft[2].bitcast(I)
                    nf = load_chunk(I(0), (~ok_n) & (nit > 0))
                    fx.gpu.barrier()
                att_res = yield [r_n, done_n, cnt_n, tm_n, ta_n, tc_n] + list(nf)
            count = _uniform(att_res[2])
            if tx == 0:
                buffer_ops.buffer_store(count, cnt_r, row)
            if _uniform(att_res[1]) != 0:
                ncopy = fx.min(fx.min(count, I(CAP)), i32_cstride)
                for c in range(0, (ncopy + (NT - 1)) // NT, 1):
                    j = tx + NT * I(c)
                    buffer_ops.buffer_store(fx.Vector.from_elements([I(smem.lidx.ptr[j]), I(smem.lkey.ptr[j])], dtype=I), cand_r,
                                            (row * i32_cstride + j) * 2, mask=j < ncopy)

    @flyc.jit
    def launch(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64, i64_stride: fx.Int64,
               i32_cstride: fx.Int32, rows: fx.Int32, stream: fx.Stream):
        k2b(arg_lg, arg_ks, arg_ke, arg_cnt, arg_cand, i64_stride, i32_cstride).launch(
            grid=(I64(rows), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


