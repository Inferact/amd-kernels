# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""vLLM's one-program-per-row top-p kernel (``_topk_topp_kernel``, top-k off) on more than 64 rows, bit for bit
(``reference.py``, ``top_p_mask_rows_reference``).

One 512-thread workgroup per row, as Triton's program, so every ``tl.sum`` keeps its layout: thread t holds columns
2048 r + 4 t + j of an 8192-column tile, a 4096-position chunk of the compacted buffer 2048 r + 4 t + j for r < 2.
The tile and chunk sums are the in-thread pairs, the wave's DPP rows and the waves' LDS tree (``reference.py``
offsets); the 25 tile sums of a pass meet in LDS after the pass and are chained in order.

Pass 1 stores e = exp(x - max_sample) in the workgroup's workspace slice. Division by the row's exp sum is monotone,
so the outliers (e / se > outlier probability) are exactly e >= e_min, the least such e (found once by a few exact
divisions around outlier probability * se): pass 2 needs no division. It compacts the outliers' e in place (a tile's
positions lie below its columns, which every wave has loaded by the tile's barrier), keeps the maximum of every
256-position block and zeroes the rest of the last chunk (search loads read whole chunks, unmasked). Pass 2's
outlier mass decides the search's branch from a sum with a 1e-4 relative margin (its error bound is under 3e-5);
inside the margin a pass sums Triton's exact probabilities. A search round evaluates only the (chunk, wave) groups
whose maximum probability exceeds the pivot (a group below adds exactly what Triton's masked sum adds: 0.0, the 1.0
sentinel, a count of 0); the K groups of largest maximum per wave stay in registers as probabilities, the others are
loaded in batches and divided on first use (written back: each wave owns its groups). Minima of probabilities are
integer minima of their bit patterns (non-negative). Rows whose exp sum or outlier probability is not a positive
finite number take Triton's division per element. The mask pass writes -inf without reading a tile whose maximum
cannot be kept.

Names assigned inside a dynamic loop are unique in the kernel: FlyDSL carries every name a loop body assigns that
an enclosing scope also binds.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .kernel import EPS, LN2_HI, LN2_LO, _bf, _bi, _exp, _fbits, _fmax, _fmin, _launder, _lane_val, _popc, _uni
from .kernel import _dpp, _wmax, _wmin, _wsum
from .reference import CHUNK, ITERS, TILE, TILES, V

F, I, I64 = fx.Float32, fx.Int32, fx.Int64
NT = 512
INF = float("inf")
NCH = -(-V // CHUNK)  # 49
NBLK = -(-V // 256)  # 782
PF = 2  # tiles in flight per pass
K = 4  # groups per wave held in registers through the search
BATCH = 4  # groups past them loaded together
LN2X32 = _fbits(0x41B17218)  # fp32 32 ln2
SIGN = -(2**31)
PART = NCH * 8  # search partials per parity: [chunk][wave]
MARGIN = 1e-4  # relative margin of the outlier-mass decision


def _sqrt(x):
    """v_sqrt_f32 (Triton's tl.sqrt on gfx950)."""
    return F(_llvm.call_intrinsic(T.f32, "llvm.amdgcn.sqrt.f32", [fx.as_ir_value(x)], [], []))


def _log_any(x):
    """Triton's fp32 ``tl.log`` for any x (below 2^-126 scaled by 2^32, minus 32 ln2; inf / nan pass through)."""
    small = x < F(_fbits(0x00800000))
    y = fx.math.log2(small.select(x * F(4294967296.0), x), fastmath=fx.FastMathFlags.afn)
    r0 = _launder(y * F(LN2_HI))
    f0 = fx.math.fma(y, F(LN2_HI), r0 * F(-1.0))
    f1 = fx.math.fma(y, F(LN2_LO), f0)
    r = fx.math.fma(y, F(LN2_HI), f1)
    r = (fx.math.absf(y) < F(INF)).select(r, y)
    return r - small.select(F(LN2X32), F(0.0))


def _wtree8(w):
    """Eight wave values summed as Triton's cross-wave butterfly (4, 2, 1)."""
    return ((w[0] + w[4]) + (w[2] + w[6])) + ((w[1] + w[5]) + (w[3] + w[7]))


def _tree16(e):
    """In-thread sum of an 8192-tile's 16 registers e[4 r + j] (offsets 2, 2048, 4096, 1)."""
    def half(j0):
        return (e[j0] + e[j0 + 2] + (e[4 + j0] + e[6 + j0])) + (e[8 + j0] + e[10 + j0] + (e[12 + j0] + e[14 + j0]))
    return half(0) + half(1)


def _tree8(e):
    """In-thread sum of a 4096-chunk's 8 registers e[4 r + j] (offsets 2, 2048, 1)."""
    return ((e[0] + e[2]) + (e[4] + e[6])) + ((e[1] + e[3]) + (e[5] + e[7]))


def _ult(a, b):
    """Unsigned a < b on i32 bit patterns."""
    return (a ^ I(SIGN)) < (b ^ I(SIGN))


def _u2f(v):
    """uint32 -> fp32 (uitofp) of an i32 bit pattern."""
    return F(_arith.uitofp(T.f32, fx.as_ir_value(v)))


def _ballot(pred):
    return I64(fx.rocdl.ballot(T.i64, pred))


def _mbcnt(bal):
    """Lanes below this one set in ``bal``."""
    mb = I(_llvm.call_intrinsic(T.i32, "llvm.amdgcn.mbcnt.lo", [bal.to(I).ir_value(), I(0).ir_value()], [], []))
    return I(_llvm.call_intrinsic(T.i32, "llvm.amdgcn.mbcnt.hi", [(bal >> I64(32)).to(I).ir_value(), mb.ir_value()],
                                  [], []))


def _lds_ptr(addr):
    return _llvm.inttoptr(_ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(I64(addr)))


def _lds_umax(addr, v):
    _llvm.AtomicRMWOp(_llvm.AtomicBinOp.umax, _lds_ptr(addr), I(v).ir_value(), _llvm.AtomicOrdering.monotonic,
                      syncscope="workgroup", alignment=4)


def _lds_load8(addr):
    """8 dwords at LDS byte address ``addr`` (16-byte aligned)."""
    out = []
    for h in range(2):
        v = Vec(_llvm.load(T.vec(4, T.i32), _lds_ptr(addr + I(16 * h)), alignment=16))
        out += [I(v[k]) for k in range(4)]
    return out


def _imax(a, b):
    return (a > b).select(a, b)


def _imin(a, b):
    return (a < b).select(a, b)


def _wmin_bits(v):
    """Wave minimum of non-negative fp32 bit patterns (integer min), uniform; lane 63's DPP chain stays in range."""
    for ctrl in (0x118, 0x114, 0x112, 0x111):
        v = _imin(v, _dpp(v, ctrl))
    v = _imin(v, _dpp(v, 0x142, old=v, row_mask=0xA))
    v = _imin(v, _dpp(v, 0x143))
    return _lane_val(v, 63)


def _wmax_bits(v):
    """Wave maximum of non-negative fp32 bit patterns (integer max; out-of-row DPP lanes read 0), uniform."""
    for ctrl in (0x118, 0x114, 0x112, 0x111):
        v = _imax(v, _dpp(v, ctrl))
    v = _imax(v, _dpp(v, 0x142, old=v, row_mask=0xA))
    v = _imax(v, _dpp(v, 0x143))
    return _lane_val(v, 63)


def _scan32(v):
    """Inclusive prefix sum over lanes 0-31 (integer, DPP rows then the row-15 broadcast)."""
    for ctrl in (0x111, 0x112, 0x114, 0x118):
        v = v + _dpp(v, ctrl)
    return v + _dpp(v, 0x142, old=I(0), row_mask=0xA)


def _cols(i, tx):
    """Column of each of a thread's 16 registers in tile i."""
    return [i * I(TILE) + I(2048 * (k // 4) + k % 4) + tx * I(4) for k in range(16)]


def _load_tile(rsrc, i, tx, fill):
    """Tile i's 16 values of this thread, ``fill`` past V (a whole tile past V loads nothing)."""
    out = []
    for r in range(4):
        off = i * I(TILE) + I(2048 * r) + tx * I(4)
        v = Vec(buffer_ops.buffer_load(rsrc, off, vec_width=4, dtype=I))
        for j in range(4):
            out.append(((off + I(j)) < I(V)).select(I(v[j]).bitcast(F), F(fill)))
    return out


def _load_raw(rsrc, i, tx):
    """Tile i's 16 values of this thread as loaded (0 past V)."""
    out = []
    for r in range(4):
        v = Vec(buffer_ops.buffer_load(rsrc, i * I(TILE) + I(2048 * r) + tx * I(4), vec_width=4, dtype=I))
        out += [I(v[j]).bitcast(F) for j in range(4)]
    return out


def _last_tile(vals, tx, fill):
    """The last tile's values with ``fill`` past V."""
    cols = _cols(I(TILES - 1), tx)
    return [(cols[k] < I(V)).select(vals[k], F(fill)) for k in range(16)]


def _store_tile(rsrc, vals, i, tx):
    for r in range(4):
        vec = Vec.from_elements([_bi(vals[4 * r + j]) for j in range(4)], I)
        buffer_ops.buffer_store(vec, rsrc, i * I(TILE) + I(2048 * r) + tx * I(4))


def _pass1_tile(xt, ms):
    """A tile's e, their in-thread sum and the max (the last tile's columns past V are -inf: e 0)."""
    e = [_exp(v - ms) for v in xt]
    mx = F(-INF)
    for v in xt:
        mx = _fmax(mx, v)
    return e, _tree16(e), mx


def _pass1_last(raw, ms, tx):
    """The last tile: -inf past V, e 0 there."""
    xl = _last_tile(raw, tx, -INF)
    cols = _cols(I(TILES - 1), tx)
    e = [(cols[k] < I(V)).select(_exp(xl[k] - ms), F(0.0)) for k in range(16)]
    mx = F(-INF)
    for v in xl:
        mx = _fmax(mx, v)
    return e, _tree16(e), mx


def _part_counts(flags):
    """Per 2048-column part r: the flagged registers of the lanes below this one, and of the wave."""
    before, tot = [], []
    for r in range(4):
        b = I(0)
        t = I(0)
        for j in range(4):
            bal = _ballot(flags[4 * r + j])
            b = b + _mbcnt(bal)
            t = t + I(fx.math.ctpop(bal).to(I))
        before.append(b)
        tot.append(t)
    return before, tot


def _outliers_fast(et, e_min):
    """A tile's outlier flags (e >= e_min > 0; e is 0 past V), this thread's outlier e sum (any order), and the part
    counts."""
    om = [v >= e_min for v in et]
    acc = F(0.0)
    for k in range(16):
        acc = acc + om[k].select(et[k], F(0.0))
    before, tot = _part_counts(om)
    return om, acc, before, tot


def _outliers_exact(xt, ms, se, oprob, i, tx):
    """Triton's probabilities of tile i, the outlier flags, the in-thread outlier sum and the part counts."""
    cols = _cols(i, tx)
    pr = [_exp(v - ms) / se for v in xt]
    om = [(pr[k] > oprob) & (cols[k] < I(V)) for k in range(16)]
    ts = _tree16([om[k].select(pr[k], F(0.0)) for k in range(16)])
    before, tot = _part_counts(om)
    return pr, om, ts, before, tot


def _write_counts(arr, par, wave, tot):
    for r in range(4):
        arr.ptr[par + I(8 * r) + wave] = tot[r]


def _bases(arr, par, run, lane, wave):
    """Per part r: the first position of this wave's run and its length (from the tile's 32 wave counts [r][w] in
    LDS, prefix-summed across lanes); and the new running total."""
    cw = (lane < I(32)).select(I(arr.ptr[par + (lane & I(31))]), I(0))
    inc = _scan32(cw)
    exc = inc - cw
    base = [run + _lane_val(exc, I(8 * r) + wave) for r in range(4)]
    own = [_lane_val(cw, I(8 * r) + wave) for r in range(4)]
    return base, own, run + _lane_val(inc, I(31))


def _compact(buf, vals, om, before, base):
    """Store the outliers at their compacted positions; per part the maximum of this wave's run (fp32 bits)."""
    runs = []
    for r in range(4):
        cnt = I(0)
        m = I(0)
        for j in range(4):
            k = 4 * r + j
            buffer_ops.buffer_store(_bi(vals[k]), buf, om[k].select(base[r] + before[r] + cnt, I(V)))
            m = _imax(m, (om[k] & (vals[k] > F(0.0))).select(_bi(vals[k]), I(0)))
            cnt = cnt + om[k].select(I(1), I(0))
        runs.append(_wmax_bits(m))
    return runs


def _block_maxima(bmax_base, runs, base, own):
    """Lane 0: fold each run's maximum into the 256-position blocks it touches (an upper bound per block)."""
    for r in range(4):
        b_lo = base[r] >> I(8)
        b_hi = (base[r] + own[r] - I(1)) >> I(8)
        a = (own[r] > I(0)).select(bmax_base + b_lo * I(4), bmax_base + I((NBLK + 2) * 4))
        _lds_umax(a, runs[r])
        _lds_umax((b_hi != b_lo).select(a + I(4), bmax_base + I((NBLK + 2) * 4)), runs[r])


def _load_group(buf, q, wave, lane, b1):
    """Group (chunk q, wave): this lane's 8 (compacted) or 16 (full) registers; the buffer reads 0 past the data
    (zero-filled to the chunk end, or past V)."""
    vals = []
    size = CHUNK if b1 else TILE
    for r in range(2 if b1 else 4):
        v = Vec(buffer_ops.buffer_load(buf, q * I(size) + I(2048 * r) + wave * I(256) + lane * I(4), vec_width=4,
                                       dtype=I))
        vals += [I(v[j]).bitcast(F) for j in range(4)]
    return vals


def _store_group(buf, q, wave, lane, vals, b1):
    size = CHUNK if b1 else TILE
    for r in range(2 if b1 else 4):
        vec = Vec.from_elements([_bi(vals[4 * r + j]) for j in range(4)], I)
        buffer_ops.buffer_store(vec, buf, q * I(size) + I(2048 * r) + wave * I(256) + lane * I(4))


def _bit(mask, q):
    return ((mask >> I64(q)) & I64(1)) != I64(0)


def _probs(vals, se, div):
    """Triton's probabilities of stored e (``div``) or stored probabilities."""
    return [div.select(v / se, v) for v in vals]


def _group_eval(vals, piv, b1):
    """The wave's share of Triton's chunk statistics at ``piv``: the masked sum, the minimum above (1.0 if none; an
    integer min of the non-negative bit patterns) and the count within 1e-9 of that minimum."""
    ab = [v > piv for v in vals]
    contrib = [ab[k].select(vals[k], F(0.0)) for k in range(len(vals))]
    sw = _wsum(_tree8(contrib) if b1 else _tree16(contrib))
    m = I(0x3F800000)
    for k in range(len(vals)):
        m = _imin(m, ab[k].select(_bi(vals[k]), I(0x3F800000)))
    mw = _bf(_wmin_bits(m))
    c = I(0)
    for k in range(len(vals)):
        c = c + _popc(ab[k] & (fx.math.absf(vals[k] - mw) < F(EPS)))
    return sw, mw, c


def _group_count(vals, piv, mref):
    c = I(0)
    for v in vals:
        c = c + _popc((v > piv) & (fx.math.absf(v - mref) < F(EPS)))
    return c


def _write_part(smem, o, s, m, c):
    smem.psum.ptr[o] = _bi(s)
    smem.pmin.ptr[o] = _bi(m)
    smem.pcnt.ptr[o] = c


def _chunk_stats(ps, pm, pc, o, lane, nq):
    """Lane q: chunk q's sum (the waves' tree), minimum, count, and whether a wave minimum lies within 1e-9 of the
    chunk minimum without being equal (the count must then be taken against the chunk minimum). ``ps``, ``pm``,
    ``pc``: LDS byte addresses of the [chunk][wave] sums, minima, counts; ``o`` the byte offset of chunk q's row."""
    ws = [_bf(v) for v in _lds_load8(ps + o)]
    wm = [_bf(v) for v in _lds_load8(pm + o)]
    wc = _lds_load8(pc + o)
    m = F(1.0)
    for w in range(8):
        m = _fmin(m, wm[w])
    n = I(0)
    near = I(0)
    for w in range(8):
        n = n + (wm[w] == m).select(wc[w], I(0))
        near = near | ((wm[w] != m) & (fx.math.absf(wm[w] - m) < F(EPS))).select(I(1), I(0))
    ok = lane < nq
    return _wtree8(ws), ok.select(m, F(1.0)), ok.select(n, I(0)), ok.select(near, I(0))


def _sum_rec(arr, o):
    n = I(0)
    for w in range(8):
        n = n + I(arr.ptr[o + I(w)])
    return n


def _merge(s0, m0, n0, sq, mq, nq):
    """One step of Triton's chunk loop: the sum, and ``_update_min_larger_stats``."""
    n0 = (mq < m0).select(nq, n0 + (fx.math.absf(mq - m0) < F(EPS)).select(nq, I(0)))
    return s0 + sq, _fmin(m0, mq), n0


def _decide(st, o, s0, m0, n0, piv, p, lo, hi, ppiv, mlp, nml, psum, it):
    """One round's end in Triton's order (the found test, the bracket update, the stop test) into LDS ``st[o:o+7]``:
    lo, hi, done, ppiv, mlp, nml, psum."""
    found = (s0 >= p) & (fx.math.fma(m0 * F(-1.0), _u2f(n0), s0) < p)
    ppiv = found.select(piv, ppiv)
    mlp = found.select(m0, mlp)
    nml = found.select(n0, nml)
    psum = found.select(s0, psum)
    lo2 = (s0 > p).select(piv, lo)
    hi2 = (~(s0 > p) & (s0 < p)).select(piv, hi)
    stop = ((hi2 - lo2) < F(EPS)) | (it == I(ITERS - 1))
    ppiv = stop.select((hi2 + lo2) / F(2.0), ppiv)
    mlp = stop.select(m0, mlp)
    nml = stop.select(n0, nml)
    psum = stop.select(s0, psum)
    for k, v in enumerate((_bi(lo2), _bi(hi2), (found | stop).select(I(1), I(0)), _bi(ppiv), _bi(mlp), nml,
                           _bi(psum))):
        st.ptr[o + I(k)] = v


def _take(mask):
    """The lowest BATCH set lanes of ``mask`` (64 for none: a group past V loads zeros) and the rest of the mask."""
    qs = []
    for _ in range(BATCH):
        qs.append((mask != I64(0)).select(I(fx.math.cttz(mask).to(I)), I(64)))
        mask = mask & (mask - I64(1))
    return qs, mask


def _chain_start():
    return F(0.0), F(1.0), I(0)


def _publish_near(smem, o, lane, near, mq):
    """Wave 0: the chunks needing a recount (a 64-bit lane mask in ``st[o+8:o+10]``) and every chunk minimum."""
    bal = _ballot(near != I(0))
    smem.st.ptr[o + I(8)] = bal.to(I)
    smem.st.ptr[o + I(9)] = (bal >> I64(32)).to(I)
    smem.cmin.ptr[lane] = _bi(mq)


def _read_flags(st, o):
    lo = _uni(I(st.ptr[o + I(8)]))
    hi = _uni(I(st.ptr[o + I(9)]))
    return (I64(hi) << I64(32)) | (I64(lo) & I64(0xFFFFFFFF))


def _read_state(st, o):
    w = [I(st.ptr[o + I(k)]) for k in range(7)]
    return (_uni(_bf(w[0])), _uni(_bf(w[1])), _uni(w[2]), _uni(_bf(w[3])), _uni(_bf(w[4])), _uni(w[5]),
            _uni(_bf(w[6])))


def _e_threshold(se, oprob):
    """(e_min, ok): the least fp32 e with fl(e / se) > oprob, from fl(oprob * se) and at most 4 steps either way;
    ok only for a positive finite se and oprob and a walk that ends at the boundary."""
    def q(e):
        return e / se

    def nxt(e, d):
        return _bf(_bi(e) + I(d))

    e = oprob * se
    for _ in range(4):
        c = nxt(e, -1)
        e = ((c > F(0.0)) & (q(c) > oprob)).select(c, e)
    for _ in range(4):
        e = (q(e) <= oprob).select(nxt(e, 1), e)
    fin = (se > F(0.0)) & (se < F(INF)) & (oprob > F(0.0)) & (oprob < F(INF))
    ok = fin & (e > F(_fbits(0x00800000))) & (e < F(INF)) & (q(e) > oprob) & ~(q(nxt(e, -1)) > oprob)
    return e, ok


def _dup_flags(xt, dup, i, tx):
    cols = _cols(i, tx)
    dm = [(fx.math.absf(xt[k] - dup) < F(EPS)) & (cols[k] < I(V)) for k in range(16)]
    before, tot = _part_counts(dm)
    return dm, before, tot


def _masked(xt, keep):
    return [keep[k].select(xt[k], F(-INF)) for k in range(16)]


def _keep(xt, fpiv):
    return [v > fpiv for v in xt]


def _keep_ties(keep, dm, before, base, nkeep):
    out = list(keep)
    for r in range(4):
        own = I(0)
        for j in range(4):
            k = 4 * r + j
            own = own + dm[k].select(I(1), I(0))
            out[k] = keep[k] & ~(dm[k] & ((base[r] + before[r] + own) > nkeep))
    return out


def _probs_tile(xt, ms, se, i, tx):
    """Branch 2: Triton's probabilities of tile i, their maximum (columns past V excluded) and the finite min."""
    cols = _cols(i, tx)
    pr = [_exp(v - ms) / se for v in xt]
    gm = F(0.0)
    mn = F(INF)
    for k in range(16):
        gm = _fmax(gm, ((pr[k] > F(0.0)) & (cols[k] < I(V))).select(pr[k], F(0.0)))
        mn = _fmin(mn, (xt[k] > F(-INF)).select(xt[k], F(INF)))
    return pr, gm, mn


@launcher
def compile_top_p_mask_rows(path: str = "auto"):
    """``path`` (tests): "exact" takes Triton's division per element on every row, "mass" sums Triton's exact outlier
    mass on every row."""
    assert path in ("auto", "exact", "mass")
    name = "top_p_mask_rows_f32_v200064" + ("" if path == "auto" else "_" + path)

    @fx.struct
    class Shared:
        cnt: fx.Array[fx.Int32, 2 * 32]
        psum: fx.Array[fx.Int32, 2 * PART]
        pmin: fx.Array[fx.Int32, 2 * PART]
        pcnt: fx.Array[fx.Int32, 2 * PART]
        wp1: fx.Array[fx.Int32, TILES * 8]
        wp2: fx.Array[fx.Int32, TILES * 8]
        tmx: fx.Array[fx.Int32, TILES * 8]
        red: fx.Array[fx.Int32, 6 * 8]
        bmax: fx.Array[fx.Int32, NBLK + 4]
        rec: fx.Array[fx.Int32, 2 * NCH * 8]
        st: fx.Array[fx.Int32, 2 * 16]
        cmin: fx.Array[fx.Int32, 64]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # fp32 [n, >= V] logits, masked in place
        i32_sx: fx.Int32,  # row stride (elements)
        arg_p: fx.Int64,  # fp32 [n] top-p
        arg_tab: fx.Int64,  # fp32 [200] sigma table
        arg_buf: fx.Int64,  # fp32 [grid, V] workspace
        i32_n: fx.Int32,
        i32_grid: fx.Int32,
    ):
        tx = I(fx.thread_idx.x)
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        pid = I(fx.block_idx.x)
        cnt_base = I(fx.ptrtoint(smem.cnt.ptr))
        ps_base = I(fx.ptrtoint(smem.psum.ptr))
        pm_base = I(fx.ptrtoint(smem.pmin.ptr))
        pc_base = I(fx.ptrtoint(smem.pcnt.ptr))
        bmax_base = I(fx.ptrtoint(smem.bmax.ptr))
        buf = buffer_ops.create_buffer_resource_from_addr(arg_buf + I64(pid) * (V * 4), num_records_bytes=V * 4)
        tab = buffer_ops.create_buffer_resource_from_addr(arg_tab, num_records_bytes=200 * 4)
        is_l0 = lane == 0

        for row in range(pid, i32_n, I(i32_grid)):
            rw = I(row)
            pres = buffer_ops.create_buffer_resource_from_addr(arg_p + I64(rw) * 4, num_records_bytes=4)
            p = _uni(I(buffer_ops.buffer_load(pres, 0, vec_width=1, dtype=I)).bitcast(F))
            xr = buffer_ops.create_buffer_resource_from_addr(arg_x + I64(rw) * I64(i32_sx) * 4,
                                                             num_records_bytes=V * 4)
            if p < F(1.0):
                # ---- tile 0 statistics -> max_sample
                pend1 = [_load_raw(xr, I(t_), tx) for t_ in range_constexpr(PF)]
                x0 = pend1[0]
                fl = [(v > F(-INF)).select(v, F(0.0)) for v in x0]
                nfw = I(0)
                for v in x0:
                    nfw = nfw + _popc(v > F(-INF))
                s1w = _wsum(_tree16(fl))
                s2w = _wsum(_tree16([_launder(v * v) for v in fl]))
                if is_l0:
                    smem.red.ptr[wave] = _bi(s1w)
                    smem.red.ptr[8 + wave] = _bi(s2w)
                    smem.red.ptr[16 + wave] = nfw
                fx.gpu.barrier()
                s1 = _uni(_wtree8([_bf(I(smem.red.ptr[w_])) for w_ in range_constexpr(8)]))
                s2 = _uni(_wtree8([_bf(I(smem.red.ptr[8 + w_])) for w_ in range_constexpr(8)]))
                nf = I(0)
                for w_ in range_constexpr(8):
                    nf = nf + I(smem.red.ptr[16 + w_])
                nf = _uni(nf)
                has = nf > I(0)
                avg = has.select(s1 / nf.to(F), F(0.0))
                sq = has.select(s2 / nf.to(F), F(0.0))
                std = _uni(_sqrt(_fmax(fx.math.fma(avg * F(-1.0), avg, sq), F(0.0))))
                ms = _uni(fx.math.fma(std, F(10.0), avg))

                # ---- pass 1: e to the workspace, tile sums of e (chained after the pass), max, finite min, tile max
                for t1 in range(TILES - 1):
                    i1 = I(t1)
                    xt1 = pend1[0]
                    pend1 = pend1[1:] + [_load_raw(xr, i1 + I(PF), tx)]
                    e1, ts1, tmx1 = _pass1_tile(xt1, ms)
                    _store_tile(buf, e1, i1, tx)
                    sw1 = _wsum(ts1)
                    mw1 = _wmax(tmx1)
                    if is_l0:
                        smem.wp1.ptr[i1 * I(8) + wave] = _bi(sw1)
                        smem.tmx.ptr[i1 * I(8) + wave] = _bi(mw1)
                el, tsl, tmxl = _pass1_last(pend1[0], ms, tx)
                _store_tile(buf, el, I(TILES - 1), tx)
                swl = _wsum(tsl)
                mwl = _wmax(tmxl)
                if is_l0:
                    smem.wp1.ptr[(TILES - 1) * 8 + wave] = _bi(swl)
                    smem.tmx.ptr[(TILES - 1) * 8 + wave] = _bi(mwl)
                for b_ in range_constexpr(-(-(NBLK + 4) // NT)):
                    if tx + b_ * NT < NBLK + 4:
                        smem.bmax.ptr[tx + b_ * NT] = I(0)
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                se = F(0.0)
                for t_ in range_constexpr(TILES):
                    se = se + _wtree8([_bf(I(smem.wp1.ptr[t_ * 8 + w_])) for w_ in range_constexpr(8)])
                se = _uni(se)
                tmaxl = F(-INF)
                if lane < TILES:
                    for w_ in range_constexpr(8):
                        tmaxl = _fmax(tmaxl, _bf(I(smem.tmx.ptr[lane * 8 + w_])))
                max_logit = _uni(_wmax(tmaxl))
                idx = (p * F(200.0)).to(I)
                idx = _uni((idx < I(0)).select(I(0), (idx > I(199)).select(I(199), idx)))
                sig = I(buffer_ops.buffer_load(tab, idx, vec_width=1, dtype=I)).bitcast(F)
                sig = _uni(sig + _launder(fx.math.absf(sig) * F(-0.25)))
                oprob = _uni(_exp(fx.math.fma(std, sig, avg) - ms) / se)
                hi0 = _uni(_exp(max_logit - ms) / se)
                e_min, fast = _e_threshold(se, oprob)
                if const_expr(path == "exact"):
                    fast = F(0.0) > F(1.0)
                e_min = _uni(e_min)

                # ---- pass 2: the outliers, compacted in place (e, or Triton's probabilities off the fast path)
                run2 = I(0)
                acc2 = F(0.0)
                if fast:
                    pend2 = [_load_raw(buf, I(t_), tx) for t_ in range_constexpr(PF)]
                    for t2 in range(TILES):
                        i2 = I(t2)
                        et2 = pend2[0]
                        pend2 = pend2[1:] + [_load_raw(buf, i2 + I(PF), tx)]
                        om2, a2, bef2, tot2 = _outliers_fast(et2, e_min)
                        acc2 = acc2 + a2
                        par2 = (i2 % I(2)) * I(32)
                        if is_l0:
                            _write_counts(smem.cnt, par2, wave, tot2)
                        fx.gpu.barrier()
                        base2, own2, run2n = _bases(smem.cnt, par2, run2, lane, wave)
                        runs2 = _compact(buf, et2, om2, bef2, base2)
                        if is_l0:
                            _block_maxima(bmax_base, runs2, base2, own2)
                        run2 = run2n
                else:
                    pend3 = [_load_tile(xr, I(t_), tx, -INF) for t_ in range_constexpr(PF)]
                    for t3 in range(TILES):
                        i3 = I(t3)
                        xt3 = pend3[0]
                        pend3 = pend3[1:] + [_load_tile(xr, i3 + I(PF), tx, -INF)]
                        pr3, om3, ts3, bef3, tot3 = _outliers_exact(xt3, ms, se, oprob, i3, tx)
                        sw3 = _wsum(ts3)
                        par3 = (i3 % I(2)) * I(32)
                        if is_l0:
                            smem.wp2.ptr[i3 * I(8) + wave] = _bi(sw3)
                            _write_counts(smem.cnt, par3, wave, tot3)
                        fx.gpu.barrier()
                        base3, own3, run3n = _bases(smem.cnt, par3, run2, lane, wave)
                        runs3 = _compact(buf, pr3, om3, bef3, base3)
                        if is_l0:
                            _block_maxima(bmax_base, runs3, base3, own3)
                        run2 = run3n
                nout = _uni(run2)
                accw = _wsum(acc2)
                if is_l0:
                    smem.red.ptr[40 + wave] = _bi(accw)
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                # zeros from the last outlier to the end of its chunk: the search loads whole chunks
                for z_ in range_constexpr(CHUNK // NT):
                    zpos = nout + tx + I(z_ * NT)
                    buffer_ops.buffer_store(I(0), buf, (zpos < ((nout + I(CHUNK - 1)) // I(CHUNK)) * I(CHUNK)).select(
                        zpos, I(V)))
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                # the outlier mass: Triton's (exact) off the fast path; on it, sum e / se decides outside the margin
                so = F(0.0)
                for t_ in range_constexpr(TILES):
                    so = so + _wtree8([_bf(I(smem.wp2.ptr[t_ * 8 + w_])) for w_ in range_constexpr(8)])
                se_out = F(0.0)
                for w_ in range_constexpr(8):
                    se_out = se_out + _bf(I(smem.red.ptr[40 + w_]))
                so_fast = se_out / se
                sure = fx.math.absf(so_fast - p) > F(MARGIN) * _fmax(so_fast, p)
                if const_expr(path == "mass"):
                    sure = F(0.0) > F(1.0)
                b1 = _uni(fast.select((so_fast > p).select(I(1), I(0)), (so > p).select(I(1), I(0))))
                if fast & ~sure:
                    for t6 in range(TILES):
                        i6 = I(t6)
                        pr6, om6, ts6, bef6, tot6 = _outliers_exact(_load_tile(xr, i6, tx, -INF), ms, se, oprob, i6,
                                                                    tx)
                        sw6 = _wsum(ts6)
                        if is_l0:
                            smem.wp2.ptr[i6 * I(8) + wave] = _bi(sw6)
                    fx.gpu.barrier()
                    so6 = F(0.0)
                    for t_ in range_constexpr(TILES):
                        so6 = so6 + _wtree8([_bf(I(smem.wp2.ptr[t_ * 8 + w_])) for w_ in range_constexpr(8)])
                    b1 = _uni((so6 > p).select(I(1), I(0)))

                # ---- the pivot search
                def search(b1, lo, div, smem=smem, buf=buf, lane=lane, wave=wave, is_l0=is_l0, nout=nout, hi0=hi0,
                           p=p, se=se, ps_base=ps_base, pm_base=pm_base, pc_base=pc_base):
                    """(ppiv, mlp, nml, psum) of the search over the compacted outliers (b1) or all probabilities;
                    ``div``: the buffer holds e. Lane q holds the maximum probability of this wave's group of chunk
                    q; with b1 the K groups of largest maximum stay in registers and a round loads only the active
                    groups past them."""
                    if const_expr(b1):
                        nq = (nout + I(CHUNK - 1)) // I(CHUNK)
                        gml = _fmax(_bf(I(smem.bmax.ptr[lane * 16 + wave])),
                                    _bf(I(smem.bmax.ptr[lane * 16 + 8 + wave])))
                    else:
                        nq = I(TILES)
                        gml = _bf(I(smem.bmax.ptr[lane * 8 + wave]))
                    gml = (lane < nq).select(div.select(gml / se, gml), F(0.0))
                    slot_q, slot_g, slot_v = [], [], []
                    inslot = I(0)
                    if const_expr(b1):
                        avail = gml
                        for k_ in range_constexpr(K):
                            gk = _wmax(avail)
                            qk = I(fx.math.cttz(_ballot((avail == gk) & (gk > F(0.0))) | I64(-(1 << 63))).to(I))
                            avail = (lane == qk).select(F(0.0), avail)
                            inslot = inslot | (lane == qk).select(I(1), I(0))
                            slot_q.append(qk)
                            slot_g.append(gk)
                        slot_v = [_probs(_load_group(buf, slot_q[k_], wave, lane, True), se, div)
                                  for k_ in range_constexpr(K)]
                    hi = hi0
                    done = I(0)
                    conv = I64(0)
                    ppiv = F(1.0)
                    mlp = F(1.0)
                    nml = I(0)
                    psum = F(0.0)
                    for it in range(ITERS):
                        if done == I(0):
                            sr_it = I(it)
                            sr_piv = _uni((hi - lo) * F(0.5) + lo)
                            sr_par = (sr_it % I(2)) * I(PART)
                            sr_gact = (lane < nq) & (gml > sr_piv)
                            if (lane < nq) & ~sr_gact:
                                _write_part(smem, sr_par + lane * I(8) + wave, F(0.0), F(1.0), I(0))
                            for kq_ in range_constexpr(K if b1 else 0):
                                if slot_g[kq_] > sr_piv:
                                    sr_sw, sr_mw, sr_c = _group_eval(slot_v[kq_], sr_piv, b1)
                                    if is_l0:
                                        _write_part(smem, sr_par + slot_q[kq_] * I(8) + wave, sr_sw, sr_mw, sr_c)
                            sr_rem = _ballot(sr_gact & (inslot == I(0)))
                            while sr_rem != I64(0):
                                sr_qs, sr_rem = _take(sr_rem)
                                sr_raw = [_load_group(buf, q_, wave, lane, b1) for q_ in sr_qs]
                                for kb_ in range_constexpr(BATCH):
                                    if sr_qs[kb_] < I(64):
                                        # e -> probabilities once per group, written back (this wave's own group)
                                        sr_pv = sr_raw[kb_]
                                        if div & ~_bit(conv, sr_qs[kb_]):
                                            sr_pv = [v / se for v in sr_raw[kb_]]
                                            _store_group(buf, sr_qs[kb_], wave, lane, sr_pv, b1)
                                        conv = conv | (I64(1) << I64(sr_qs[kb_]))
                                        sr_sw2, sr_mw2, sr_c2 = _group_eval(sr_pv, sr_piv, b1)
                                        if is_l0:
                                            _write_part(smem, sr_par + sr_qs[kb_] * I(8) + wave, sr_sw2, sr_mw2,
                                                        sr_c2)
                            fx.gpu.barrier()
                            sr_o = (sr_par + lane * I(8)) * I(4)
                            sr_stb = (sr_it % I(2)) * I(16)
                            # wave 0: the chunk statistics, the chain over the active chunks and the decision
                            if wave == I(0):
                                sr_sq, sr_mq, sr_nq, sr_near = _chunk_stats(ps_base, pm_base, pc_base, sr_o, lane,
                                                                            nq)
                                sr_s0, sr_m0, sr_n0 = _chain_start()
                                sr_act = _ballot(sr_nq > I(0))
                                while sr_act != I64(0):
                                    sr_q4 = I(fx.math.cttz(sr_act).to(I))
                                    sr_act = sr_act & (sr_act - I64(1))
                                    sr_s0, sr_m0, sr_n0 = _merge(sr_s0, sr_m0, sr_n0, _lane_val(sr_sq, sr_q4),
                                                                 _lane_val(sr_mq, sr_q4), _lane_val(sr_nq, sr_q4))
                                _decide(smem.st, sr_stb, sr_s0, sr_m0, sr_n0, sr_piv, p, lo, hi, ppiv, mlp, nml, psum,
                                        sr_it)
                                _publish_near(smem, sr_stb, lane, sr_near, sr_mq)
                            fx.gpu.barrier()
                            sr_flags = _read_flags(smem.st, sr_stb)
                            if sr_flags != I64(0):
                                # a wave minimum within 1e-9 of its chunk minimum but not equal: recount those chunks
                                # against the chunk minimum (rare)
                                sr_recb = (sr_it % I(2)) * I(NCH * 8)
                                sr_rem3 = sr_flags
                                while sr_rem3 != I64(0):
                                    sr_q3 = I(fx.math.cttz(sr_rem3).to(I))
                                    sr_rem3 = sr_rem3 & (sr_rem3 - I64(1))
                                    sr_c3 = _group_count(
                                        _probs(_load_group(buf, sr_q3, wave, lane, b1), se,
                                               div & ~_bit(conv, sr_q3)), sr_piv, _bf(I(smem.cmin.ptr[sr_q3])))
                                    if is_l0:
                                        smem.rec.ptr[sr_recb + sr_q3 * I(8) + wave] = sr_c3
                                fx.gpu.barrier()
                                if wave == I(0):
                                    sr_sq5, sr_mq5, sr_nq5, sr_near5 = _chunk_stats(ps_base, pm_base, pc_base, sr_o,
                                                                                    lane, nq)
                                    sr_nq5 = (sr_near5 != I(0)).select(_sum_rec(smem.rec, sr_recb + lane * I(8)),
                                                                       sr_nq5)
                                    sr_s5, sr_m5, sr_n5 = _chain_start()
                                    sr_act5 = _ballot(sr_nq5 > I(0))
                                    while sr_act5 != I64(0):
                                        sr_q5 = I(fx.math.cttz(sr_act5).to(I))
                                        sr_act5 = sr_act5 & (sr_act5 - I64(1))
                                        sr_s5, sr_m5, sr_n5 = _merge(sr_s5, sr_m5, sr_n5, _lane_val(sr_sq5, sr_q5),
                                                                     _lane_val(sr_mq5, sr_q5), _lane_val(sr_nq5, sr_q5))
                                    _decide(smem.st, sr_stb, sr_s5, sr_m5, sr_n5, sr_piv, p, lo, hi, ppiv, mlp, nml,
                                            psum, sr_it)
                                fx.gpu.barrier()
                            lo, hi, done, ppiv, mlp, nml, psum = _read_state(smem.st, sr_stb)
                    return ppiv, mlp, nml, psum

                ppiv = F(1.0)
                mlp = F(1.0)
                nml = I(0)
                psum = F(0.0)
                if b1 != I(0):
                    ppiv, mlp, nml, psum = search(True, oprob, fast)
                else:
                    # Triton's probabilities of every column in column order and the (tile, wave) maxima
                    mn7 = F(INF)
                    for t7 in range(TILES):
                        i7 = I(t7)
                        pr7, gm7, tmn7 = _probs_tile(_load_tile(xr, i7, tx, -INF), ms, se, i7, tx)
                        _store_tile(buf, pr7, i7, tx)
                        gw7 = _wmax(gm7)
                        if is_l0:
                            smem.bmax.ptr[i7 * I(8) + wave] = _bi(gw7)
                        mn7 = _fmin(mn7, tmn7)
                    mnw7 = _wmin(mn7)
                    if is_l0:
                        smem.red.ptr[32 + wave] = _bi(mnw7)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    min_logit = F(INF)
                    for w_ in range_constexpr(8):
                        min_logit = _fmin(min_logit, _bf(I(smem.red.ptr[32 + w_])))
                    min_logit = _uni(_fmin(min_logit, max_logit))
                    ppiv, mlp, nml, psum = search(False, _uni(_exp(min_logit - ms) / se), F(0.0) > F(1.0))

                # ---- the mask (Triton's sixth pass): keep x > fpiv; with ties the boundary duplicates past nkeep
                # in column order go too. A tile whose maximum is neither kept nor a duplicate becomes -inf unread.
                dup = _uni(_log_any(mlp * se) + ms)
                nkeep = nml - ((psum - p) / mlp).to(fx.Uint32).to(I)
                fpiv = _uni(_log_any(ppiv * se) + ms)
                if (fpiv < max_logit) & (fpiv != F(-INF)):
                    ties = _ult(nkeep, nml)
                    clear = [F(-INF) for _ in range_constexpr(16)]
                    kept8 = I(0)
                    for t8 in range(TILES):
                        i8 = I(t8)
                        tm8 = _lane_val(tmaxl, i8)
                        far8 = ~((fx.math.absf(tm8 - dup) < F(EPS)) | (tm8 > dup))
                        if (tm8 > fpiv) | (ties & ~far8):
                            xt8 = _load_raw(xr, i8, tx)
                            keep8 = _keep(xt8, fpiv)
                            if ties:
                                dm8, bef8, tot8 = _dup_flags(xt8, dup, i8, tx)
                                par8 = (i8 % I(2)) * I(32)
                                if is_l0:
                                    _write_counts(smem.cnt, par8, wave, tot8)
                                fx.gpu.barrier()
                                base8, _own8, kept8n = _bases(smem.cnt, par8, kept8, lane, wave)
                                keep8 = _keep_ties(keep8, dm8, bef8, base8, nkeep)
                                kept8 = kept8n
                            _store_tile(xr, _masked(xt8, keep8), i8, tx)
                        else:
                            _store_tile(xr, clear, i8, tx)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()

    @flyc.jit
    def launch_top_p_mask_rows(arg_x: fx.Int64, i32_sx: fx.Int32, arg_p: fx.Int64, arg_tab: fx.Int64,
                               arg_buf: fx.Int64, i32_n: fx.Int32, i32_grid: fx.Int32,
                               stream: fx.Stream):
        kernel(arg_x, i32_sx, arg_p, arg_tab, arg_buf, i32_n, i32_grid,
               value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": f"{NT},{NT}"}).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch_top_p_mask_rows


@launcher
def compile_sqrt_f32():
    """Elementwise v_sqrt_f32 (the reference's ``sqrt_f32``)."""

    @flyc.kernel(name="v_sqrt_f32", known_block_size=[256, 1, 1])
    def kernel(arg_x: fx.Int64, arg_y: fx.Int64, i32_n: fx.Int32):
        i = I(fx.block_idx.x) * I(256) + I(fx.thread_idx.x)
        xr = buffer_ops.create_buffer_resource_from_addr(arg_x)
        yr = buffer_ops.create_buffer_resource_from_addr(arg_y)
        if i < i32_n:
            v = I(buffer_ops.buffer_load(xr, i, vec_width=1, dtype=I)).bitcast(F)
            buffer_ops.buffer_store(_bi(_sqrt(v)), yr, i)

    @flyc.jit
    def launch_sqrt_f32(arg_x: fx.Int64, arg_y: fx.Int64, i32_n: fx.Int32, i32_grid: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_y, i32_n).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch_sqrt_f32
