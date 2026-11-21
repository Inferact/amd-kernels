# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The AttnRes stages the layer tail runs after the all-reduce: prefix + delta, the source weights, the softmax mix and
the post-attention RMSNorm, on the 56 attn CTAs (shared ``stage_slice`` / ``stage_gates`` / ``stage_mix``)."""

import math

import flydsl.expr as fx
from amd_kernels.common import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from amd_kernels.common.mono.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rsq,
    ld_i32,
    rsrc,
    traced,
    uniform,
    wave_max,
)
from amd_kernels.common.mono.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    source_digest,
)
from amd_kernels.common.mono.sync import (
    publish,
)

HIDDEN = 7168
NA = 56  # attn CTAs
COLS = HIDDEN // NA  # 128 columns a slice
QC = COLS // 4  # threads a (token, source) row: 4 columns each
GSL = 7  # slices a gate / mix-norm thread sums (NA = 8 x 7)
HD = 128
NH = 12  # local heads at TP8
PROJ = NH * HD  # 1536
QKV = 3 * PROJ
G2_0 = QKV
FA_0 = G2_0 + PROJ  # 6144
BETA_0 = FA_0 + HD  # 6272
NPROJ = 6288  # in_proj rows a rank holds (6284 + 4 pad)
ROWS = 16
GEMV_TASKS = NPROJ // ROWS  # 393
GEMV_CTAS = (GEMV_TASKS + 1) // 2  # 197
KCH = 128  # K a chunk: 4 MFMAs of 32
NKCH = HIDDEN // KCH  # 56
PER_WAVE = NKCH // WAVES  # 7
VCH = 16  # state rows a KDA task
NVCH = HD // VCH  # 8
CONV_W = 4
LOG2E = 1.0 / math.log(2.0)
LDS_PAD = 8  # bf16 a padded x row
MAX_S = 64  # rows a launch
XT = 8  # x rows a gemv tile past 8 rows (16 padded 7168-wide rows exceed LDS)
# every source's slice in LDS up to this; past it the prefix only
VALS_BYTES = 96 * 1024
MSQ0 = 1024  # red: the mix's sums of squares (the weights below)
# timeline points: start, slice, gate, mix, load_x, gemv0, gemv1, k.stage,
# k.conv, k.gates, k.recur, k.norm
TL_POINTS = 15

assert NA + GEMV_CTAS <= BLOCKS and NA == 8 * GSL

_STREAM = fx.Stream(None)
REGIONS = ("part", "wgt", "msq", "xrdy", "proj", "qk", "onrm")
TAG_BATCH = 14  # tagged-row specs a poll batch (56 registers; ``poll_rows_to_lds``)
# every mono source file: in every build's JIT cache key
_SOURCES = source_digest(".")


TAG_SLOTS = 256  # a step's mailbox tags: epoch x 256 + a launch's slot


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (a device counter the runner
    bumps once a step, in the graph) and the launch's slot, so no step's or
    other launch's pair ever passes for this one's -- nothing is zeroed."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


def vals_all(s, ns, extra=0):
    """Whether every (token, source)'s slice stays in LDS ``vals`` (else the
    prefix only; ``stage_mix`` re-reads the blocks), ``extra`` bytes beside it."""
    return s * ns * COLS * 4 + extra <= VALS_BYTES


def vals_words(s, ns, extra=0):
    return s * (ns if vals_all(s, ns, extra) else 1) * COLS


def ld_bf(ptr, i, cm=0):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16, cache_modifier=cm)
        )
    )


def ld_bf4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as four f32."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def bf4_words(v):
    """Four f32 -> two words of packed bf16."""
    return fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def bf16r(v):
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def f32_of(w):
    return w.bitcast(fx.Float32)


def sum_polled(vals):
    acc = f32_of(vals[0][0])
    for k in range(1, len(vals)):
        acc = acc + f32_of(vals[k][0])
    return acc


# ---------------------------------------------------------------- attn CTAs
@traced
def stage_slice(c, task):
    """Columns 128 task ..: every token's sources into LDS ``vals`` (4 columns a
    thread), the prefix (+ delta) rounded to bf16 and stored back, per-(token,
    source) partial sum of squares and qk dot -> PART."""
    s, ns, nb, tid, lane = c["S"], c["NS"], c["NB"], c["tid"], c["lane"]
    a = c["args"]
    c0 = task * COLS
    n = s * ns * QC  # threads' work items
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, n - 1)
        live = (tid + THREADS * i) < n
        row = e // QC  # (token, source)
        t = row // ns
        src = row % ns
        col = c0 + (e % QC) * 4
        is_pre = src == nb
        blk = ld_bf4(
            a["blocks"], t * a["blk_sm"] + fx.min(src, nb - 1) * a["blk_sr"] + col
        )
        pre = ld_bf4(a["prefix"], t * HIDDEN + col)
        if const_expr(c["delta"]):
            if const_expr(c.get("delta_lds") is not None):
                # the delta reduced into LDS first (the all-reduced o_proj)
                dl = [
                    fx.ptr_load(c["delta_lds"] + (t * COLS + (e % QC) * 4 + j))
                    for j in range(4)
                ]
            else:
                dl = ld_bf4(a["delta"], t * HIDDEN + col)
            if const_expr(c.get("reset", False)):
                # a block-write layer's MLP seam: the prefix restarts at the
                # attention's output (the input prefix is not read)
                pre = dl
            else:
                pre = [bf16r(pre[j] + dl[j]) for j in range(4)]
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre), rsrc(a["prefix"]), (t * HIDDEN + col) // 2
                )
        if const_expr(c.get("write_idx", -1) >= 0):  # noqa: SIM102 (traced: `and` does not combine traced values)
            # the block-write layer's attention seam: the updated prefix is
            # block write_idx
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre),
                    rsrc(a["blocks"]),
                    (t * a["blk_sm"] + c["write_idx"] * a["blk_sr"] + col) // 2,
                )
        v = [
            live.select(is_pre.select(pre[j], blk[j]), fx.Float32(0.0))
            for j in range(4)
        ]
        w = ld_bf4(a["ares_nw"], col)
        q = ld_bf4(a["ares_qk"], col)
        # a thread past the end holds a clamped copy of the last element: it
        # must not store (it raced the owner with zeros)
        if const_expr(c["vals_all"]):
            if live:
                for j in range_constexpr(4):
                    fx.ptr_store(v[j], c["vals"] + (row * COLS + (e % QC) * 4 + j))
        else:
            if live & is_pre:
                for j in range_constexpr(4):
                    fx.ptr_store(v[j], c["vals"] + (t * COLS + (e % QC) * 4 + j))
        sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        dt = (
            v[0] * (w[0] * q[0])
            + v[1] * (w[1] * q[1])
            + v[2] * (w[2] * q[2])
            + v[3] * (w[3] * q[3])
        )
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        dt = butterfly(dt, (16, 8, 4, 2, 1))
        if live & (lane % 32 == 0):
            at = ((task * s + t) * ns + src) * 2
            c["put"](c["part"], at, sq)
            c["put"](c["part"], at + 1, dt)
    gpu.barrier()


@traced
def stage_gate(c, t):
    """Token t: the 56 slices' partials -> source weights e_s and their sum."""
    s, ns, tid, lane, wave, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    nq = ns * 2
    if tid < 8 * nq:
        g = tid // nq
        q = tid % nq
        vals = c["poll"](
            [(c["part"], (((g * GSL + k) * s + t) * ns) * 2 + q, 1) for k in range(GSL)]
        )
        fx.ptr_store(sum_polled(vals), red + (g * nq + q))
    gpu.barrier()
    if wave == 0:
        src = fx.min(lane, ns - 1)
        sq = fx.ptr_load(red + src * 2)
        dt = fx.ptr_load(red + src * 2 + 1)
        for g in range_constexpr(1, 8):
            sq = sq + fx.ptr_load(red + (g * nq + src * 2))
            dt = dt + fx.ptr_load(red + (g * nq + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            c["put"](c["wgt"], t * (ns + 1) + lane, ex)
        if lane == 0:
            c["put"](c["wgt"], t * (ns + 1) + ns, den)
    gpu.barrier()


def mix_src(c, t, src, cc, c0):
    """Token t's source ``src``, slice columns cc .. cc + 4: from ``vals``, or
    (prefix only in LDS) a block re-read from ``blocks`` -- the same bf16 values
    ``stage_slice`` read."""
    if const_expr(c["vals_all"]):
        return [
            fx.ptr_load(c["vals"] + ((t * c["NS"] + src) * COLS + cc + j))
            for j in range(4)
        ]
    if const_expr(src == c["NB"]):
        return [fx.ptr_load(c["vals"] + (t * COLS + cc + j)) for j in range(4)]
    a = c["args"]
    col = c0 + cc
    return ld_bf4(a["blocks"], t * a["blk_sm"] + src * a["blk_sr"] + col)


@traced
def stage_mix(c, task):
    """Columns 128 task ..: the softmax mix of the sources (4 columns a thread,
    16 tokens a round), its sum of squares (MSQ), every slice's -> rstd, then
    bf16 x = mixed rstd w -> the plain x rows, and the slice's XRDY flag."""
    s, ns, tid, lane, red = c["S"], c["NS"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    c0 = task * COLS
    nw = s * (ns + 1)
    assert nw <= MSQ0 and 8 * s <= THREADS
    # read here, not in a traced branch (a method call on ``c`` there makes
    # FlyDSL carry ``c`` as branch state)
    tag_x = c.get("tag_x", False)
    for i in range_constexpr((nw + THREADS - 1) // THREADS):
        w = tid + THREADS * i
        if w < nw:
            v = c["poll"]([(c["wgt"], w, 1)])[0][0]
            fx.ptr_store(f32_of(v), red + w)
    gpu.barrier()
    tpr = THREADS // QC  # tokens a round
    cc = (tid % QC) * 4
    rounds = []
    for rr in range_constexpr((s + tpr - 1) // tpr):
        tr = rr * tpr + tid // QC
        mine = tr < s
        t = fx.min(tr, s - 1)
        mixed = [fx.Float32(0.0) for _ in range(4)]
        for src in range_constexpr(ns):
            wt = fx.ptr_load(red + (t * (ns + 1) + src))
            v = mix_src(c, t, src, cc, c0)
            mixed = [mixed[j] + wt * v[j] for j in range(4)]
        den = fx.ptr_load(red + (t * (ns + 1) + ns))
        mixed = [mixed[j] / den for j in range(4)]
        sq = (
            mixed[0] * mixed[0]
            + mixed[1] * mixed[1]
            + mixed[2] * mixed[2]
            + mixed[3] * mixed[3]
        )
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        if mine & (lane % 32 == 0):
            c["put"](c["msq"], task * s + t, sq)
        rounds.append((mine, t, mixed))
    gpu.barrier()
    if tid < 8 * s:
        g = tid // s
        tt = tid % s
        vals = c["poll"]([(c["msq"], (g * GSL + k) * s + tt, 1) for k in range(GSL)])
        fx.ptr_store(sum_polled(vals), red + (MSQ0 + g * s + tt))
    gpu.barrier()
    wn = ld_bf4(a["in_nw"], c0 + cc)
    for mine, t, mixed in rounds:
        tot = fx.ptr_load(red + (MSQ0 + t))
        for g in range_constexpr(1, 8):
            tot = tot + fx.ptr_load(red + (MSQ0 + g * s + t))
        r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["out_eps"]))
        xw = bf4_words([mixed[j] * r * wn[j] for j in range(4)])
        if mine:
            at = (t * HIDDEN + c0 + cc) // 2
            if const_expr(tag_x):
                # (``kernel.OPT_TAG_X``) the rows also as tagged pairs,
                # which the router and latent GEMVs poll (no XRDY wait)
                c["put_words"](c["xt"], at, [xw[0], xw[1]])
            bo.buffer_store(xw, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
    publish(c["put"], c["xrdy"], task, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def poll_rows_to_lds(c, region, k, n, xl, xrow, ldk, k0, t0):
    """``gemv.rows_to_lds`` from tagged pairs: columns k0 .. k0 + k of rows
    t0 .. t0 + n of the [S, ldk] bf16 rows a producer put as (word, tag) pairs
    into ``region``, polled (two pairs a spec, every thread's specs one batch)
    -> LDS rows 0 .. n - 1."""
    tid = c["tid"]
    quads = n * k // 4  # 4 bf16 = 2 pairs a spec
    per = (quads + THREADS - 1) // THREADS
    # a batch at most TAG_BATCH specs (4 registers each), stored before the next
    for b0 in range_constexpr(0, per, TAG_BATCH):
        ii = list(range(b0, min(per, b0 + TAG_BATCH)))
        es = [fx.min(tid + THREADS * i, quads - 1) for i in ii]
        tk = [(e // (k // 4), e % (k // 4) * 4) for e in es]
        got = c["poll"](
            [(region, ((t0 + t) * ldk + k0 + kk) // 2, 2) for t, kk in tk],
            batch=len(ii),
        )
        for j in range_constexpr(len(ii)):
            t, kk = tk[j]
            if tid + THREADS * ii[j] < quads:
                fx.ptr_store(
                    fx.Vector.from_elements(got[j], fx.Int32), xl + (t * xrow + kk) // 2
                )
    gpu.barrier()


@traced
def stage_gates(c, bid):
    """The gates of tokens bid, bid + 56, .. (< S) on attn CTA ``bid``."""
    s = c["S"]
    for rr in range_constexpr((s + NA - 1) // NA):
        if bid + NA * rr < s:
            stage_gate(c, bid + NA * rr)
