# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""K1: one Kimi-K3 KDA layer from the AttnRes seam to the gated core output, for a spec-verify step (R requests x L
tokens, S = R L rows), in one launch of ``BLOCKS`` CTAs x ``THREADS`` threads (all resident).

    token CTAs 0 .. S-1   AttnRes of one row (prefix + delta -> bf16, softmax weights, mix, RMSNorm) -> the bf16 x
                          row (plain device-scope stores) and its XRDY flag
    the other CTAs        26 rows of in_proj each (tile A: 16 rows, tile B: up to 10), issued at kernel start into
                          registers: the stream is HBM-bound, so every wave but one issues them at once. The
                          stager waves (``NSTAGER``) issue the rest of theirs after they have polled the XRDY flags and DMA'd x
                          into LDS (``stage_x``): a poll or DMA queued behind a wave's weight loads lands with them.
                          The MFMAs of a wave wait for its own chunks only; the sums over the waves' K slices -> PROJ
    KDA tasks             (request, head, 16 state rows), NVCH consecutive CTAs a head, in rounds when R x 12 x 8
                          exceeds the CTAs left: one poll batch for everything read from PROJ, conv of the head's q / k
                          channels (redundant in the head's chunks) and the chunk's v channels, f_b MFMA + gate, l2norm,
                          the recurrence (a state a token), partial norms (ONRM), gated RMSNorm -> core_out

x is read through LDS in MFMA-block order (``xblock_col``); what does not fit LDS (S > 10 rows) is loaded straight
from the x rows (``x_direct``). A chunk's conv-state stores wait for its head's ONRM pairs, which every chunk puts after
reading its window: the chunks read each other's q / k channels.

Rounding follows the stock vLLM path where it is elementwise (the AttnRes prefix add, every bf16 hand-off); sums over the
hidden dimension and K differ only in their order.
"""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from amd_kernels.common.mono.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from amd_kernels.common.mono.ops import (
    CM_DEV,
    CM_NT,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rsq,
    ld_i32,
    sld_i32,
    lds_dma,
    dma_wait,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_max,
    wave_sum,
    sum32_dpp,
)
from amd_kernels.common.mono.abi import FAILED  # noqa: F401
from amd_kernels.common.mono.plan import (
    BLOCKS,
    HEALTH_WORDS,  # noqa: F401
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
    source_digest,
)
from amd_kernels.common.mono.sync import (
    Mailbox,
    publish,
    sreg,
)

HIDDEN = 7168
HD = 128
NH = 12  # local heads at TP8
PROJ = NH * HD  # 1536
QKV = 3 * PROJ
G2_0 = QKV
FA_0 = G2_0 + PROJ  # 6144
BETA_0 = FA_0 + HD  # 6272
NPROJ = 6288  # in_proj rows a rank holds (6284 + 4 pad)
ROWS = 16
KCH = 128  # K a chunk: 4 MFMAs of 32
NKCH = HIDDEN // KCH  # 56
PER_WAVE = NKCH // WAVES  # 7
NSTAGER = 4  # waves that stage x
EARLY = 5  # a stager wave's K chunks in flight with everyone's
VCH = 16  # state rows a KDA task
NVCH = HD // VCH  # 8
CONV_W = 4
LOG2E = 1.0 / math.log(2.0)
LDS_PAD = 8  # bf16 a padded x row
MAX_S = 16  # rows a launch
XL_BYTES = 163_200 - 192  # LDS of the x rows: all of it but the timeline stamps
XS_MAX = 12  # rows x stays resident in LDS for
WGT0 = 256  # red: the source weights
MSQ0 = 512  # red: the mix's per-wave sums of squares
# timeline points: start, slice, gate, mix, load_x, gemv0, gemv1, k.stage,
# k.conv, k.gates, k.recur, k.norm
TL_POINTS = 32


_STREAM = fx.Stream(None)
REGIONS = ("xrdy", "proj", "onrm")
# every mono source file: in every build's JIT cache key
_SOURCES = source_digest("common/mono", "attention/kda_mono_decode_k1_h12_d128_ranks8")


@dataclass(frozen=True)
class KdaPreBuild:
    tokens: int  # S
    qlen: int  # L, tokens a request
    nblocks: int  # AttnRes block sources (the prefix is one more)
    delta: bool
    state_len: int  # conv state entries: CONV_W - 1 + num_spec
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    onorm_eps: float = 1e-5
    lower_bound: float = -5.0
    timeline: bool = False
    write_idx: int = -1  # a block-write layer's block: the updated prefix stored there


ABI = KernelAbi(
    (
        "prefix",
        "delta",
        "blocks",
        "blk_sm",
        "blk_sr",
        "ares_nw",
        "ares_qk",
        "in_nw",
        "w_in",
        "w_fb",
        "conv_w",
        "conv_st",
        "cs_seq",
        "cs_dim",
        "cs_tok",
        "a_log",
        "dt_bias",
        "on_w",
        "rstate",
        "rs_seq",
        "st_idx",
        "st_stride",
        "num_acc",
        "core_out",
        "scratch",
        "layer",
        "epoch",
        "health",
        "alarm",
        "tl",
    )
)


def scratch_layout(key: KdaPreBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    assert s % key.qlen == 0
    pairs = {
        "xrdy": s,
        "proj": s * NPROJ,
        "onrm": s * NH * NVCH,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    # the plain bf16 x rows (not pairs), 16 B aligned
    lay["x"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: KdaPreBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


TAG_SLOTS = 256  # a step's mailbox tags: epoch x 256 + a launch's slot


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (a device counter the runner
    bumps once a step, in the graph) and the launch's slot, so no step's or
    other launch's pair ever passes for this one's -- nothing is zeroed."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


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


def sum4(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def sum_polled(vals):
    acc = f32_of(vals[0][0])
    for k in range(1, len(vals)):
        acc = acc + f32_of(vals[k][0])
    return acc


def token_tiles(s, n=16):
    """(first token, rows) of each tile of ``n`` rows over S = ``s``."""
    return [(t0, min(n, s - t0)) for t0 in range(0, s, n)]


# ---------------------------------------------------------------- attn CTAs
def ld16(ptr, i, cm=0):
    """bf16 elements i .. i + 8 (i a multiple of 8) as four words."""
    return fx.Vector(
        bo.buffer_load(
            rsrc(ptr), i // 2, vec_width=4, dtype=T.i32, cache_modifier=cm
        )
    )


def unpack8(w):
    return [f for j in range(4) for f in (bf_lo(w[j]), bf_hi(w[j]))]


def pack8(v):
    return fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


@traced
def stage_attn(c, t):
    """Token t, whole: the prefix (+ delta) rounded to bf16 and stored back, the
    softmax weights of the sources, the mix, its RMSNorm -> the bf16 x row, and
    the token's XRDY flag."""
    nb, ns, tid, lane, wave, red = (
        c["NB"], c["NS"], c["tid"], c["lane"], c["wave"], c["red"]
    )  # fmt: skip
    a = c["args"]
    chunks = HIDDEN // 8
    nj = (chunks + THREADS - 1) // THREADS
    live = [(tid + THREADS * j) < chunks for j in range(nj)]
    col = [fx.min(tid + THREADS * j, chunks - 1) * 8 for j in range(nj)]
    blk = [
        [ld16(a["blocks"], t * a["blk_sm"] + s * a["blk_sr"] + col[j]) for s in range(nb)]
        for j in range(nj)
    ]
    pre = [ld16(a["prefix"], t * HIDDEN + col[j]) for j in range(nj)]
    dl = [ld16(a["delta"], t * HIDDEN + col[j]) for j in range(nj)] if c["delta"] else None
    wq = []
    for j in range_constexpr(nj):
        w = unpack8(ld16(a["ares_nw"], col[j]))
        q = unpack8(ld16(a["ares_qk"], col[j]))
        wq.append([w[e] * q[e] for e in range(8)])
    wn = [unpack8(ld16(a["in_nw"], col[j])) for j in range(nj)]
    vs = []  # vs[j][s]: 8 f32 of (token t, source s)
    for j in range_constexpr(nj):
        pf = unpack8(pre[j])
        if const_expr(c["delta"]):
            df = unpack8(dl[j])
            pf = [bf16r(pf[e] + df[e]) for e in range(8)]
            if live[j]:
                bo.buffer_store(pack8(pf), rsrc(a["prefix"]), (t * HIDDEN + col[j]) // 2)
        if const_expr(c["write_idx"] >= 0):
            if live[j]:
                bo.buffer_store(
                    pack8(pf),
                    rsrc(a["blocks"]),
                    (t * a["blk_sm"] + c["write_idx"] * a["blk_sr"] + col[j]) // 2,
                )
        zero = fx.Float32(0.0)
        row = [[live[j].select(f, zero) for f in unpack8(b)] for b in blk[j]]
        row.append([live[j].select(f, zero) for f in pf])
        vs.append(row)
    stamp(c["on"], c["tls"], tid, 1)
    # per source: sum of squares and the logit's dot, every thread's chunks
    parts = []
    for s in range_constexpr(ns):
        sq = fx.Float32(0.0)
        dt = fx.Float32(0.0)
        for j in range_constexpr(nj):
            sj = fx.Float32(0.0)
            dj = fx.Float32(0.0)
            for e in range_constexpr(8):
                sj = sj + vs[j][s][e] * vs[j][s][e]
                dj = dj + vs[j][s][e] * wq[j][e]
            sq = sq + sj
            dt = dt + dj
        parts += [wave_sum(sq), wave_sum(dt)]
    if lane == 0:
        for k in range_constexpr(2 * ns):
            fx.ptr_store(parts[k], red + (wave * 2 * ns + k))
    gpu.barrier()
    if wave == 0:
        src = fx.min(lane, ns - 1)
        sq = fx.ptr_load(red + src * 2)
        dt = fx.ptr_load(red + src * 2 + 1)
        for w in range_constexpr(1, WAVES):
            sq = sq + fx.ptr_load(red + (w * 2 * ns + src * 2))
            dt = dt + fx.ptr_load(red + (w * 2 * ns + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            fx.ptr_store(ex, red + (WGT0 + lane))
        if lane == 0:
            fx.ptr_store(den, red + (WGT0 + ns))
    gpu.barrier()
    wt = [fx.ptr_load(red + (WGT0 + s)) for s in range(ns)]
    den = fx.ptr_load(red + (WGT0 + ns))
    mixed = []
    msq = fx.Float32(0.0)
    for j in range_constexpr(nj):
        mj = [fx.Float32(0.0) for _ in range(8)]
        for s in range_constexpr(ns):
            mj = [mj[e] + wt[s] * vs[j][s][e] for e in range(8)]
        mj = [mj[e] / den for e in range(8)]
        mixed.append(mj)
        sj = fx.Float32(0.0)
        for e in range_constexpr(8):
            sj = sj + mj[e] * mj[e]
        msq = msq + live[j].select(sj, fx.Float32(0.0))
    msq = wave_sum(msq)
    if lane == 0:
        fx.ptr_store(msq, red + (MSQ0 + wave))
    gpu.barrier()
    tot = fx.ptr_load(red + MSQ0)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(red + (MSQ0 + w))
    r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["out_eps"]))
    for j in range_constexpr(nj):
        xw = pack8([mixed[j][e] * r * wn[j][e] for e in range(8)])
        if live[j]:
            bo.buffer_store(
                xw, rsrc(c["xbuf"]), (t * HIDDEN + col[j]) // 2, cache_modifier=CM_DEV
            )
    stamp(c["on"], c["tls"], tid, 2)
    publish(c["put"], c["xrdy"], t, fx.Int32(1), tid == 0)


# ---------------------------------------------------------------- gemv CTAs
def proj_row(lr):
    """The in_proj row of logical row ``lr``: f_a / beta first, then q, k, v, g."""
    return lr + FA_0 - (lr >= NPROJ - FA_0).select(fx.Int32(NPROJ), fx.Int32(0))


def gemv_loads(c, lr0, nrows, chunks):
    """This wave's weight operands of logical rows ``lr0`` .. + ``nrows`` for its
    K chunks ``chunks`` (indices into wave, wave + 8, ..): a lane 64 contiguous
    bytes of its row a chunk (4 dwordx4), a flat list [4 chunk + q]. A row past
    ``nrows`` or the last loads nothing (an out-of-range buffer offset)."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_in"], NPROJ * HIDDEN * 2)
    lr = lr0 + lane % ROWS
    valid = (lane % ROWS < nrows) & (lr < NPROJ)
    row = valid.select(proj_row(lr), fx.Int32(NPROJ))
    g = lane // ROWS
    out = []
    for i in chunks:
        k = (wave + WAVES * i) * KCH + g * 8
        base = (row * HIDDEN + k) // 2  # dwords
        out += [
            fx.Vector(
                bo.buffer_load(
                    r_w, base + 16 * q, vec_width=4, dtype=T.i32, cache_modifier=0
                )
            )
            for q in range(4)
        ]
    return out


@traced
def stage_x(c, t0, n, poll=True, wait=True):
    """The stager waves' part (the others hold weights in flight, and a poll or
    DMA queued behind them would land with them): every token's x row ready (one
    lane a row polls its XRDY flag), then rows t0 .. t0 + n of x into LDS rows
    0 .. n in block order (``xblock_col``): 1 KB DMAs of 16 blocks of 32 columns
    a (row, piece), the blocks past ``c["XB"]`` stay out (``x_direct``)."""
    lane = c["lane"]
    nb = c["XB"]
    if const_expr(poll):
        if lane < c["S"]:
            c["poll"]([(c["xrdy"], lane, 1)])
        stamp(c["on"], c["tls"], c["tid"] - 64 * (WAVES - 1), 24)
    l4 = lane // 4
    col0 = (l4 % 8) * KCH + (l4 // 8) * 32 + (lane % 4) * 8
    pieces = (nb + 15) // 16
    sidx = uniform(c["wave"]) - (WAVES - NSTAGER)
    for t in range_constexpr(n):
        for p in range_constexpr(pieces):
            col = col0 + (p // 2) * (WAVES * KCH) + (p % 2) * 64
            src = c["xbuf"] + fx.Int64(((t0 + t) * HIDDEN + col) * 2)
            dst = c["xl_b"] + t * c["XROW"] + p * 1024
            mine = sidx == ((t * pieces + p) % NSTAGER)
            if const_expr(nb % 16 == 0):
                if mine:
                    lds_dma(src, dst, "sc1")
            else:
                if mine & (p * 16 + l4 < nb):
                    lds_dma(src, dst, "sc1")
    if const_expr(wait):
        stamp(c["on"], c["tls"], c["tid"] - 64 * (WAVES - 1), 25)
        vm_wait(0)
        stamp(c["on"], c["tls"], c["tid"] - 64 * (WAVES - 1), 26)


def xblock_col(blk):
    """First column of x block ``blk``: the 32 columns of MFMA step q of K chunk i
    of wave w, blk = (4 i + q) 8 + w."""
    i = blk // 32
    q = (blk // 8) % 4
    w = blk % 8
    return (w + WAVES * i) * KCH + 32 * q


@traced
def gemv_weights(c, lr_a, n_a, lr_b, n_b):
    """Every wave's weights in flight at once, except the last K chunks of the
    stager waves' (the last ``NSTAGER`` waves): they poll and stage x first
    (``stage_x``), then load those, which land while the others' MFMAs wait for
    the barrier."""
    wave = uniform(c["wave"])
    me = wave >= WAVES - NSTAGER
    zero = fx.Int32(0)
    # the stager waves' first EARLY chunks are in flight with everyone's; their
    # poll queues behind those only
    early = range(EARLY)
    rest = range(EARLY, PER_WAVE)
    ops_a = gemv_loads(c, lr_a, n_a, early) + gemv_loads(c, lr_a, me.select(zero, n_a), rest)
    ops_b = gemv_loads(c, lr_b, n_b, early) + gemv_loads(c, lr_b, me.select(zero, n_b), rest)
    if me:
        stage_x(c, 0, c["NT"])
        ops_a = ops_a[: 4 * EARLY] + gemv_loads(c, lr_a, n_a, rest)
        ops_b = ops_b[: 4 * EARLY] + gemv_loads(c, lr_b, n_b, rest)
    gpu.barrier()
    return ops_a, ops_b


def x_direct(c):
    """The B operands of the x blocks past the LDS ones, straight from the x
    rows: {(i, q): 4 words a lane}."""
    lane, wave = c["lane"], c["wave"]
    t = fx.min(lane % ROWS, c["S"] - 1)
    g = lane // ROWS
    out = {}
    for i in range(PER_WAVE):
        for q in range(4):
            if (4 * i + q) * 8 + 8 > c["XB"]:
                k = (wave + WAVES * i) * KCH + g * 8
                out[(i, q)] = fx.Vector(
                    bo.buffer_load(
                        rsrc(c["xbuf"]),
                        (t * HIDDEN + k + 32 * q) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
    return out


def vm_wait(n):
    llvm.inline_asm(None, [], f"s_waitcnt vmcnt({n})", "", has_side_effects=True)


def gemv_mfma(c, ops_a, ops_b, xd):
    """The weights ``ops_a`` / ``ops_b`` against the S x rows (each B operand read
    once for both) -> this wave's two 16x16 accumulators."""
    lane, wave = c["lane"], c["wave"]
    g = lane // ROWS
    t = fx.min(lane % ROWS, c["NT"] - 1)
    nb = c["XB"]
    acc_a = fx.Vector.filled(4, 0.0, fx.Float32)
    acc_b = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(PER_WAVE):
        for q in range_constexpr(4):
            base = (4 * i + q) * 8
            if const_expr(base >= nb):
                xb = xd[(i, q)]
            else:
                xb = fx.Vector(
                    fx.ptr_load(
                        c["xl"] + (t * c["XROW"] + (base + wave) * 64 + g * 16) // 4,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                if const_expr(base + 8 > nb):
                    xb = fx.Vector.from_elements(
                        [(wave >= nb - base).select(xd[(i, q)][e], xb[e]) for e in range(4)],
                        fx.Int32,
                    )
            xb = xb.bitcast(fx.BFloat16)
            acc_a = mfma_bf16(ops_a[4 * i + q].bitcast(fx.BFloat16), xb, acc_a)
            acc_b = mfma_bf16(ops_b[4 * i + q].bitcast(fx.BFloat16), xb, acc_b)
    return acc_a, acc_b


@traced
def gemv_put(c, accs, lr_a, n_a, lr_b, n_b, t0, n):
    """The waves' accumulators summed -> bf16 PROJ rows of tokens t0 .. t0 + n
    (``n_a`` / ``n_b`` rows from logical rows ``lr_a`` / ``lr_b``; none for a CTA
    without weights)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    gpu.barrier()  # every wave done reading x: ``red`` may share its LDS
    fx.ptr_store(accs[0], red + (wave * 64 + lane) * 4)
    fx.ptr_store(accs[1], c["red2"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    r = tid % ROWS
    tt = tid // ROWS
    if tid < ROWS * n:
        if (r < n_a) & (lr_a + r < NPROJ):
            v = bf16r(row_sum(red, r, tt))
            c["put"](c["proj"], (t0 + tt) * NPROJ + proj_row(lr_a + r), v)
        if (r < n_b) & (lr_b + r < NPROJ):
            vb = bf16r(row_sum(c["red2"], r, tt))
            c["put"](c["proj"], (t0 + tt) * NPROJ + proj_row(lr_b + r), vb)
    gpu.barrier()


# ---------------------------------------------------------------- kda
NCONV = 2 * HD + VCH  # conv channels a task computes: its head's q and k, its v chunk


def kda_meta(c, r):
    """This CTA's round-``r`` task and its request's metadata (scalar loads). A
    head's NVCH chunk tasks run in one round on consecutive CTAs: they read each
    other's conv-state channels (``kda_inputs``)."""
    s, L, a = c["S"], c["L"], c["args"]
    ntasks = (s // L) * NH * NVCH
    gi = c["bid"] - s
    per_round = (BLOCKS - s) // NVCH
    task = (gi // NVCH + r * per_round) * NVCH + gi % NVCH
    valid = (gi >= 0) & (gi < per_round * NVCH) & (task < ntasks)
    t = fx.max(fx.min(task, ntasks - 1), 0)
    m = {"valid": valid, "req": t // (NH * NVCH), "h": (t // NVCH) % NH, "ch": t % NVCH}
    req = m["req"]
    m["acc_idx"] = sld_i32(a["num_acc"], req) - 1
    slots = [sld_i32(a["st_idx"], req * a["st_stride"] + j) for j in range(L)]
    m["slots"] = slots
    m["slot_c"] = slots[0]
    slot0 = slots[0]
    for j in range(1, L):
        slot0 = (m["acc_idx"] == j).select(slots[j], slot0)
    m["slot0"] = slot0
    return m


def kda_stage(c):
    """The task's inputs that need no hand-off, issued before its first poll: the
    conv channels' window / weights, the initial state rows, the f_b rows of this
    wave and the per-head constants."""
    L, sl, tid, lane, wave = c["L"], c["SL"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    m = c["tk"]
    h, ch = m["h"], m["ch"]
    k = {
        "acc_idx": m["acc_idx"],
        "slot_c": m["slot_c"],
        "slot0": m["slot0"],
        "slots": m["slots"],
    }
    # conv: thread < NCONV owns a channel: q / k channel d of the head, or v
    # channel d of this task's chunk
    tc = fx.min(tid, NCONV - 1)
    isqk = tc < 2 * HD
    part = isqk.select(tc // HD, fx.Int32(2))
    d = isqk.select(tc % HD, ch * VCH + (tc - 2 * HD))
    chan = part * PROJ + h * HD + d
    k["own"] = (part == 2) | ((d // VCH) == ch)
    k["part"], k["d"], k["chan"] = part, d, chan
    cs = a["conv_st"] + fx.Int64(fx.max(k["slot_c"], 0)) * fx.Int64(a["cs_seq"]) * 2
    co = chan * a["cs_dim"]  # the channel's first entry (bf16 elements)
    k["cs"], k["co"] = cs, co
    k["cw"] = fx.Vector(
        bo.buffer_load(rsrc(a["conv_w"]), chan * CONV_W, vec_width=4, dtype=T.f32)
    )
    k["win"] = [
        ld_bf(cs, co + (k["acc_idx"] + j) * a["cs_tok"]) for j in range(CONV_W - 1)
    ]
    k["old"] = [
        ld_bf(cs, co + (k["acc_idx"] + 1 + j) * a["cs_tok"]) for j in range(sl - L)
    ]
    # recurrence: thread -> state row v = tid / 32, k columns (tid % 32) 4 ..
    vr = ch * VCH + tid // 32
    k0 = (tid % 32) * 4
    k["head_off"] = (h * HD + vr) * HD + k0
    base0 = a["rstate"] + fx.Int64(fx.max(k["slot0"], 0)) * fx.Int64(a["rs_seq"]) * 4
    k["st"] = fx.Vector(
        bo.buffer_load(rsrc(base0), k["head_off"], vec_width=4, dtype=T.f32)
    )
    # f_b: wave w's rows 16 w .. of head h, a lane 64 B of its row
    fr = h * HD + wave * 16 + lane % 16
    fb = (fr * HD + (lane // 16) * 32) // 2
    k["fb"] = [
        fx.Vector(bo.buffer_load(rsrc(a["w_fb"]), fb + 4 * q, vec_width=4, dtype=T.i32))
        for q in range(4)
    ]
    k["aa"] = hw_exp2(ld_f32(a["a_log"], h) * fx.Float32(LOG2E))
    k["dtb"] = [
        ld_f32(a["dt_bias"], h * HD + wave * 16 + 4 * (lane // 16) + i)
        for i in range(4)
    ]
    return k


@traced
def kda_inputs(c, k, req, h):
    """One poll batch for everything the task reads from PROJ: its conv channels'
    L inputs, f_a and beta. The spec conv update of request ``req``'s L tokens
    -> new conv state of the owned channels; q / k (raw conv outputs) and v to
    LDS; f_a and sigmoid(beta) to LDS."""
    L, sl, tid = c["L"], c["SL"], c["tid"]
    a = c["args"]
    n_fa = L * HD
    nfa = (n_fa + THREADS - 1) // THREADS
    specs = [
        (c["proj"], (req * L + j) * NPROJ + k["chan"], 1) for j in range(L)
    ]
    fa_at = []
    for i in range_constexpr(nfa):
        e = fx.min(tid + THREADS * i, n_fa - 1)
        fa_at.append(e)
        specs.append((c["proj"], (req * L + e // HD) * NPROJ + FA_0 + e % HD, 1))
    bt = fx.min(tid, L - 1)
    specs.append((c["proj"], (req * L + bt) * NPROJ + BETA_0 + h, 1))
    got = c["poll"](specs)
    for i in range_constexpr(nfa):
        if tid + THREADS * i < n_fa:
            fx.ptr_store(fx.BFloat16(f32_of(got[L + i][0])), c["fa"] + fa_at[i])
    if tid < L:
        fx.ptr_store(sigmoid(f32_of(got[L + nfa][0])), c["beta"] + tid)
    xv = [f32_of(got[j][0]) for j in range(L)]
    k["xv"] = xv
    if tid < NCONV:
        seq = k["win"] + xv
        for j in range_constexpr(L):
            acc = fx.Float32(0.0)
            for q in range_constexpr(CONV_W):
                acc = acc + k["cw"][q] * seq[j + q]
            o = bf16r(acc * sigmoid(acc))
            if k["part"] < 2:
                fx.ptr_store(o, c["qkl"] + ((2 * j + k["part"]) * HD + k["d"]))
            if k["part"] == 2:
                fx.ptr_store(o, c["vloc"] + (j * VCH + (tid - 2 * HD)))
    gpu.barrier()


@traced
def kda_gates(c, k, req, h):
    """g1 = f_b f_a by MFMA (a wave 16 rows) -> lower-bound gate -> decay in LDS;
    q / k l2-normed (q scaled)."""
    L, tid, lane, wave = c["L"], c["tid"], c["lane"], c["wave"]
    t = fx.min(lane % 16, L - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    stamp(c["on"], c["tls"], tid, 17)
    for q in range_constexpr(4):
        kb = (lane // 16) * 32 + 8 * q
        xb = fx.Vector(
            fx.ptr_load(c["fa"] + (t * HD + kb), result_type=fx.Vector.make_type(8, fx.BFloat16))
        )
        acc = mfma_bf16(k["fb"][q].bitcast(fx.BFloat16), xb, acc)
    for i in range_constexpr(4):
        kk = wave * 16 + 4 * (lane // 16) + i
        g = bf16r(acc[i]) + k["dtb"][i]
        gate = fx.Float32(c["lb"]) * sigmoid(k["aa"] * g)
        if lane % 16 < L:
            fx.ptr_store(hw_exp2(gate * fx.Float32(LOG2E)), c["decay"] + (t * HD + kk))
    stamp(c["on"], c["tls"], tid, 18)
    # l2norm: q / k row r (2 L rows of 128) on wave r % 8
    for rr in range_constexpr((2 * L + WAVES - 1) // WAVES):
        row = fx.min(wave + WAVES * rr, 2 * L - 1)
        x0 = fx.ptr_load(c["qkl"] + (row * HD + lane))
        x1 = fx.ptr_load(c["qkl"] + (row * HD + 64 + lane))
        ss = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
        sc = hw_rsq(ss + fx.Float32(1e-6))
        sc = (row % 2 == 0).select(sc * fx.Float32(HD**-0.5), sc)
        if wave + WAVES * rr < 2 * L:
            fx.ptr_store(x0 * sc, c["qkl"] + (row * HD + lane))
            fx.ptr_store(x1 * sc, c["qkl"] + (row * HD + 64 + lane))
    gpu.barrier()


@traced
def kda_recur(c, k, req, h, ch):
    """State rows 16 ch .. of head h: the recurrence over the L tokens, a state
    written a token, bf16 outputs to LDS ``ol`` and their partial norms (ONRM)."""
    L, tid, lane = c["L"], c["tid"], c["lane"]
    a = c["args"]
    v = tid // 32
    k0 = (tid % 32) * 4
    live0 = k["slot0"] > 0
    stv = [live0.select(k["st"][e], fx.Float32(0.0)) for e in range(4)]
    stamp(c["on"], c["tls"], tid, 12)
    # every token's LDS operands first: the chain below is then ALU and DPP only
    qv = [[fx.ptr_load(c["qkl"] + ((2 * j) * HD + k0 + e)) for e in range(4)] for j in range(L)]
    kv = [[fx.ptr_load(c["qkl"] + ((2 * j + 1) * HD + k0 + e)) for e in range(4)] for j in range(L)]
    dk = [[fx.ptr_load(c["decay"] + (j * HD + k0 + e)) for e in range(4)] for j in range(L)]
    vl = [fx.ptr_load(c["vloc"] + (j * VCH + v)) for j in range(L)]
    bt = [fx.ptr_load(c["beta"] + j) for j in range(L)]
    for j in range_constexpr(L):
        stv = [stv[e] * dk[j][e] for e in range(4)]
        p = sum4(stv, kv[j])
        p = sum32_dpp(p)
        vn = (vl[j] - p) * bt[j]
        stv = [stv[e] + vn * kv[j][e] for e in range(4)]
        o = bf16r(sum32_dpp(sum4(stv, qv[j])))
        o = live0.select(o, fx.Float32(0.0))
        slot = k["slots"][j]
        # branch-free: a null / dead slot's store lands past the bounded descriptor, and every lane of a
        # row's 32 holds the same o
        bo.buffer_store(
            fx.Vector.from_elements(stv, fx.Float32),
            rsrc(a["rstate"] + fx.Int64(fx.max(slot, 0)) * fx.Int64(a["rs_seq"]) * 4, NH * HD * HD * 4),
            k["head_off"],
            mask=(slot > 0) & live0,
            cache_modifier=c["st_cm"],
        )
        fx.ptr_store(o, c["ol"] + (j * VCH + v))
        if const_expr(j == 0):
            stamp(c["on"], c["tls"], tid, 13)
    stamp(c["on"], c["tls"], tid, 14)
    gpu.barrier()
    if tid < L:
        ss = fx.Float32(0.0)
        for e in range_constexpr(VCH):
            x = fx.ptr_load(c["ol"] + (tid * VCH + e))
            ss = ss + x * x
        c["put"](c["onrm"], ((req * NH + h) * L + tid) * NVCH + ch, ss)
    gpu.barrier()
    stamp(c["on"], c["tls"], tid, 15)


@traced
def kda_norm(c, k, req, h, ch):
    """Rows 16 ch ..: gated RMSNorm over the head's 128 (every chunk's partial)
    -> core_out."""
    L, tid = c["L"], c["tid"]
    a = c["args"]
    if tid < L * VCH:
        j = tid // VCH
        e = tid % VCH
        vr = ch * VCH + e
        got = c["poll"](
            [(c["onrm"], ((req * NH + h) * L + j) * NVCH + q, 1) for q in range(NVCH)]
            + [(c["proj"], (req * L + j) * NPROJ + G2_0 + h * HD + vr, 1)]
        )
        stamp(c["on"], c["tls"], tid, 16)
        ss = sum_polled(got[:NVCH])
        r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["onorm_eps"]))
        y = (
            fx.ptr_load(c["ol"] + (j * VCH + e))
            * r
            * ld_bf(a["on_w"], vr)
            * sigmoid(f32_of(got[NVCH][0]))
        )
        bo.buffer_store(
            y.to(fx.BFloat16), rsrc(a["core_out"]), (req * L + j) * PROJ + h * HD + vr
        )
    gpu.barrier()
    # the head's other chunks read this chunk's q / k channels' window too: the
    # new conv state goes out once every chunk has passed ``kda_inputs``
    # (their ONRM pairs are here)
    sl = c["SL"]
    if (tid < NCONV) & k["own"] & (k["slot_c"] > 0):
        rs = rsrc(k["cs"])
        for j in range_constexpr(sl):
            nv = k["old"][j] if j < sl - L else k["xv"][j - (sl - L)]
            bo.buffer_store(
                fx.Float32(nv).to(fx.BFloat16), rs, k["co"] + j * a["cs_tok"]
            )


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: KdaPreBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, L = key.tokens, key.qlen
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    assert 1 <= s <= MAX_S and L <= 16 and key.nblocks >= 1
    # past XS_MAX rows x goes through LDS in two passes of s / 2 rows, with the
    # reduction buffers beside it; below, x takes the whole LDS (the buffers
    # share it) and the blocks that do not fit are loaded straight from global
    npass = 1 if s <= XS_MAX else 2
    nt = -(-s // npass)
    xl_bytes = XL_BYTES if npass == 1 else XL_BYTES - 2 * WAVES * 64 * 4 * 4
    xb_lds = min(PER_WAVE * 4 * WAVES, (xl_bytes // nt - 16) // 64)
    xrow = xb_lds * 64 + 16

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, nt * xrow // 4, 16]

    @fx.struct
    class KdaLds:
        fa: fx.Array[fx.BFloat16, L * HD, 16]
        qkl: fx.Array[fx.Float32, L * 2 * HD, 16]
        decay: fx.Array[fx.Float32, L * HD, 16]
        vloc: fx.Array[fx.Float32, L * VCH, 16]
        ol: fx.Array[fx.Float32, L * VCH, 16]
        beta: fx.Array[fx.Float32, 16, 16]

    @fx.struct
    class GemvAcc:
        v: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]

    @fx.union
    class RoleLds:
        gemv: GemvLds
        acc: GemvAcc
        kda: KdaLds

    @fx.struct
    class SmemSep:
        tls: fx.Array[fx.Int64, TL_POINTS, 16]
        red: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]

    @fx.struct
    class Smem:
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_kda_pre_s{s}_l{L}_nb{key.nblocks}_d{int(key.delta)}"
        f"_t{int(key.timeline)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kda_pre(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(SmemSep if npass > 1 else Smem).peek()
        role = alloc.allocate(RoleLds)
        gl, kl, ga = role.gemv.peek(), role.kda.peek(), role.acc.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(REGIONS))
        c = {
            "S": s,
            "L": L,
            "NS": ns,
            "NB": key.nblocks,
            "SL": key.state_len,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "delta": key.delta,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "onorm_eps": key.onorm_eps,
            "lb": key.lower_bound,
            "write_idx": key.write_idx,
            "st_cm": CM_NT | CM_DEV,
            "XB": xb_lds,
            "XROW": xrow,
            "put": mb.put,
            "poll": mb.poll,
            "red": lds.red.ptr if npass > 1 else ga.v.ptr,
            "red2": (lds.red.ptr if npass > 1 else ga.v.ptr) + WAVES * 64 * 4,
            "NT": nt,
            "xl": gl.xl.ptr,
            "xl_b": fx.Int32(fx.ptrtoint(gl.xl.ptr)),
            "fa": kl.fa.ptr,
            "qkl": kl.qkl.ptr,
            "decay": kl.decay.ptr,
            "vloc": kl.vloc.ptr,
            "ol": kl.ol.ptr,
            "beta": kl.beta.ptr,
            "xbuf": scratch + fx.Int64(lay["x"][0]),
            "args": {
                "prefix": prefix,
                "delta": delta,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "w_in": w_in,
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "on_w": on_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
                "core_out": core_out,
            },
        }
        for region in REGIONS:
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        c["on"], c["tls"] = on, tls
        stamp_begin(on, tls, tid, TL_POINTS)
        c["tk"] = kda_meta(c, 0)
        if bid < s:
            stage_attn(c, bid)
        gi = bid - s
        ng = BLOCKS - s
        rows_b = -(-(NPROJ - ROWS * ng) // ng)
        isg = bid >= s
        lr_a = ROWS * gi
        n_a = isg.select(fx.Int32(ROWS), fx.Int32(0))
        lr_b = ROWS * ng + rows_b * gi
        n_b = isg.select(fx.max(fx.min(fx.Int32(rows_b), NPROJ - lr_b), 0), fx.Int32(0))
        ops_a, ops_b = gemv_weights(c, lr_a, n_a, lr_b, n_b)
        stamp(on, tls, tid, 27)
        xd = x_direct(c)
        stamp(on, tls, tid, 5)
        accs = gemv_mfma(c, ops_a, ops_b, xd)
        stamp(on, tls, tid, 19)
        if const_expr(npass == 1):
            gemv_put(c, accs, lr_a, n_a, lr_b, n_b, 0, s)
        else:
            # the second pass's x streams in while the first pass's sums go out
            gpu.barrier()
            me = uniform(c["wave"]) >= WAVES - NSTAGER
            if me:
                stage_x(c, nt, s - nt, poll=False, wait=False)
            gemv_put(c, accs, lr_a, n_a, lr_b, n_b, 0, nt)
            if me:
                vm_wait(0)
            gpu.barrier()
            accs = gemv_mfma(c, ops_a, ops_b, xd)
            gemv_put(c, accs, lr_a, n_a, lr_b, n_b, nt, s - nt)
        stamp(on, tls, tid, 21)
        k = kda_stage(c)
        stamp(on, tls, tid, 20)
        nrounds = -(-(s // L) * NH // ((BLOCKS - s) // NVCH))
        for r in range_constexpr(nrounds):
            if r > 0:
                c["tk"] = kda_meta(c, r)
                k = kda_stage(c)
            tk = c["tk"]
            if tk["valid"]:
                stamp(on, tls, tid, 7)
                kda_inputs(c, k, tk["req"], tk["h"])
                stamp(on, tls, tid, 8)
                kda_gates(c, k, tk["req"], tk["h"])
                stamp(on, tls, tid, 9)
                kda_recur(c, k, tk["req"], tk["h"], tk["ch"])
                stamp(on, tls, tid, 10)
                kda_norm(c, k, tk["req"], tk["h"], tk["ch"])
                stamp(on, tls, tid, 11)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kda_pre(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_in,
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            on_w,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            core_out,
            scratch,
            layer,
            epoch,
            health,
            alarm,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(kda_pre, launch)
    _BUILDS[key] = launch
    return launch
