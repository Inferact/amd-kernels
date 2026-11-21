# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.: atom/mono layer_post
"""The layer tail: one persistent launch (256 CTAs x 512 threads, all resident, one rank of eight) from the
attention's core output (the o_proj input) to the layer's MoE output: o_proj, its TP8 all-reduce over peer memory
and the MLP AttnRes seam (``oproj``, ``attn_res``), then the latent MoE (``experts``).

The MoE's static weights stream while the AttnRes chain (partials -> reduce -> slice -> gate -> mix) is still
settling: the CTAs that will run the router, latent and shared gate/up GEMVs issue their weights before they first
wait.

    CTAs 0 .. 55     reduce + AttnRes (slice, gate, mix -> x rows + XRDY); then route (0 .. S-1) and the shared
                     SiTU (8 .. 8+S-1)
    CTAs 56 .. 159   o_proj (448 x 16 rows) -> partials to every rank; then the shared gate/up (96 tasks)
    CTAs 160 .. 243  router parts (28) and latent parts (56): weights first
    every CTA        ug; down (0 .. 223) or the shared down (224 .. 255); lnorm (56 .. 83), up (0 .. 55),
                     reduce (200 .. 255)

Hand-offs are tagged (value, tag) pairs or sentinel words (``common.mono.sync``); the tag is the step epoch and
the launch's slot, so no buffer is zeroed between steps. Every wait gives up after 10 s (fail-stop, health words).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from amd_kernels.common.mono.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from amd_kernels.common.mono.ops import (
    ld_i32,
    lds_bytes,
    load_ptr64,
    traced,
    uniform,
)
from amd_kernels.common.mono.plan import (
    BLOCKS,
    LDS_BYTES,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
    source_digest,
)
from amd_kernels.common.mono.ranks import peer_bases
from amd_kernels.common.mono.sync import (
    Mailbox,
    preg,
    shift,
    sreg,
    stall,
)

from . import experts as k2b
from . import gemv
from . import oproj as k2a
from .attn_res import (
    COLS,
    HIDDEN,
    NA,
    stage_gates,
    stage_mix,
    stage_slice,
    step_tag,
    vals_all,
    vals_words,
)

_SOURCES = source_digest("common/mono", "moe/layer_tail_k2_h7168_ranks8")

O0 = NA  # the first CTA past AttnRes's
OTASKS = k2a.OTASKS  # 448
O_N = BLOCKS  # o_proj tasks gi, gi + 256 on every CTA
O_ROUNDS = (OTASKS + O_N - 1) // O_N  # 2
E0 = 160  # router / latent CTAs (their weights in flight under their o_proj)
H0 = 20  # the shared SiTU's CTAs: past the route CTAs (tokens 0 .. 15)
SD0 = k2b.DOWN_TASKS  # 224: the shared down's CTAs
# timeline points (``common.mono.debug``): 0 start, 1 o_proj done / AR reduced, 2 early done / slice, 3 ug,
# 4 down + shared down, 5 lnorm, 6 up, 7 reduce, 8 gate, 9 mix (x published), 10 route, 11 route table,
# 12 latent quant, 13 shared SiTU, 15 INTER loaded, 16 down done, 18 / 19 router x loaded / pushed, 20 router
# logits, 21 route end, 22 ROUTE polled, 23 / 24 route start / parts polled, 26 lnorm polled, 27 up polled,
# 28 up rows in LDS, 29 up MFMA done, 30 LN rows published, 31 reduce polled
TL_POINTS = 32

assert E0 + k2b.R_TASKS + k2b.L_TASKS <= BLOCKS and k2b.G_TASKS <= E0 - O0

REGIONS: tuple[str, ...] = (
    "part",
    "wgt",
    "msq",
    "xrdy",
    "ar",
    "crdy",
    "cmb",
    "ardy",
) + k2b.REGIONS
# the all-reduce sentinel regions, RLOGP and the tagged rows (``K2Build.opt``),
# past every region above
REGIONS = REGIONS + ("ars", "latrs", "finals", "xt", "rlogp")

# ``K2Build.opt`` bits, each bit-exact against opt=0; K2_OPT is the default.
#   1 / 16 / 32  o_proj / latent / final all-reduce via sentinel regions
#                (plain bf16 pair words, reader re-arms; ``sync.put_sent``)
#   2    MLP AttnRes slice loads issued before the all-reduce poll
#   8    x rows also as tagged pairs to the router / latent GEMVs (no XRDY)
#   64   shared-down tasks that up waits on run first
#   256  router K parts push fp32 partials to every rank's RLOGP
#   512  batched polls (route, down, up)
OPT_SENT = 1
OPT_PRELOAD = 2
OPT_TAG_X = 8
OPT_SENT_LATR = 16
OPT_SENT_FINAL = 32
OPT_SENTS = OPT_SENT | OPT_SENT_LATR | OPT_SENT_FINAL
OPT_SDOWN_FIRST = 64
OPT_RLOG_PARTS = 256
OPT_BATCH = 512
K2_OPT = (
    OPT_SENTS | OPT_PRELOAD | OPT_TAG_X | OPT_SDOWN_FIRST | OPT_RLOG_PARTS | OPT_BATCH
)
SENT_ROWS = 16  # the widest served width (``host.SERVED_ROWS``)

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int
    nblocks: int  # the MLP AttnRes block sources
    eps: float = 1e-5  # mlp_res_norm
    out_eps: float = 1e-5  # post_attention_layernorm
    ln_eps: float = 1e-5  # the latent RMSNorm
    reduce: bool = True  # the final reduce here (else the next layer's K1)
    reset: bool = False  # a block-write layer: the prefix restarts at the o_proj output
    # the fault hook (sync.stall): rank fault_rank stalls fault_ticks once, in
    # slot fault_layer's launch of the step at epoch fault_epoch
    fault_rank: int = -1
    fault_epoch: int = 0
    fault_layer: int = 0
    fault_ticks: int = 0
    timeline: bool = False
    # the OPT_* bits
    opt: int = K2_OPT


def k2_sent(key: K2Build) -> bool:
    """(OPT_SENTS) Whether build ``key``'s all-reduces go through the sentinel
    regions. The one predicate the push and the reduce of every one of them
    use: a push the reduce does not take would leave words neither polled nor
    re-armed, and the next launch's reader would hang."""
    return bool(key.opt & OPT_SENTS) and key.tokens <= SENT_ROWS


def sent_layout() -> dict:
    """(OPT_SENTS) One layer parity's sentinel regions, name -> (byte
    offset, bytes): a word a (rank, bf16 pair, row), SENT_ROWS rows."""
    n = {
        "ars": TP_ * SENT_ROWS * HIDDEN // 2 * 4,
        "latrs": TP_ * SENT_ROWS * k2b.LAT // 2 * 4,
        "finals": TP_ * SENT_ROWS * HIDDEN // 2 * 4,
    }
    out, off = {}, 0
    for name, b in n.items():
        out[name] = (off, b)
        off += -(-b // 256) * 256
    return out


def sent_half_bytes() -> int:
    return max(o + n for o, n in sent_layout().values())


def sent_offset() -> int:
    """(OPT_SENTS) Byte offset (from the kernels' peer base) of the sentinel regions, past this kernel's regions at
    their largest. Two layer parities."""
    return -(-peer_bytes(k2b.MAX_S) // 4096) * 4096


def sent_bytes() -> int:
    return 2 * sent_half_bytes()


def rlogp_half_bytes() -> int:
    """(OPT_RLOG_PARTS) One epoch parity's RLOGP, tagged pairs
    `[src rank][part][token][router row]`, SENT_ROWS rows."""
    return TP_ * k2b.R_PARTS * SENT_ROWS * (k2b.E // TP_) * 8


def rlogp_offset() -> int:
    return -(-(sent_offset() + sent_bytes()) // 4096) * 4096


def extra_peer_end() -> int:
    """The end (from the kernels' peer base) of every region the OPT_* bits
    add to the peer buffer."""
    return rlogp_offset() + 2 * rlogp_half_bytes()


def k2_rlogp(key: K2Build) -> bool:
    if not (key.opt & OPT_RLOG_PARTS) or key.tokens > SENT_ROWS:
        return False
    vextra = key.tokens * COLS * 4
    return vals_words(key.tokens, key.nblocks + 1, vextra) >= k2b.R_PARTS * k2b.E


TAG_OFFSET = 7360512  # past the scratch layouts of this kernel and of the KDA kernel at 64 rows, + 1 MiB


def tag_layout() -> dict:
    """(OPT_TAG_X) The tagged rows' scratch region (name -> (byte offset, bytes)): past every launch's layout at
    its largest, so no other launch's plain word sits where these keep tags."""
    return {"xt": (TAG_OFFSET, k2b.MAX_S * HIDDEN // 2 * 8)}


def tag_end() -> int:
    return max(o + n for o, n in tag_layout().values())


ABI = KernelAbi(
    (
        "core",
        "w_o",
        "prefix",
        "blocks",
        "blk_sm",
        "blk_sr",
        "ares_nw",
        "ares_qk",
        "in_nw",
        "w_gate",
        "bias",
        "w_ld",
        "w_sgu",
        "w_sd",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "ln_w",
        "w_up",
        "out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "layer",
        "health",
        "alarm",
        "tl",
    )
)


def peer_layout(s):
    """``oproj``'s AR region, then ``experts``': one parity's (name -> (byte offset, bytes))."""
    pairs = {"ar": k2a.ar_half_pairs(s), **k2b.peer_pairs(s)}
    return pair_layout(pairs.items())


def peer_bytes(s):
    lay = peer_layout(s)
    return 2 * max(o + n for o, n in lay.values())


def scratch_layout(key: K2Build) -> dict:
    s = key.tokens
    a = k2a.scratch_layout(s, key.nblocks)
    end = max(o + n for o, n in a.values())
    b = k2b.scratch_layout(s)
    lay = dict(a)
    for name, (o, n) in b.items():
        lay[name] = ((end + 15) // 16 * 16 + o, n)
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def xl_words(s):
    """The int32 words of the bf16 row tile LDS ``xl`` the GEMV stages share:
    the core tile (o_proj), the early stages' x tiles, every h row (the shared
    down) and the LN tile (up)."""
    tiled = [k2a.OK, HIDDEN // k2b.R_PARTS, HIDDEN // k2b.L_PARTS, HIDDEN, k2b.LAT]
    rows = [(k, min(s, gemv.tile_rows(k))) for k in tiled] + [(k2b.SI, s)]
    return max(n * (k + gemv.LDS_PAD) // 2 for k, n in rows)


@traced
def stage_oproj(c, gi):
    """o_proj tasks gi, gi + 200, gi + 400 (< 448): every task's weights in
    flight at once (one memory latency, not three), then the MFMAs a core tile
    of 16 rows at a time."""
    ops = [k2a.o_loads(c, fx.min(gi + O_N * j, OTASKS - 1)) for j in range(O_ROUNDS)]
    for t0, n in gemv.token_tiles(c["S"]):
        cm = 0
        k2a.load_core(c, t0, n, cm)
        for j in range_constexpr(O_ROUNDS):
            t = gi + O_N * j
            if t < OTASKS:
                k2a.stage_o(c, t, ops[j], t0, n)


_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    assert 1 <= s <= k2b.MAX_S
    # the AttnRes stages' delta tile beside their slices
    vextra = s * COLS * 4
    lay = scratch_layout(key)
    play = peer_layout(s)
    half = peer_bytes(s) // 2

    # the route table: a slot's kof row (``few_rows``), else its pick tiles
    few = k2b.few_rows(s)
    pad = few and k2b.inter_lds(s)
    nkof = k2b.max_u(s) * s if few else 1
    nslot = 1 if few else k2b.max_u(s) + 1
    npick = 1 if few else s * k2b.TOPK
    ntile = 1 if few else k2b.max_tiles(s)

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * k2b.TOPK, 16]
        eslot: fx.Array[fx.Int32, k2b.E, 16]
        expert: fx.Array[fx.Int32, k2b.max_u(s), 16]
        kof: fx.Array[fx.Int32, nkof, 16]
        first: fx.Array[fx.Int32, nslot, 16]
        tfirst: fx.Array[fx.Int32, nslot, 16]
        order: fx.Array[fx.Int32, npick, 16]
        tslot: fx.Array[fx.Int32, ntile, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    # red, and as words the route table's token bitmaps (``load_route``)
    @fx.struct
    class RedF:
        v: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    @fx.struct
    class RedI:
        v: fx.Array[fx.Int32, WAVES * 64 * 4, 16]

    @fx.union
    class RedLds:
        f: RedF
        i: RedI

    @fx.struct
    class EarlyLds:
        xl: fx.Array[fx.Int32, xl_words(s), 16]
        av: fx.Array[fx.Float32, THREADS, 16]

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, vals_words(s, ns, vextra), 16]
        dl: fx.Array[fx.Float32, s * COLS, 16]

    @fx.struct
    class UgLds:
        latq: fx.Array[fx.Int32, s * k2b.lq_row(pad), 16]
        latsc: fx.Array[fx.Int32, s * k2b.ls_row(pad) if few else k2b.latsc_words(s), 16]

    # every pick's INTER in LDS, or (polled at use) none
    nq = s * k2b.TOPK if k2b.inter_lds(s) else 1

    @fx.struct
    class DownLds:
        interq: fx.Array[fx.Int32, nq * k2b.ip_row(pad), 16]
        intsc: fx.Array[fx.Int32, nq * k2b.UG_GROUPS, 16]
        contrib: fx.Array[fx.Float32, s * k2b.TOPK * k2b.ROWS, 16]

    @fx.union
    class StageLds:
        early: EarlyLds
        attn: AttnLds
        ug: UgLds
        down: DownLds

    @fx.struct
    class K2Lds:
        red: RedLds
        route: RouteLds
        stage: StageLds

    @fx.union
    class MainLds:
        k2: K2Lds

    @fx.struct
    class Smem:
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    # LLVM's own over-the-limit error doesn't stop a build (COMPILE_ONLY)
    assert lds_bytes(Smem) + lds_bytes(MainLds) <= LDS_BYTES, lds_bytes(
        Smem
    ) + lds_bytes(MainLds)

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_k2_s{s}_nb{key.nblocks}_x{int(key.reset)}_r{int(key.reduce)}"
        f"_t{int(key.timeline)}"
    )
    if key.opt:
        name += f"_o{key.opt}"
    if key.fault_rank >= 0:
        name += (
            f"_f{key.fault_rank}_{key.fault_epoch}_{key.fault_layer}_{key.fault_ticks}"
        )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        main = alloc.allocate(MainLds)
        rd = main.k2.red
        rl = main.k2.route.peek()
        st = main.k2.stage
        el, al, ul, dl = st.early.peek(), st.attn.peek(), st.ug.peek(), st.down.peek()
        if const_expr(key.fault_rank >= 0):
            stall(
                key.fault_rank,
                key.fault_epoch,
                key.fault_layer,
                key.fault_ticks,
                rank,
                epoch,
                layer,
            )
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(REGIONS))
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP_)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("rlog", "lat", "latq4", "latr", "final", "ar"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
        c = {
            "S": s,
            "NS": ns,
            "NB": key.nblocks,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "delta": True,
            "reset": key.reset,
            "vals_all": vals_all(s, ns, vextra),
            "eps": key.eps,
            "out_eps": key.out_eps,
            "ln_eps": key.ln_eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": rd.f.peek().v.ptr,
            "redi": rd.i.peek().v.ptr,
            "peer": peer,
            "ar_peer": peer["ar"],
            "ar_own": peer["ar_own"],
            "xl": el.xl.ptr,
            "vals": al.vals.ptr,
            "delta_lds": al.dl.ptr,
            "latq": ul.latq.ptr,
            "latsc": ul.latsc.ptr,
            "av": el.av.ptr,
            "interq": dl.interq.ptr,
            "intsc": dl.intsc.ptr,
            "contrib": dl.contrib.ptr,
            "scan": rl.scan.ptr,
            "rt": {
                "route": rl.route.ptr,
                "eslot": rl.eslot.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
                "first": rl.first.ptr,
                "tfirst": rl.tfirst.ptr,
                "order": rl.order.ptr,
                "tslot": rl.tslot.ptr,
            },
            "xbuf": scratch + fx.Int64(lay["xrow"][0]),
            "pad": pad,
            "xflags": NA,
            "tl_on": key.timeline,
            "tls": lds.tls.ptr,
            "ln": scratch + fx.Int64(lay["ln"][0]),
            "shp": scratch + fx.Int64(lay["shp"][0]),
            "args": {
                "core": core,
                "w_o": w_o,
                "prefix": prefix,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "x_dbg": 0,
                "w_gate": w_gate,
                "bias": bias,
                "w_ld": w_ld,
                "w_sgu": w_sgu,
                "w_sd": w_sd,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "ln_w": ln_w,
                "w_up": w_up,
                "out": out,
            },
        }
        # the OPT_* bits' switches and regions
        c["tag_x"] = bool(key.opt & OPT_TAG_X)
        c["batch"] = bool(key.opt & OPT_BATCH)
        rlogp = k2_rlogp(key)
        c["rlog_parts"] = rlogp
        if const_expr(rlogp):
            roff = rlogp_offset()
            rhalf = fx.Int64(par) * rlogp_half_bytes()
            c["rlogp"] = [shift(preg(b, roff, "rlogp"), rhalf) for b in bases]
            c["rlogp_own"] = shift(preg(own, roff, "rlogp"), rhalf)
        if const_expr(key.opt & OPT_TAG_X):
            c["xt"] = sreg(scratch, tag_layout()["xt"][0], "xt")
        sent = k2_sent(key)
        if const_expr(sent):
            # this layer parity's sentinel regions
            lpar = fx.Int64(layer % 2) * sent_half_bytes()
            sl = sent_layout()
            c["sent_ar"] = bool(key.opt & OPT_SENT)
            c["sent_latr"] = bool(key.opt & OPT_SENT_LATR)
            c["sent_final"] = bool(key.opt & OPT_SENT_FINAL)
            c["mb"] = mb
            for region in ("ars", "latrs", "finals"):
                off = sent_offset() + sl[region][0]
                c[region + "_peer"] = [shift(preg(b, off, region), lpar) for b in bases]
                c[region + "_own"] = shift(preg(own, off, region), lpar)
        for region in (
            "part",
            "wgt",
            "msq",
            "xrdy",
            "sgu",
            "h",
            "rpart",
            "lpart",
            "route",
            "inter",
            "isc",
            "lnrdy",
            "shrdy",
        ):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        # ---- o_proj on every CTA past AttnRes's
        eb = bid - E0
        lb = eb - k2b.R_TASKS
        is_r = (bid >= E0) & (eb < k2b.R_TASKS)
        is_l = (bid >= E0 + k2b.R_TASKS) & (lb < k2b.L_TASKS)
        # o_proj first (every AttnRes CTA waits on the slowest partial; loads
        # retire in order), the early weights after: they land before x does
        if bid < NA:
            stage_oproj(c, bid)
        if is_r:
            stage_oproj(c, bid)
            stamp(on, tls, tid, 1)
            rops = k2b.early_loads(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS)
            k2b.early_compute(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS, rops)
        if is_l:
            stage_oproj(c, bid)
            stamp(on, tls, tid, 1)
            lops = k2b.early_loads(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS)
            k2b.early_compute(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS, lops)
        if (bid >= O0) & ~is_r & ~is_l:
            stage_oproj(c, bid)
            stamp(on, tls, tid, 1)
            gi = bid - O0
            if gi < k2b.G_TASKS:
                k2b.early_compute(c, 2, gi, 0, k2b.early_loads(c, 2, gi, 0))
        # ---- AttnRes: the reduced o_proj -> x; then route / SiTU
        if bid < NA:
            if const_expr(key.opt & OPT_PRELOAD):
                # the slice's loads before the AR poll
                sld = k2a.slice_loads(c, bid)
                k2a.stage_reduce(c, bid)
                stamp(on, tls, tid, 1)
                k2a.stage_slice_pre(c, bid, sld)
            else:
                k2a.stage_reduce(c, bid)
                stamp(on, tls, tid, 1)
                stage_slice(c, bid)
            stamp(on, tls, tid, 2)
            stage_gates(c, bid)
            stamp(on, tls, tid, 8)
            stage_mix(c, bid)
            if const_expr(sent and (key.opt & OPT_SENT)):
                # the sentinel words stage_reduce read, off its critical path
                k2a.rearm_ar(c, bid)
            stamp(on, tls, tid, 9)
            # the shared SiTU of token t on CTA (t + H0) % 56, first: SGU lands
            # before the logits
            hb = (bid + NA - H0) % NA
            for rr in range_constexpr((s + NA - 1) // NA):
                if hb + NA * rr < s:
                    k2b.stage_h(c, hb + NA * rr)
            stamp(on, tls, tid, 13)
            for rr in range_constexpr((s + NA - 1) // NA):
                if bid + NA * rr < s:
                    stamp(on, tls, tid, 23)
                    if const_expr(rlogp):
                        k2b.route_parts(c, bid + NA * rr)
                    stamp(on, tls, tid, 24)
                    k2b.stage_route(c, bid + NA * rr)
                    stamp(on, tls, tid, 21)
                    stamp(on, tls, tid, 21)
            stamp(on, tls, tid, 10)
        if bid >= NA:
            stamp(on, tls, tid, 2)
        # ---- the routed experts
        # the latent (ready ~10 us before the route) quantized first
        if const_expr(k2b.few_rows(s)):
            k2b.load_latq4(c)
        else:
            k2b.load_latq(c)
        stamp(on, tls, tid, 12)
        nt = k2b.load_route(c)
        stamp(on, tls, tid, 11)
        total = nt * k2b.UG_GROUPS
        last = total - 1
        # reversed placement: the round's leftover tasks go to the high CTAs,
        # whose early work ends first (the AttnRes / route CTAs start ug last)
        task = BLOCKS - 1 - bid
        ops = k2b.ug_loads(c, fx.min(task, last))
        while task < total:
            nxt = task + BLOCKS
            ops_n = k2b.ug_loads(c, fx.min(nxt, last))
            k2b.stage_ug(c, task, ops)
            task = nxt
            ops = ops_n
        stamp(on, tls, tid, 3)
        if bid < k2b.DOWN_TASKS:
            dops = k2b.down_loads(c, bid, nt, fx.min(c["wave"], nt - 1))
            if const_expr(k2b.inter_lds(s)):
                k2b.load_inter(c)
            stamp(on, tls, tid, 15)
            k2b.stage_down(c, bid, nt, dops)
            stamp(on, tls, tid, 16)
        if const_expr(key.opt & OPT_SDOWN_FIRST):
            k2b.run_sdown_first(c, bid, w_sd)
            k2b.run_sdown_rest(c, bid, w_sd)
        else:
            k2b.run_sdown_rest(c, bid, w_sd)
            k2b.run_sdown(c, bid, w_sd)
        stamp(on, tls, tid, 4)
        if (bid >= k2b.LN0) & (bid < k2b.LN0 + k2b.LN_TASKS):
            k2b.run_lnorm(c, bid)
        stamp(on, tls, tid, 5)
        if bid < k2b.UP_TASKS * k2b.up_splits(s):
            k2b.run_up(c, bid)
        stamp(on, tls, tid, 6)
        if const_expr(key.reduce):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if (bid >= k2b.RED0) & (bid < k2b.RED0 + k2b.RED_TASKS):
                k2b.stage_reduce(c, bid - k2b.RED0)
        stamp(on, tls, tid, 7)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            w_o,
            prefix,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_gate,
            bias,
            w_ld,
            w_sgu,
            w_sd,
            w13,
            w13s,
            w2,
            w2s,
            ln_w,
            w_up,
            out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            health,
            alarm,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    _BUILDS[key] = launch
    return launch

TP_ = k2b.TP
