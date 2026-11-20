# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""FP8-KV sliding-window attention over 128-token pages staged through an LDS ring, up to 8 query rows per request.

A workgroup is one (request, KV head, split). Its waves are 4 key lanes x row halves of ``rows_per_half`` rows (one
half up to that many rows). Key lane j of every half attends keys 32 j .. 32 j + 31 of each page, so the waves of a
SIMD read the same chunk, each for its own rows (<= 4 rows of O: 128 registers, two waves per SIMD).

A step is one page of the split's contiguous page range. Every wave DMAs its 1 KB pieces of the page
(``buffer_load ... lds``, lane-linear) into a ``STAGES``-deep LDS ring, ``STAGES - 1`` pages ahead; a step waits for
its page's pieces (``vmcnt``), meets the other waves at a barrier (which also frees the previous step's slot) and
issues the next DMA. The DMAs are inline asm: LLVM would make every LDS read wait for all of them. Token t's 16 B
column c sits at column ``c ^ swz(t)`` (``_swz``), so a K read (16 tokens per 16 lanes) and a V transpose read (16
tokens per 32 lanes) hit distinct banks. Bytes past the sequence end are out of the DMA's range and land as zeros (an
unwritten slot may hold an e4m3 NaN).

Per chunk: S^T = K Q^T (v_mfma_f32_16x16x32_bf16, 16 keys x 16 heads; Q^T in LDS), online softmax in fp32 with keys
8 g .. 8 g + 7 in lane group g, O^T += V^T P^T straight from S^T. O^T tile dt is dims 16 dt .. 16 dt + 15, so its
V^T operand comes from one ``ds_read_b64_tr_b8`` (no byte transposes) and lane (n, g) holds dims 16 dt + 4 g + e of
head n; an MXFP8 group of 32 dims spans the 4 lanes of a head. The running max is lazy: a row keeps its reference max
until a score passes it by 2^TAU_LOG2, so the rescale branch is rare and most chunks are one basic block;
P = exp2(s c - m) <= 2^TAU_LOG2 is rounded to bf16 (the same relative rounding), sums stay fp32.

After the key loop the waves of a half merge its rows through LDS (the ring is free), and with splits > 1 the last
split of a request (a ticket) combines the partials in split order, so the result does not depend on arrival order.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import CH, MAX_QL
from .kernel import (
    AGENT_SCOPE,
    D,
    HG,
    TINY_F32,
    NDT,
    NKS,
    PART_F,
    PRELOAD_HINTS,
    TOK_B,
    _as_f32,
    _bf16x8_e4m3,
    _bits,
    _exp2,
    _global_atomic_add,
    _global_i32_ptr,
    _maxf_nn,
    _mfma,
    _permlane_swap,
    _pk_bf16,
    _pk_fp8,
    _xlane4,
)

PAGE = 128
PAGE_B = PAGE * TOK_B  # one (page, KV head): 32 KB
STAGES = 4
TAU_LOG2 = 8.0
KEY_LANES = 4
ROWS_PER_HALF = 4
DMA_B = 1024  # bytes per wave per DMA instruction (64 lanes x 16 B)


def _swz(t):
    """The 16 B column XOR of token ``t``: column bits (t0, t1, t2 ^ t4, t3), a bijection over the 16 tokens of a
    K read (bits 0, 1, 3, 4; bit 2 fixed) and of a V transpose read (bits 0-3; bit 4 fixed)."""
    return (t & 3) | (((t >> 2) ^ (t >> 4)) & 1) << 2 | (t & 8)


def _lds_tr8(addr):
    """``ds_read_b64_tr_b8`` at LDS byte ``addr``: lane 16 G + 8 h + k gets byte k of the 8 bytes addressed by
    lanes 16 G + 2 j + h, j = 0 .. 7."""
    ptr = _llvm.IntToPtrOp(_ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int32(addr))).result
    return Vec(_rocdl_ops.ds_read_tr8_b64(Vec.make_type(2, fx.Int32), ptr).result)


def _imax(a, b):
    return (a > b).select(a, b)


def _mxfp8_group8(words):
    """MXFP8 of a 32-value group held 8 per lane by the 4 lanes n, n + 16, n + 32, n + 48: ``words`` are this lane's
    4 packed-bf16 dwords. Returns its 2 code dwords and the group's e8m0 scale (fp32, 0..254), as
    ``kernel._mxfp8_group`` computes them (the amax an integer max over the magnitude bits, so a NaN stays NaN)."""
    mags = []
    for wd in words:
        mags += [wd & fx.Int32(0x7FFF), (wd >> 16) & fx.Int32(0x7FFF)]
    while len(mags) > 1:
        mags = [_imax(mags[i], mags[i + 1]) for i in range(0, len(mags), 2)]
    mag = mags[0]
    a, b = _permlane_swap(_rocdl.permlane32_swap, mag, mag)
    mag = _imax(a, b)
    a, b = _permlane_swap(_rocdl.permlane16_swap, mag, mag)
    mag = _imax(a, b)
    amax = fx.arith.maximumf(fx.Float32(TINY_F32), _as_f32(mag << 16))
    sb = fx.math.ceil(fx.math.log2(amax / fx.Float32(448.0))) + fx.Float32(127.0)
    sb = fx.arith.minimumf(fx.arith.maximumf(sb, fx.Float32(0.0)), fx.Float32(254.0))
    rs = fx.Float32(fx.math.exp2(fx.Float32(127.0) - sb))
    dws = []
    for i in range_constexpr(2):
        w0, w1 = words[2 * i], words[2 * i + 1]
        lo = _pk_fp8(_as_f32(w0 << 16) * rs, _as_f32(w0 & fx.Int32(-65536)) * rs, fx.Int32(0), False)
        dws.append(_pk_fp8(_as_f32(w1 << 16) * rs, _as_f32(w1 & fx.Int32(-65536)) * rs, lo, True))
    return dws, sb


def _dma16(m0, voff, rsrc):
    """``buffer_load_dwordx4 ... lds``: 16 B per lane to LDS byte m0 + 16 lane (inline asm: no LLVM waitcnt)."""
    _llvm.inline_asm(
        None,
        [fx.as_ir_value(fx.Int32(m0)), fx.as_ir_value(fx.Int32(voff)), fx.as_ir_value(rsrc)],
        "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, 0 offen lds",
        "s,v,s,~{m0}",
        has_side_effects=True,
    )


def _sync():
    """LDS writes done, then ``s_barrier`` (an intrinsic: MLIR erases back-to-back ``gpu.barrier`` ops, and the two
    halves' bodies must meet the same number of barriers)."""
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.rocdl.s_barrier()


def row_halves(ql: int, rows_per_half: int = ROWS_PER_HALF) -> tuple[int, ...]:
    """Rows per half: (ql,) up to ``rows_per_half`` rows, else (rows_per_half, ql - rows_per_half)."""
    return (ql,) if ql <= rows_per_half else (rows_per_half, ql - rows_per_half)


def compile_swa_fp8kv_bf16mfma_gqa16_d128_p128_ring(
    ql: int,
    splits: int,
    window: int,
    causal: bool,
    xcds: int,
    quant: bool = False,
    rows_per_half: int = ROWS_PER_HALF,
):
    """``ql`` query rows per request (1..8), ``splits`` workgroups per (request, KV head) over contiguous page ranges,
    128-token pages; ``xcds`` > 0: a request's splits on one XCD; ``quant``: the MXFP8 o_proj-input epilogue."""
    assert 1 <= ql <= MAX_QL and splits >= 1
    assert 0 < window <= 64 * CH * 4 - max(ql, 6) - CH + 2
    QL, S, W, XCDS = ql, splits, window, xcds
    HALVES = row_halves(ql, rows_per_half)
    NH = len(HALVES)
    NW = KEY_LANES * NH
    NDMA = PAGE_B // (NW * DMA_B)  # DMA instructions per wave and page
    PART_SCOPE = 0 if XCDS else AGENT_SCOPE
    QUANT = quant
    QSZ = QL * NKS * 64 * 4  # Q^T operand words
    RING_I = STAGES * PAGE_B // 4
    MR = 2  # rows of a half merged per round
    O_I = NH * MR * KEY_LANES * NDT * 64 * 4  # merge slots, in the ring after the key loop
    assert O_I <= RING_I and NW * QL * 2 * 64 <= QSZ  # (max, sum) per (wave, row, lane) in the dead Q^T
    name = (
        f"swa_fp8kv_bf16mfma_gqa16_d128_ring_r{QL}_s{S}_w{W}_p{PAGE}_n{NW}"
        + (f"_h{HALVES[0]}" if rows_per_half != ROWS_PER_HALF else "")
        + ("_c" if causal else "")
        + (f"_x{XCDS}" if XCDS else "")
        + ("_mxfp8" if quant else "")
    )

    @fx.struct
    class Shared:
        ring: fx.Array[fx.Int32, RING_I, 16]
        q: fx.Array[fx.Int32, QSZ, 16]
        ticket: fx.Array[fx.Int32, 4]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_reqs: fx.Int32,
        i32_bt_s: fx.Int32,  # block-table row stride (elements)
        i32_page_b: fx.Int32,  # cache page stride (bytes)
        i32_head_b: fx.Int32,  # cache KV-head stride (bytes)
        i32_H: fx.Int32,  # query heads
        i32_hkv: fx.Int32,
        arg_seq: fx.Int64,  # [reqs] int32
        arg_qsl: fx.Int64,  # [reqs + 1] int32
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_q: fx.Int64,  # [rows, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, Hkv, 128, 256] e4m3
        arg_out: fx.Int64,  # [rows, H, 128] bf16
        arg_ks: fx.Int64,  # fp32 K scale
        arg_vs: fx.Int64,  # fp32 V scale
        arg_part: fx.Int64,  # splits: [grid requests * Hkv, S, QL] records of 64 PART_F fp32: O^T, then (max, sum)
        arg_cnt: fx.Int64,  # splits: [grid requests * Hkv] int32 tickets, zero
        i64_part_bytes: fx.Int64,
        i32_rows: fx.Int32,
        f32_c: fx.Float32,  # sm_scale * log2(e)
        arg_qc: fx.Int64,  # quant: [>= ceil(rows / 16) * 16, H * 128] e4m3 codes
        arg_qs: fx.Int64,  # quant: [>= ceil(rows / 32) * 32, H * 4] e8m0 scales
        i32_tile: fx.Int32,  # quant: 0 row-major codes, > 0 the (16, 16) tile layout
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        if const_expr(XCDS > 0):
            # grid.x is padded to XCDS x S: a request's splits share an XCD
            xcd = bx % XCDS
            slot = bx // XCDS
            req = (slot // S) * XCDS + xcd
            split = slot % S
        else:
            req = bx // S
            split = bx % S
        req_ok = req < i32_reqs
        reqc = req_ok.select(req, fx.Int32(0))
        seq_p = _global_i32_ptr(arg_seq)
        qsl_p = _global_i32_ptr(arg_qsl)
        bt = _global_i32_ptr(arg_bt)
        f32p = fx.PointerType.get(T.f32, address_space=fx.AddressSpace.Global, alignment=4)
        # (the scales are made uniform here: a use inside the key loop would wait for every DMA in flight)
        ks_ld = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        vs_ld = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        seq_ld = fx.Int32(seq_p[reqc])
        q0_ld = fx.Int32(qsl_p[reqc])
        q1_ld = fx.Int32(qsl_p[reqc + 1])
        c = _as_f32(fx.rocdl.readfirstlane(T.i32, _bits(f32_c * ks_ld)))
        inv_c = fx.Float32(1.0) / c
        v_scale = _as_f32(fx.rocdl.readfirstlane(T.i32, _bits(vs_ld)))
        neg_inf = fx.Float32(float("-inf"))

        seq_len = fx.Int32(fx.rocdl.readfirstlane(T.i32, seq_ld))
        q0 = fx.Int32(fx.rocdl.readfirstlane(T.i32, q0_ld))
        q1 = fx.Int32(fx.rocdl.readfirstlane(T.i32, q1_ld))
        qlen = req_ok.select(q1 - q0, fx.Int32(0))
        qlen = (qlen > QL).select(fx.Int32(QL), qlen)
        qlen = (seq_len < qlen).select(fx.Int32(0), qlen)  # graph padding: no keys
        ctx = seq_len - qlen
        k_lo = ctx - (W - 1)
        k_lo = (k_lo < 0).select(fx.Int32(0), k_lo)
        p_first = k_lo // PAGE
        n_pg = (qlen > 0).select((seq_len - 1) // PAGE - p_first + 1, fx.Int32(0))
        v_beg = (split * n_pg) // S
        n_st = fx.Int32(fx.rocdl.readfirstlane(T.i32, ((split + 1) * n_pg) // S - v_beg))
        pb = p_first + v_beg  # the split's first page (logical block)
        head_off = kvh * i32_head_b
        # lane l = the page of step l
        lane_ok = lane < n_st
        pg_lane = fx.Int32(bt[reqc * i32_bt_s + pb + lane_ok.select(lane, fx.Int32(0))])
        pg_lane = lane_ok.select(pg_lane, fx.Int32(0))

        # Q^T into LDS in operand order: row t, k-step ks = 2 h + s, lane (n, g) holds dims 64 h + 16 g + 8 s .. + 8
        # of head n (the Q^T stores wait for page 0's DMA too)
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        q_lds = fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, smem.q.ptr), fx.make_layout(QSZ, 1)),
            fx.make_layout(4, 1),
        )
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        NQI = (QL * HG * 16 + 64 * NW - 1) // (64 * NW)
        q_items = []
        for k in range_constexpr(NQI):
            item = tx + k * (64 * NW)  # (row, head, 16-byte piece j of the head)
            t_i = item // (HG * 16)
            h_i = (item // 16) % HG
            jp = item % 16
            row = q0 + (t_i < qlen).select(t_i, fx.Int32(0))
            qv = fx.Vector(
                buffer_ops.buffer_load(
                    qr, ((row * i32_H + kvh * HG + h_i) * D + 8 * jp) // 2, vec_width=4, dtype=fx.Int32
                )
            )
            q_items.append((item, t_i, h_i, jp, qv))

        # DMA: wave w's piece k of a page is LDS bytes (k NW + w) KB of the slot; lane l writes 16 B unit
        # u = (k NW + w) 64 + l = token u // 16, column u % 16, read from column (u % 16) ^ swz(token)
        ring_base = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(smem.ring.ptr))))
        voffs = []
        for k in range_constexpr(NDMA):
            tok = (k * NW + w) * 4 + lane // 16
            voffs.append(tok * TOK_B + ((lane % 16) ^ _swz(tok)) * 16)

        def dma_page(step, stage):
            """The DMA of step ``step`` (past the split's pages: an empty range, zeros) into ring slot ``stage``."""
            ok = step < n_st
            stc = ok.select(step, fx.Int32(0))
            pg = fx.Int32(fx.rocdl.readlane(T.i32, pg_lane, stc))
            valid = seq_len - (pb + stc) * PAGE
            valid = (valid > PAGE).select(fx.Int32(PAGE), valid)
            valid = (valid < 0).select(fx.Int32(0), valid)
            valid = ok.select(valid, fx.Int32(0))
            valid = fx.Int32(fx.rocdl.readfirstlane(T.i32, valid))
            rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_cache + fx.Int64(pg) * fx.Int64(i32_page_b) + fx.Int64(head_off),
                num_records_bytes=valid * TOK_B,
            )
            for k in range_constexpr(NDMA):
                _dma16(ring_base + stage * PAGE_B + (k * NW + w) * DMA_B, voffs[k], rsrc)

        fx.rocdl.sched_barrier(0)  # the Q loads go out before the page lookup wait
        dma_page(fx.Int32(0), fx.Int32(0))
        for k in range_constexpr(NQI):
            item, t_i, h_i, jp, qv = q_items[k]
            ks_i = 2 * (jp // 8) + jp % 2
            g_i = (jp % 8) // 2
            if item < QL * HG * 16:
                r_q = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                r_q.store(qv)
                fx.copy(lds_atom, r_q, fx.slice(q_lds, (None, (t_i * NKS + ks_i) * 64 + g_i * 16 + h_i)))
        for s_ in range_constexpr(1, STAGES - 1):
            dma_page(fx.Int32(s_), fx.Int32(s_))

        ring16 = fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, smem.ring.ptr), fx.make_layout(RING_I, 1)),
            fx.make_layout(4, 1),
        )
        jl = w % KEY_LANES
        # K: lane (n, g), tile st, half h: token 32 j + 8 (n // 4) + 4 st + n % 4, column 4 h + g
        k_units = []
        for st in range_constexpr(2):
            tk = jl * CH + 8 * (n // 4) + 4 * st + n % 4
            for h in range_constexpr(2):
                k_units.append(tk * 16 + ((4 * h + g) ^ _swz(tk)))
        # V by transpose reads: as a supplier, lane (n, g) addresses token 8 g + n // 2, bytes 128 + 16 dt + 8 (n % 2)
        # .. + 8, so lane (n, g) receives dim 16 dt + n of tokens 8 g .. 8 g + 7: the V^T operand of O^T tile dt
        tv = jl * CH + 8 * g + n // 2
        v_bytes = [tv * TOK_B + ((8 + dt) ^ _swz(tv)) * 16 + 8 * (n % 2) for dt in range_constexpr(NDT)]

        def read_chunk(stage):
            """This lane's 16 K dwords (the global loads' order) and per O^T tile dt the 2 dwords of its V^T
            operand (8 e4m3 of tokens 8 g .. 8 g + 7) of its key lane's chunk in ring slot ``stage``."""
            raw = []
            for i in range_constexpr(4):
                r_k = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(lds_atom, fx.slice(ring16, (None, stage * (PAGE_B // 16) + k_units[i])), r_k)
                kv = Vec(r_k.load())
                raw += [kv[e] for e in range_constexpr(4)]
            sb_ = ring_base + stage * PAGE_B
            for dt in range_constexpr(NDT):
                vv = _lds_tr8(sb_ + v_bytes[dt])
                raw += [vv[0], vv[1]]
            return raw

        # per-row key bounds: row t at position ctx + t sees [ctx + t - W + 1, hi)
        klo = [ctx + (t - (W - 1)) for t in range_constexpr(QL)]
        khi = [(ctx + (t + 1)) if causal else seq_len for t in range_constexpr(QL)]
        last_t = (qlen > 0).select(qlen - 1, fx.Int32(0))
        full_lo = ctx + last_t - (W - 1)
        full_hi = (ctx + 1) if causal else seq_len
        zero4 = Vec.filled(4, 0.0, fx.Float32)

        def q_frag(t, ks):
            r_f = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(lds_atom, fx.slice(q_lds, (None, (t * NKS + ks) * 64 + lane)), r_f)
            return Vec(r_f.load())

        def widen_v(vraw):
            """V^T operands vf[dt] (bf16): dim 16 dt + n of tokens 8 g .. 8 g + 7."""
            return [_bf16x8_e4m3(vraw[2 * dt], vraw[2 * dt + 1]) for dt in range_constexpr(NDT)]

        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )

        def store_out(t, acc, inv):
            """Row t's O (``acc`` x ``inv``) in bf16: lane (n, g) holds dims 16 dt + 4 g + e of head n."""
            o_row = ((q0 + t) * i32_H + kvh * HG + n) * (D // 2) + g * 2
            for dt in range_constexpr(NDT):
                val = Vec.from_elements(
                    [_pk_bf16(acc[dt][0] * inv, acc[dt][1] * inv), _pk_bf16(acc[dt][2] * inv, acc[dt][3] * inv)],
                    fx.Int32,
                )
                buffer_ops.buffer_store(val, outr, o_row + dt * 8, mask=t < qlen)

        def store_q(t, acc, inv):
            """Row t's o_proj input: the MXFP8 codes of O rounded to bf16 (row-major, or the (16, 16) tile layout
            when ``i32_tile`` > 0) and the e8m0 scales; group b (dims 32 b .. 32 b + 31 of head n, tiles 2 b and
            2 b + 1) spans lanes g = 0 .. 3, and lane g writes group g's scale."""
            row = q0 + t
            k_row = i32_H * D
            head = kvh * HG + n
            rows16 = (i32_rows + 15) // 16 * 16
            qcr = buffer_ops.create_buffer_resource_from_addr(
                arg_qc, num_records_bytes=fx.Int64(rows16) * fx.Int64(k_row)
            )
            shuffled = i32_tile > 0
            sbs = []
            for b in range_constexpr(4):
                words = [
                    _pk_bf16(acc[2 * b + i][2 * j] * inv, acc[2 * b + i][2 * j + 1] * inv)
                    for i in range_constexpr(2)
                    for j in range_constexpr(2)
                ]
                dws, sb = _mxfp8_group8(words)
                sbs.append(sb)
                for i in range_constexpr(2):
                    k = head * D + 32 * b + 16 * i + 4 * g
                    off_t = (
                        (row // 16) * (16 * k_row) + (k // 64) * 1024 + ((k % 64) // 16) * 256 + (row % 16) * 16
                        + k % 16
                    ) // 4
                    off_l = (row * k_row + k) // 4
                    buffer_ops.buffer_store(dws[i], qcr, shuffled.select(off_t, off_l), mask=t < qlen)
            sb = (g == 0).select(sbs[0], (g == 1).select(sbs[1], (g == 2).select(sbs[2], sbs[3])))
            gi = head * 4 + g
            s_off = (
                ((((row // 32) * (k_row // 256) + gi // 8) * 4 + gi % 4) * 16 + row % 16) * 2 + (gi % 8) // 4
            ) * 2 + (row % 32) // 16
            rows32 = (i32_rows + 31) // 32 * 32
            qsr = buffer_ops.create_buffer_resource_from_addr(
                arg_qs, num_records_bytes=fx.Int64(rows32) * fx.Int64(k_row // 32)
            )
            buffer_ops.buffer_store(sb.to(fx.Int32).to(fx.Int8), qsr, s_off, mask=t < qlen)

        def store_row(t, acc, inv):
            if const_expr(QUANT):
                store_q(t, acc, inv)
            else:
                store_out(t, acc, inv)

        tid = req * i32_hkv + kvh
        pr = buffer_ops.create_buffer_resource_from_addr(arg_part, num_records_bytes=i64_part_bytes)
        o_lds = fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, smem.ring.ptr), fx.make_layout(O_I, 1)),
            fx.make_layout(4, 1),
        )
        ml = fx.recast_iter(fx.Float32, smem.q.ptr)

        def body(R0, NR, jl):
            """The key loop and the row merge of the half whose rows are R0 .. R0 + NR - 1 (static)."""

            def scores(raw, k0):
                """S^T of the half's rows for one chunk (keys k0 ..), masked at the window's edges: value 4 st + e
                of a lane is key k0 + 8 g + 4 st + e."""
                s_all = [[zero4, zero4] for _ in range_constexpr(NR)]
                for ks in range_constexpr(NKS):
                    kf = [
                        _bf16x8_e4m3(
                            raw[st * 8 + (ks // 2) * 4 + 2 * (ks % 2)],
                            raw[st * 8 + (ks // 2) * 4 + 2 * (ks % 2) + 1],
                        )
                        for st in range_constexpr(2)
                    ]
                    for r in range_constexpr(NR):
                        qv = q_frag(R0 + r, ks)
                        s_all[r] = [_mfma(kf[st], qv, s_all[r][st]) for st in range_constexpr(2)]
                edge = (k0 < full_lo) | (k0 + CH > full_hi)
                if edge:
                    key = k0 + 8 * g
                    s_all = [
                        [
                            Vec.from_elements(
                                [
                                    ((key + 4 * st + e >= klo[R0 + r]) & (key + 4 * st + e < khi[R0 + r])).select(
                                        s_all[r][st][e], neg_inf
                                    )
                                    for e in range_constexpr(4)
                                ],
                                fx.Float32,
                            )
                            for st in range_constexpr(2)
                        ]
                        for r in range_constexpr(NR)
                    ]
                return s_all

            def accumulate(s_all, vraw, m_t, l_t, th_t, mn_t, o_t):
                """A chunk's scores and V into the half's rows' state: reference max m (scaled units), sum l, the
                raw-score threshold th = (m + TAU_LOG2) / c that triggers a rescale, mn = -m (0 while m is -inf),
                and O."""
                lmax = []
                for r in range_constexpr(NR):
                    a = [s_all[r][i // 4][i % 4] for i in range_constexpr(8)]
                    mx = a[0]
                    for i in range_constexpr(1, 8):
                        mx = _maxf_nn(mx, a[i])
                    lmax.append(mx)
                need = lmax[0] > th_t[0]
                for r in range_constexpr(1, NR):
                    need = need | (lmax[r] > th_t[r])
                if fx.Int64(fx.rocdl.ballot(T.i64, need)) != fx.Int64(0):
                    m_n = [_maxf_nn(m_t[r], _xlane4(lmax[r], _maxf_nn) * c) for r in range_constexpr(NR)]
                    mu_n = [(m_n[r] == neg_inf).select(fx.Float32(0.0), m_n[r]) for r in range_constexpr(NR)]
                    alphas = [_exp2(m_t[r] - mu_n[r]) for r in range_constexpr(NR)]
                    o_t = [
                        [o_t[r][dt] * Vec.filled(4, alphas[r], fx.Float32) for dt in range_constexpr(NDT)]
                        for r in range_constexpr(NR)
                    ]
                    l_t = [l_t[r] * alphas[r] for r in range_constexpr(NR)]
                    th_t = [(m_n[r] + fx.Float32(TAU_LOG2)) * inv_c for r in range_constexpr(NR)]
                    mn_t = [-mu_n[r] for r in range_constexpr(NR)]
                    m_t = m_n
                c4 = Vec.filled(4, c, fx.Float32)
                pfs = []
                l_o = []
                for r in range_constexpr(NR):
                    mn4 = Vec.filled(4, mn_t[r], fx.Float32)
                    p = []
                    for st in range_constexpr(2):
                        x4 = Vec(fx.math.fma(s_all[r][st], c4, mn4))
                        p += [_exp2(x4[e]) for e in range_constexpr(4)]
                    q4 = Vec.from_elements(p[0:4], fx.Float32) + Vec.from_elements(p[4:8], fx.Float32)
                    q2 = Vec.from_elements([q4[0], q4[1]], fx.Float32) + Vec.from_elements([q4[2], q4[3]], fx.Float32)
                    l_o.append(l_t[r] + (q2[0] + q2[1]))
                    pfs.append(
                        Vec.from_elements([_pk_bf16(p[2 * i], p[2 * i + 1]) for i in range_constexpr(4)], fx.Int32)
                    )
                vf = widen_v(vraw)
                o_o = [
                    [_mfma(vf[dt], pfs[r], o_t[r][dt]) for dt in range_constexpr(NDT)] for r in range_constexpr(NR)
                ]
                return m_t, l_o, th_t, mn_t, o_o

            NS = 4 * NR  # scalar state: m, l, th, mn per row
            init = [neg_inf for _ in range_constexpr(NR)]
            init += [fx.Float32(0.0) for _ in range_constexpr(NR)]
            init += [neg_inf for _ in range_constexpr(NR)]
            init += [fx.Float32(0.0) for _ in range_constexpr(NR)]
            init += [zero4 for _ in range_constexpr(NR * NDT)]
            results = init
            for it, st_ in range(  # type: ignore[call-overload]
                fx.Index(0), fx.Index(n_st), fx.Index(1), init=[fx.as_ir_value(x) for x in init]
            ):
                sc = [[fx.Float32(st_[e * NR + r]) for r in range_constexpr(NR)] for e in range_constexpr(4)]
                o_t = [[Vec(st_[NS + r * NDT + dt]) for dt in range_constexpr(NDT)] for r in range_constexpr(NR)]
                v = fx.Int32(it)
                stage = v % STAGES
                # this step's page has landed (each wave waits for its own pieces, the barrier for the others')
                # and every wave is done with the previous step's slot
                fx.rocdl.s_waitcnt(vmcnt=(STAGES - 2) * NDMA, lgkmcnt=0)
                fx.rocdl.s_barrier()
                dma_page(v + (STAGES - 1), (v + (STAGES - 1)) % STAGES)
                raw = read_chunk(stage)
                k0 = (pb + v) * PAGE + jl * CH
                m_t, l_t, th_t, mn_t, o_t = accumulate(scores(raw, k0), raw[16:32], sc[0], sc[1], sc[2], sc[3], o_t)
                state = m_t + l_t + th_t + mn_t + [o for row in o_t for o in row]
                results = yield [fx.as_ir_value(x) for x in state]

            # the ring is free once every DMA has landed and every wave has left the loop
            fx.rocdl.s_waitcnt(vmcnt=0)
            _sync()
            for r in range_constexpr(NR):
                for e in range_constexpr(2):
                    ml[((w * QL + R0 + r) * 2 + e) * 64 + lane] = fx.Float32(results[e * NR + r])
            hs = R0 // HALVES[0]  # this half's merge slot

            def put_o(r, slot):
                for dt in range_constexpr(NDT):
                    r_w = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r_w.store(Vec(results[NS + r * NDT + dt]).bitcast(fx.Int32))
                    fx.copy(lds_atom, r_w, fx.slice(o_lds, (None, ((slot * KEY_LANES + jl) * NDT + dt) * 64 + lane)))

            def merge_row(t, slot):
                """Row t's key lanes merged from the half's LDS slot: (acc, max, sum)."""
                w0 = hs * KEY_LANES
                ms = [fx.Float32(ml[(((w0 + kl) * QL + t) * 2) * 64 + lane]) for kl in range_constexpr(KEY_LANES)]
                big = ms[0]
                for kl in range_constexpr(1, KEY_LANES):
                    big = _maxf_nn(big, ms[kl])
                big_raw = big
                big = (big == neg_inf).select(fx.Float32(0.0), big)
                wt = [fx.Float32(_exp2(ms[kl] - big)) for kl in range_constexpr(KEY_LANES)]
                l_tot = fx.Float32(0.0)
                for kl in range_constexpr(KEY_LANES):
                    l_w = fx.Float32(ml[(((w0 + kl) * QL + t) * 2 + 1) * 64 + lane])
                    l_tot = l_tot + l_w * wt[kl]
                l_tot = _xlane4(l_tot, lambda x, y: x + y)
                acc = [zero4 for _ in range_constexpr(NDT)]
                for kl in range_constexpr(KEY_LANES):
                    wv = Vec.filled(4, wt[kl], fx.Float32)
                    for dt in range_constexpr(NDT):
                        r_c = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        src = fx.slice(o_lds, (None, ((slot * KEY_LANES + kl) * NDT + dt) * 64 + lane))
                        fx.copy(lds_atom, src, r_c)
                        acc[dt] = acc[dt] + Vec(r_c.load()).bitcast(fx.Float32) * wv
                return acc, big_raw, l_tot

            # MR rows of each half per round, merged by key lanes 0 .. MR - 1 (every half takes the first half's
            # rounds: the barriers match)
            for k in range_constexpr((HALVES[0] + MR - 1) // MR):
                if const_expr(k > 0):
                    _sync()  # the previous round's reads are done
                for i in range_constexpr(MR):
                    if const_expr(k * MR + i < NR):
                        put_o(k * MR + i, hs * MR + i)
                _sync()
                for i in range_constexpr(MR):
                    if const_expr(k * MR + i < NR):
                        t = R0 + k * MR + i
                        if jl == i:
                            acc, big_raw, l_tot = merge_row(t, hs * MR + i)
                            if const_expr(S > 1):
                                rec = ((tid * S + split) * QL + t) * (64 * PART_F)
                                for dt in range_constexpr(NDT):
                                    buffer_ops.buffer_store(
                                        acc[dt].bitcast(fx.Int32), pr, rec + dt * 64 * 4 + lane * 4,
                                        cache_modifier=PART_SCOPE,
                                    )
                                buffer_ops.buffer_store(
                                    Vec.from_elements([_bits(big_raw), _bits(l_tot)], fx.Int32),
                                    pr,
                                    rec + NDT * 64 * 4 + lane * 2,
                                    cache_modifier=PART_SCOPE,
                                )
                            else:
                                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                                store_row(t, acc, inv)

        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()  # Q^T is in LDS
        if const_expr(NH == 1):
            body(0, HALVES[0], jl)
        else:
            if w < KEY_LANES:
                body(0, HALVES[0], jl)
            if w >= KEY_LANES:
                body(HALVES[0], HALVES[1], jl)

        if const_expr(S > 1):
            # every row's partial is stored before the ticket is taken
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx == 0:
                smem.ticket.ptr[0] = _global_atomic_add(arg_cnt + fx.Int64(tid) * 4, 1)
            fx.gpu.barrier()
            last = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.ticket.ptr[0])))
            if last == S - 1:
                for batch in range_constexpr((QL + NW - 1) // NW):
                    t_f = w + batch * NW
                    if t_f < QL:
                        recs = [((tid * S + sp) * QL + t_f) * (64 * PART_F) for sp in range_constexpr(S)]
                        mls = [
                            Vec(
                                buffer_ops.buffer_load(
                                    pr, rc + NDT * 64 * 4 + lane * 2, vec_width=2, dtype=fx.Int32,
                                    cache_modifier=PART_SCOPE,
                                )
                            )
                            for rc in recs
                        ]
                        bigs = [_as_f32(x[0]) for x in mls]
                        top = bigs[0]
                        for sp in range_constexpr(1, S):
                            top = _maxf_nn(top, bigs[sp])
                        top = (top == neg_inf).select(fx.Float32(0.0), top)
                        ws = [fx.Float32(_exp2(b - top)) for b in bigs]
                        l_all = fx.Float32(0.0)
                        for sp in range_constexpr(S):
                            l_all = l_all + _as_f32(mls[sp][1]) * ws[sp]
                        inv_all = (l_all > fx.Float32(0.0)).select(v_scale / l_all, fx.Float32(0.0))
                        acc_all = [zero4 for _ in range_constexpr(NDT)]
                        for sp in range_constexpr(S):
                            wv = Vec.filled(4, ws[sp], fx.Float32)
                            for dt in range_constexpr(NDT):
                                part = Vec(
                                    buffer_ops.buffer_load(
                                        pr, recs[sp] + dt * 64 * 4 + lane * 4, vec_width=4, dtype=fx.Int32,
                                        cache_modifier=PART_SCOPE,
                                    )
                                ).bitcast(fx.Float32)
                                acc_all[dt] = acc_all[dt] + part * wv
                        store_row(t_f, acc_all, inv_all)
            if (last == S - 1) & (tx == 0):
                _global_i32_ptr(arg_cnt + fx.Int64(tid) * 4)[0] = fx.Int32(0)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_qsl: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_reqs: fx.Int32,
        i32_rows: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        i32_head_b: fx.Int32,
        f32_c: fx.Float32,
        arg_qc: fx.Int64,
        arg_qs: fx.Int64,
        i32_tile: fx.Int32,
        i32_grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_reqs, i32_bt_s, i32_page_b, i32_head_b, i32_H, i32_hkv, arg_seq, arg_qsl, arg_bt, arg_q, arg_cache,
            arg_out, arg_ks, arg_vs, arg_part, arg_cnt, i64_part_bytes, i32_rows, f32_c, arg_qc, arg_qs, i32_tile,
        ).launch(grid=(i32_grid_x, i32_hkv, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = {**PRELOAD_HINTS, "waves_per_eu": 2 if NW == 8 else 1}
    return launch
