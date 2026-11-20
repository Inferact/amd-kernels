# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP2 QKV MXFP8 GEMM (N 4992, K 6144) with the split-K reduce + norm / RoPE + NVFP4 inserts in its tail, one
launch (gfx950). It builds for every deferred split row (1-48 rows); the host serves 1-16, where it wins.

GEMM: ``gemm/mxfp8_common``'s decode kernel on the deferred row's own tile (``gemm/mxfp8_n4992_k6144``
``DEFERRED_ROWS``: RT 16-row tiles, all rows in one block, x NI 16-column tiles, KW K-waves, KS K splits over
workgroups), so every output is the same KS * KW sequential MFMA chains over equal K ranges, the KW chains of a split
summed in wave order (parked in LDS, the decode kernel's park-all sums). Each split stores its fp32 slab
``[KS, m, 4992]`` at agent scope (sc1), and the tail sums the slabs ``((0 + s0) + s1) + ...``, rounds to bf16 and runs
``splitk_reduce_kernel_q64_kv4_iq4``'s arithmetic: the deferred chain's bytes.

Hand-off: a 48 / 64-column tile overlaps one or two heads; a head's counter collects its KS x (3 or 4) tiles. Once a
workgroup's slab stores are acknowledged (``vmcnt(0)``, a barrier), wave 0 adds to the counter of each head its tile
covers: the split-0 first tile of the head adds ``EPOCH - count + 1``, every other tile 1, so only the last arrival
sees the counter reach the epoch (epoch counters: each call adds ``EPOCH`` to every counter it uses; no reset, i32 wrap
included). The last arrival runs the head for every row. No workgroup waits for another.

Tail: 8 waves. The tail waves stage every slab-independent input in LDS while the K-waves run the GEMM (cos / sin
rows, main and index slots of the rows and 8 tokens past them, the norm weights, the exact NVFP4 scale reciprocals of
the K, V and index-K scales: ``_quant4_lds``); after a barrier each wave reads its token pairs' inputs and the V octets'
old bytes before the hand-off, so after it only the slab loads remain. Wave w takes pairs w, w + 8, ...; a V head's
waves quantize their rows into LDS, and after a barrier the first token of each octet run packs the run from LDS
(``packing.py``).

Head columns (of 128): q 0-31, k 32-33, v 34-35, index-q 36-37, index-k 38.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import (
    PRELOAD_HINTS,
    W_CACHE_MOD,
    _lookahead,
    _mfma,
    _pack8,
    _pin_accumulators,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128.kernel import (
    BLK,
    F448,
    F448N,
    RCP6,
    D,
    _bf16x4,
    _e2m1x2,
    _e4m3_byte,
    _e4m3_value,
    _f32,
    _gptr,
    _iq_fp8,
    _k_scale_off,
    _maxnum,
    _minnum,
    _norm_rope4,
    _pk_bf16,
    _xmax,
)
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import BFloat16, Float32, T
from flydsl.expr.typing import Vector as Vec

from .packing import _v_expand, _v_expanded_words

N, K = 4992, 6144
NQ, NKV, NIQ, ROT = 32, 2, 2, 64
HEADS = N // D  # 39
HK, HV, HIQ, HIK = NQ, NQ + NKV, NQ + 2 * NKV, NQ + 2 * NKV + NIQ  # first k / v / index-q head, the index-k head
NWAVE = 8
EPOCH = 16  # a power of 2 >= every head's contributor count (KS 3 x 4 tiles = 12)
SC1 = 0x10  # agent-scope (cross-XCD) stores / loads of the slabs
HALF = ROT // 2
SIDE = NKV * BLK * 72
CNT_STRIDE = 32  # one counter per 128-byte line: atomics of neighbouring heads do not share a line
COUNTERS = HEADS * CNT_STRIDE
TABLE = 260  # dwords of one reciprocal table (256 codes + 1 / gs)


def _global_atomic_add(addr, value):
    """``old = *addr; *addr += value`` on a global i32 (agent scope, relaxed)."""
    ptr = fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4), fx.Int64(addr))
    return fx.Int32(_llvm.AtomicRMWOp(_llvm.AtomicBinOp.add, ptr.llvm_ptr, fx.Int32(value).ir_value(),
                                      _llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4).result)


def _lds_ptr(addr):
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int64(addr)))


def _lds_i32(addr):
    return fx.Int32(_llvm.load(T.i32, _lds_ptr(addr), alignment=4))


def _lds_vec(addr, n):
    return Vec(_llvm.load(T.vec(n, T.i32), _lds_ptr(addr), alignment=4 * n))


def _lds_store(addr, v, align):
    _llvm.StoreOp(fx.as_ir_value(v), _lds_ptr(addr), alignment=align)


def _quant4_lds(xs, table):
    """``_quant4`` with its divisions read from the LDS table at byte address ``table``: entries 0-255 hold the same
    correctly rounded 1 / (gs * e4m3(byte)) (0 for a zero scale), entry 256 the same 1 / gs, so the codes and scale
    bytes are ``_quant4``'s."""
    a = [fx.math.absf(x) for x in xs]
    m = _maxnum(_maxnum(a[0], a[1]), _maxnum(a[2], a[3]))
    m = _xmax(_xmax(m, 1), 2)
    inv_gs = _lds_i32(table + 256 * 4).bitcast(fx.Float32)
    sf = _minnum((m * _f32(RCP6)) * inv_gs, _f32(F448))
    byte = _e4m3_byte(sf)
    quant = _lds_i32(table + byte * 4).bitcast(fx.Float32)
    y = [x * quant for x in xs]
    return _e2m1x2(y[2], y[3], _e2m1x2(y[0], y[1], fx.Int32(0), 0), 1), byte


@launcher
def compile_qkv_fused(RT: int, NI: int, KW: int, KS: int, LA: tuple | None = None):
    """The one-launch chain on a deferred GEMM row's tile; every row in one block (m <= 16 * RT).

    Args:
        RT, NI, KW, KS: 16-row tiles, 16-column tiles, K-waves and K splits of the deferred row (its bytes).
        LA: (W steps, A blocks) in flight ahead of the MFMAs (default: the deepest that fits).
    """
    KSTEPS = K // 128
    assert KSTEPS % (KS * KW) == 0 and (KSTEPS // (KS * KW)) % 2 == 0 and KW < NWAVE
    SW = KSTEPS // (KS * KW)
    BM = 16 * RT
    C = 16 * NI
    NB = N // C
    assert N % C == 0 and C <= D
    SCB = K // 256
    RP = max(1, RT // 2)
    if RT > 1 and RT % 2 == 1:
        RP = (RT + 1) // 2
    CP = NI if NI % 2 else NI // 2
    LA_W, LA_A = LA or _lookahead(SW, RT, NI, RP, CP)
    NA = SW * RT
    TW = NWAVE - KW
    NT = RT * NI  # accumulator tiles of a K-wave
    PPW = BM // 2 // NWAVE  # token pairs per wave
    PARK = KW * NT * 1024 if KW > 1 else 0
    CS_OFF = PARK  # [BM, 32] dwords: cos | sin of each row's position
    SL_OFF = CS_OFF + BM * 128  # main slots of tokens 0 .. BM + 7 (-1 past the batch)
    ISL_OFF = SL_OFF + (BM + 8) * 4
    W_OFF = ISL_OFF + BM * 4  # norm weights: q, k, index-q, index-k
    RCP_OFF = W_OFF + 4 * 64 * 4  # reciprocal tables: K, V, index-K
    VROW_OFF = RCP_OFF + 3 * TABLE * 4  # per head slot, BM + 8 V rows quantized: (expanded codes, scale byte)
    VROW_BYTES = (BM + 8) * 32 * 8
    BUF_BYTES = VROW_OFF + 2 * VROW_BYTES
    assert BUF_BYTES <= 160 * 1024, BUF_BYTES
    NSL = -(-(BM + 8) // 64)
    name = (f"mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128_rt{RT}_ni{NI}_kw{KW}_ks{KS}"
            f"_la{LA_W}x{LA_A}")

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, BUF_BYTES, 16]
        flags: fx.Array[fx.Int32, 4]  # the tile's two heads: the last arrival is here

    @flyc.kernel(name=name, known_block_size=[64 * NWAVE, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_as: fx.Int64,
        arg_a: fx.Int64,
        arg_slab: fx.Int64,  # [KS, m, N] fp32 partials (workspace)
        arg_cnt: fx.Int64,  # [COUNTERS] i32 head counters, a multiple of EPOCH between calls
        i32_cs_rows: fx.Int32,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_islot: fx.Int64,
        arg_cs: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_q: fx.Int64,
        arg_iq: fx.Int64,
        arg_kv: fx.Int64,
        arg_idx: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_iq_st: fx.Int32,
        i32_eps: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        kz, nb = bx // NB, bx % NB
        slab_bytes = i32_m * (N * 4 * KS)
        slab = buffer_ops.create_buffer_resource_from_addr(arg_slab, num_records_bytes=slab_bytes)
        lds = fx.Int32(fx.ptrtoint(smem.buf.ptr))
        hl = lane % 32
        hi16 = (hl & 16) != 0
        dim0 = hl * 4
        eps = i32_eps.bitcast(fx.Float32)
        first = dim0 < HALF
        in_rope = dim0 < ROT
        ib = first.select(dim0, dim0 - HALF)
        h0 = (nb * C) // D
        heads = [h0, h0 + 1]
        has1 = (nb * C + (C - 1)) // D > h0  # the tile reaches a second head
        is_vs = [(h >= HV) & (h < HIQ) for h in heads]

        def gemm_tile(kwave):
            """The decode kernel's K-wave ``kwave``: its RT x NI accumulators (fp32, pinned)."""
            k_base = kwave * (SW * 2048)
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_a, num_records_bytes=BM * K)
            w_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K)
            as_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_as, num_records_bytes=a_scale_bytes)
            ws_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_ws, num_records_bytes=(N // 32) * SCB * 256)
            lane_off = q16 * 256 + l16 * 16
            n16 = [nb * NI + ni for ni in range_constexpr(NI)]
            w_dw = [(n16[ni] * (16 * K) + k_base + lane_off) // 4 for ni in range_constexpr(NI)]
            a_dw = [(rt * (16 * K) + k_base + lane_off) // 4 for rt in range_constexpr(RT)]
            sc_lane = q16 * 16 + l16
            sc_k = kwave * (SW // 2 * 64)
            ws_dw = [((n16[cp if NI % 2 else 2 * cp] // 2) * SCB) * 64 + sc_k + sc_lane for cp in range_constexpr(CP)]
            as_dw = [(rp * SCB) * 64 + sc_k + sc_lane for rp in range_constexpr(RP)]
            w_shift = [(n16[ni] % 2) * 8 for ni in range_constexpr(NI)] if NI % 2 else None

            def load_pair(rsrc, dw, s, nt):
                kw = dict(vec_width=4, dtype=fx.Int32)
                if const_expr(nt):
                    kw["cache_modifier"] = W_CACHE_MOD
                return _pack8(buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048, **kw),
                              buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048 + 1024, **kw))

            def load_scales(rsrc, dws, s, shift):
                vals = [fx.Int32(buffer_ops.buffer_load(rsrc, dw, vec_width=1, dtype=fx.Int32,
                                                        soffset_bytes=(s // 2) * 256)) for dw in dws]
                if const_expr(shift is None):
                    return vals
                return [v.shrui(sh) for v, sh in zip(vals, shift)]

            def load_w_step(s):
                bb = [load_pair(w_rsrc, w_dw[ni], s, True) for ni in range_constexpr(NI)]
                if const_expr(s % 2 == 0):
                    return bb, (load_scales(ws_rsrc, ws_dw, s, w_shift), load_scales(as_rsrc, as_dw, s, None))
                return bb, None

            def load_a(t):
                rt, s = t % RT, t // RT
                return load_pair(a_rsrc, a_dw[rt], s, False)

            acc = [[None] * NI for _ in range(RT)]
            w_q = [None] * SW
            a_q = [None] * NA
            for s in range_constexpr(LA_W):
                w_q[s] = load_w_step(s)
            for t in range_constexpr(LA_A):
                a_q[t] = load_a(t)
            sw = sa = None
            for s in range_constexpr(SW):
                if const_expr(s + LA_W < SW):
                    w_q[s + LA_W] = load_w_step(s + LA_W)
                bb, sc = w_q[s]
                if const_expr(s % 2 == 0):
                    sw, sa = sc
                for rt in range_constexpr(RT):
                    t = s * RT + rt
                    if const_expr(t + LA_A < NA):
                        a_q[t + LA_A] = load_a(t + LA_A)
                    a_op = a_q[t]
                    a_q[t] = None
                    for ni in range_constexpr(NI):
                        acc[rt][ni] = _mfma(bb[ni], a_op, acc[rt][ni], sw[ni if NI % 2 else ni // 2], sa[rt // 2],
                                            0 if NI % 2 else ni % 2, rt % 2, s % 2)
                w_q[s] = None
            return _pin_accumulators([acc[rt][ni] for rt in range(RT) for ni in range(NI)])

        def store_tile(i, v):
            """Accumulator tile ``i`` of this split's slab, agent scope."""
            row = (i // NI) * 16 + l16
            col = (nb * NI + i % NI) * 16 + q16 * 4
            buffer_ops.buffer_store(v.bitcast(fx.Int32), slab, ((kz * i32_m + row) * N + col) * 4,
                                    offset_is_bytes=True, mask=row < i32_m, cache_modifier=SC1)

        def preload(w0):
            """The tail's slab-independent inputs into LDS (tail wave ``w0`` of TW)."""
            pos_r = buffer_ops.create_buffer_resource_from_addr(arg_pos, num_records_bytes=fx.Int64(i32_m) * 8)
            sl_r = buffer_ops.create_buffer_resource_from_addr(arg_slot, num_records_bytes=fx.Int64(i32_m) * 8)
            isl_r = buffer_ops.create_buffer_resource_from_addr(arg_islot, num_records_bytes=fx.Int64(i32_m) * 8)
            cs_r = buffer_ops.create_buffer_resource_from_addr(arg_cs,
                                                               num_records_bytes=fx.Int64(i32_cs_rows) * (ROT * 2))
            toks = []  # 8 rows of cos / sin per wave load
            for k in range_constexpr(-(-(BM // 8) // TW)):
                c = w0 + k * TW
                toks.append((c < BM // 8).select(c, fx.Int32(BM // 8 - 1)) * 8 + lane // 8)
            poss = [fx.Int32(buffer_ops.buffer_load(pos_r, t * 2, vec_width=1, dtype=fx.Int32, mask=t < i32_m))
                    for t in toks]
            entries = []
            for k in range_constexpr(-(-NSL // TW)):
                e = w0 + k * TW
                entries.append((e < NSL).select(e, fx.Int32(NSL - 1)) * 64 + lane)
            live = [e < i32_m for e in entries]
            slots = [fx.Int32(buffer_ops.buffer_load(sl_r, e * 2, vec_width=1, dtype=fx.Int32, mask=ok))
                     for e, ok in zip(entries, live)]
            islots = [fx.Int32(buffer_ops.buffer_load(isl_r, e * 2, vec_width=1, dtype=fx.Int32, mask=ok))
                      for e, ok in zip(entries, live)]
            ws = []
            for a in (arg_qw, arg_kw, arg_iqw, arg_ikw):
                wr = buffer_ops.create_buffer_resource_from_addr(a, num_records_bytes=D * 2)
                ws.append(fx.Int32(buffer_ops.buffer_load(wr, lane, vec_width=1, dtype=fx.Int32)))
            ks = fx.Float32(_gptr(arg_ks, T.f32, 4)[0])
            vs = fx.Float32(_gptr(arg_vs, T.f32, 4)[0])
            css = [Vec(buffer_ops.buffer_load(cs_r, p * (ROT // 2) + (lane % 8) * 4, vec_width=4, dtype=fx.Int32))
                   for p in poss]
            for k, gs in enumerate((ks, vs, fx.Float32(1.0))):
                table = lds + RCP_OFF + k * (TABLE * 4)
                for i in range_constexpr(4):
                    sv = _e4m3_value(lane + i * 64)
                    quant = (sv != 0.0).select(fx.Float32(1.0) / (gs * sv), fx.Float32(0.0))
                    _lds_store(table + (lane + i * 64) * 4, quant.bitcast(fx.Int32), 4)
                _lds_store(table + 256 * 4, (fx.Float32(1.0) / gs).bitcast(fx.Int32), 4)
            for e, ok, v, iv in zip(entries, live, slots, islots):
                if e < BM + 8:
                    _lds_store(lds + SL_OFF + e * 4, ok.select(v, fx.Int32(-1)), 4)
                if e < BM:
                    _lds_store(lds + ISL_OFF + e * 4, ok.select(iv, fx.Int32(-1)), 4)
            for k in range_constexpr(4):
                _lds_store(lds + W_OFF + (k * 64 + lane) * 4, ws[k], 4)
            for t, v in zip(toks, css):
                _lds_store(lds + CS_OFF + t * 128 + (lane % 8) * 16, v, 16)

        if wave < KW:
            flat = gemm_tile(kz * KW + wave)
            if const_expr(KW == 1):
                for i in range_constexpr(NT):
                    store_tile(i, Vec(flat[i]))
            else:
                for i in range_constexpr(NT):
                    _lds_store(lds + (wave * NT + i) * 1024 + lane * 16, Vec(flat[i]).bitcast(fx.Int32), 16)
        else:
            preload(wave - KW)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        if const_expr(KW > 1):
            # the K-waves' partials summed in wave order (the decode kernel's park-all sums), tiles over all waves
            for j in range_constexpr(-(-NT // NWAVE)):
                i = wave + j * NWAVE
                if i < NT:
                    v = None
                    for w in range_constexpr(KW):
                        p = _lds_vec(lds + (w * NT + i) * 1024 + lane * 16, 4).bitcast(fx.Float32)
                        v = p if v is None else v + p
                    store_tile(i, v)

        # this wave's token pairs: the V run each one starts and its octet's old bytes, cos / sin, weights
        pairs = []
        for k in range_constexpr(PPW):
            tok = (wave + k * NWAVE) * 2 + lane // 32
            live = tok < i32_m
            slot = _lds_i32(lds + SL_OFF + tok * 4)
            prev = (tok > 0).select(_lds_i32(lds + SL_OFF + (tok - 1) * 4), fx.Int32(-1))
            near_t = tok - 7 + hl
            near = ((hl < 16) & (near_t >= 0)).select(_lds_i32(lds + SL_OFF + near_t * 4), fx.Int32(-1))
            base_o = slot - slot % 8
            in_oct = (near >= base_o) & (near < base_o + 8)
            votes = fx.Int64(fx.rocdl.ballot(T.i64, in_oct))
            mine = (votes >> fx.Int64((lane // 32) * 32)).to(fx.Int32) & 0xFFFF
            n_run = fx.Int32(fx.math.ctpop(mine))
            go = live & (slot >= 0) & ((prev < 0) | ((prev >> 3) != (slot >> 3)))
            wslot = go.select(slot, fx.Int32(0))
            tp_ = wslot & (BLK - 1)
            olds = []
            for j in range_constexpr(2):
                vh = is_vs[j].select(heads[j] - HV, fx.Int32(0))
                v_side = arg_kv + fx.Int64(wslot >> 7) * i64_kv_st + SIDE
                v_data = v_side + fx.Int64(vh * (BLK * 64) + (tp_ // 8) * (D * 4))
                v_sf = v_side + fx.Int64(NKV * BLK * 64 + vh * (BLK * 8) + tp_ - tp_ % 8)
                dp = _gptr(v_data + fx.Int64(dim0 * 4), T.i32, 4)
                sp = _gptr(v_sf + fx.Int64((hl // 4) * BLK), T.i32, 4)
                old = [fx.Int32(dp[i]) for i in range_constexpr(4)] + [fx.Int32(sp[i]) for i in range_constexpr(2)]
                olds.append((dp, sp, old))
            cs_row = lds + CS_OFF + tok * 128 + (ib // 2) * 4
            cos = _bf16x4(_lds_vec(cs_row, 2))
            sin = _bf16x4(_lds_vec(cs_row + 64, 2))
            pairs.append((tok, live, n_run, go, tp_, olds, cos, sin))
        wts = []
        for h in heads:
            ty = (h < HK).select(fx.Int32(0), ((h >= HK) & (h < HV)).select(
                fx.Int32(1), (h == HIK).select(fx.Int32(3), fx.Int32(2))))
            wts.append(_bf16x4(_lds_vec(lds + W_OFF + (ty * 64 + hl * 2) * 4, 2)))

        # every slab store is acknowledged before the counters move
        fx.rocdl.s_waitcnt(vmcnt=0)
        if tx < 4:
            smem.flags.ptr[tx] = fx.Int32(0)
        fx.gpu.barrier()
        if wave == 0:
            h = h0 + lane
            t0 = (h * D) // C
            count = ((h * D + (D - 1)) // C - t0 + 1) * KS
            inc = ((kz == 0) & (nb == t0)).select(EPOCH + 1 - count, fx.Int32(1))
            if (lane == 0) | ((lane == 1) & has1):
                old = _global_atomic_add(arg_cnt + fx.Int64(h) * (4 * CNT_STRIDE), inc)
                if ((old & (EPOCH - 1)) + inc) == EPOCH:
                    smem.flags.ptr[lane] = fx.Int32(1)
        fx.gpu.barrier()
        done = [fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.flags.ptr[j]))) != 0 for j in range(2)]

        def row4(tok, h, live):
            """This lane's 4 slab vectors at (token, head)."""
            return [Vec(buffer_ops.buffer_load(slab, (k * i32_m + tok) * N + h * D + dim0, vec_width=4,
                                               dtype=Float32, mask=live, cache_modifier=SC1))
                    for k in range_constexpr(KS)]

        def sum4(raw):
            """The slabs summed in the reduce's order, rounded to bf16, as 4 fp32."""
            acc = Vec.filled(4, 0.0, Float32)
            for k in range_constexpr(KS):
                acc = acc + raw[k]
            e = acc.to(BFloat16).to(Float32)
            return [e[i] for i in range_constexpr(4)]

        raws = [[row4(tok, heads[j], live & done[j]) for tok, live, *_ in pairs] for j in range(2)]
        q_r = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=fx.Int64(i32_m) * (NQ * D * 2))
        iq_r = buffer_ops.create_buffer_resource_from_addr(arg_iq,
                                                           num_records_bytes=fx.Int64(i32_m) * fx.Int64(i32_iq_st))
        vtab = lds + RCP_OFF + TABLE * 4
        for j in range_constexpr(2):
            h = heads[j]
            vrow = lds + VROW_OFF + j * VROW_BYTES
            if done[j] & is_vs[j]:
                for (tok, *_), raw in zip(pairs, raws[j]):
                    nib, sf = _quant4_lds(sum4(raw), vtab)
                    _lds_store(vrow + (tok * 32 + hl) * 8,
                               Vec.from_elements([_v_expand(fx.Int32(nib)), fx.Int32(sf)], fx.Int32), 8)
            if done[j] & ~is_vs[j]:
                is_q = h < HK
                is_k = (h >= HK) & (h < HV)
                is_ik = h == HIK
                for (tok, live, n_run, go, tp_, olds, cos, sin), raw in zip(pairs, raws[j]):
                    out = _norm_rope4(sum4(raw), (cos, sin), wts[j], eps, hi16, first, in_rope, HALF // 4)
                    qv = Vec.from_elements([_pk_bf16(out[0], out[1]), _pk_bf16(out[2], out[3])], fx.Int32)
                    if is_q:
                        buffer_ops.buffer_store(qv, q_r, tok * (NQ * D // 2) + h * (D // 2) + hl * 2, mask=live,
                                                cache_modifier=2)
                    if (h >= HIQ) & (h < HIK):
                        f = [_minnum(_maxnum(xv, _f32(F448N)), _f32(F448)) for xv in _bf16x4(qv)]
                        word = _iq_fp8(f[2], f[3], _iq_fp8(f[0], f[1], fx.Int32(0), False), True)
                        buffer_ops.buffer_store(word, iq_r, tok * i32_iq_st + (h - HIQ) * D + dim0, mask=live,
                                                offset_is_bytes=True, cache_modifier=2)
                    if is_k | is_ik:
                        kslot = _lds_i32(is_k.select(lds + SL_OFF, lds + ISL_OFF) + tok * 4)
                        table = lds + RCP_OFF + is_k.select(fx.Int32(0), fx.Int32(2 * TABLE * 4))
                        packed, ksf = _quant4_lds(_bf16x4(qv), table)
                        kh = h - HK
                        grp = hl // 4
                        ktp = kslot & (BLK - 1)
                        page = is_k.select(arg_kv + fx.Int64(kslot >> 7) * i64_kv_st,
                                           arg_idx + fx.Int64(kslot >> 7) * i64_idx_st)
                        d_off = is_k.select(kh * (BLK * 64), fx.Int32(0)) + ktp * 64 + hl * 2
                        s_off = is_k.select(NKV * BLK * 64 + kh * (BLK * 8) + _k_scale_off(ktp, grp),
                                            BLK * 64 + ktp * 8 + grp)
                        d_ptr = _gptr(page + fx.Int64(d_off), T.i16, 2)
                        s_ptr = _gptr(page + fx.Int64(s_off), T.i8, 1)
                        if live & (kslot >= 0):
                            d_ptr[0] = packed.to(fx.Int16)
                            if hl % 4 == 0:
                                s_ptr[0] = ksf.to(fx.Int8)
        if (done[0] & is_vs[0]) | (done[1] & is_vs[1]):
            fx.gpu.barrier()
            for j in range_constexpr(2):
                vrow = lds + VROW_OFF + j * VROW_BYTES
                for tok, live, n_run, go, tp_, olds, cos, sin in pairs:
                    dp, sp, oldv = olds[j]
                    for run_rows, take in ((4, go & done[j] & is_vs[j] & (n_run <= 4)),
                                           (8, go & done[j] & is_vs[j] & (n_run > 4))):
                        if take:
                            nibbles = []
                            scales = fx.Uint64(0)
                            for r in range_constexpr(run_rows):
                                got = _lds_vec(vrow + ((tok + r) * 32 + hl) * 8, 2)
                                ok = (tok + r < i32_m) & (r < n_run)
                                nibbles.append(ok.select(got[0], fx.Int32(0)))
                                scales = scales | (ok.select(got[1], fx.Int32(0)).to(fx.Uint64) << (8 * r))
                            words = _v_expanded_words(nibbles)
                            p0 = tp_ % 8
                            mask = (fx.Uint32(0xFFFFFFFF) >> ((8 - n_run) * 4)) << (p0 * 4)
                            for i in range_constexpr(4):
                                dp[i] = (words[i] << (p0 * 4)) | (oldv[i] & ~mask.to(fx.Int32))
                            smask = (fx.Uint64(0xFFFFFFFFFFFFFFFF) >> ((8 - n_run) * 8)) << (p0 * 8)
                            scales = scales << (p0 * 8)
                            sc0 = scales.to(fx.Int32) | (oldv[4] & ~smask.to(fx.Int32))
                            sc1 = (scales >> 32).to(fx.Int32) | (oldv[5] & ~(smask >> 32).to(fx.Int32))
                            if hl % 4 == 0:
                                sp[0] = sc0
                                sp[1] = sc1

    @flyc.jit
    def launch(
        arg_a: fx.Int64,
        arg_w: fx.Int64,
        arg_as: fx.Int64,
        arg_ws: fx.Int64,
        arg_slab: fx.Int64,
        arg_cnt: fx.Int64,
        arg_q: fx.Int64,
        arg_iq: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_cs: fx.Int64,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_islot: fx.Int64,
        arg_kv: fx.Int64,
        arg_idx: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        i32_iq_st: fx.Int32,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_cs_rows: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, a_scale_bytes, arg_w, arg_ws, arg_as, arg_a, arg_slab, arg_cnt, i32_cs_rows, arg_pos, arg_slot,
            arg_islot, arg_cs, arg_qw, arg_kw, arg_iqw, arg_ikw, arg_q, arg_iq, arg_kv, arg_idx, arg_ks, arg_vs,
            i64_kv_st, i64_idx_st, i32_iq_st, i32_eps,
        ).launch(grid=(KS * NB, 1, 1), block=(64 * NWAVE, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
