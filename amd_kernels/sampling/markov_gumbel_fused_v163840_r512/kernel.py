# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The k Markov draft steps of a Kimi-K3.1 DSpark draft in one persistent launch, FlyDSL (gfx950).

256 workgroups of four waves, one per CU, all resident; the steps run inside the kernel. Per step, ``MT`` row tiles of 16
(<= 48 rows): the previous tokens (in LDS) select the ``w1`` rows, which sit in LDS (row pitch 1040 B) and in registers as the
MFMA's second operand; each wave streams ten 16-column tiles of ``w2`` (the first operand, 16x16x32 bf16: a lane owns four
consecutive columns of one row), one tile ahead, and per tile rounds the fp32 sum to bf16 (the GEMV output), adds the base
logits with one bf16 rounding, writes the logits back into the wave's LDS slab, adds vLLM's murmur3 Gumbel noise and folds
the value into the lane's maximum (first column on ties). The slab (base logits of the wave's 160 columns, all rows) is the
next step's and arrives by LDS DMA while the step's reduction runs; it goes to the draft logits cache in two 16-byte
coalesced stores a half. A XCC owns a contiguous eighth of the columns (its workgroups take a rank from a per-XCC counter);
each wave keeps its tiles 1 and 2 (7 / 5 tiles of a workgroup at 16 / 32 rows) in LDS from step 0 on, so steps 1.. read them
from LDS. A workgroup reduces its columns per row and publishes one (value, column, tag) record per row with agent-scope
stores; workgroup r < n waits for the 256 records of row r (a tag is generation + step + 1) and publishes the row's result;
every workgroup polls the n results into LDS as the next step's tokens (workgroup 0 also writes ``draft_tokens``). The
temperature picks the loop body of the whole wave (uniform branch): no row noised, no division (every row T in {0, 1}), or
the general one. The 256 workgroups must be co-resident; a wait that never ends is abandoned after ``SPIN_LIMIT`` polls.
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.common import buffer_ops

from .reference import RANK, V

# _log_normal, _mix, _fmix, _noise, _xor_lane and _comb follow sampling/gumbel_partials_bias_bf16_v200064/kernel.py
F, I = fx.Float32, fx.Int32
NEG_INF = float("-inf")
SC1 = 0x10  # agent-scope (cross-XCD) stores / loads
REC_DW = 4  # dwords of a candidate record: value, column, tag, pad
RES_OFF = 16  # dwords of the scratch: per-row results, XCC rank counters, candidate records
CNT_OFF = 1024
REC_OFF = 2048
SCRATCH_DW = REC_OFF + 48 * 256 * REC_DW
NWG = 256
NW = 4
SPIN_LIMIT = 1 << 22  # polls (about 3 s) before a wait gives up: wrong draft tokens (verified downstream) rather than a hung GPU
ERR_OFF = 8  # dword of the scratch that a timed-out wait sets
TILES = V // 16 // (NWG * NW)  # 16-column tiles of a wave
NT = 2  # non-temporal: the base-logit read and the cache write must not push ``w2`` lines out of L2 / the infinity cache
RES_TILES = {1: (2, 2, 2, 1), 2: (2, 1, 1, 1), 3: (0, 0, 0, 0)}  # tiles per wave kept in LDS across the steps, by row tiles
AFR_LDS = {1: True, 2: False, 3: True}  # the MFMA operand of the w1 rows read from LDS a k step ahead, not held in registers
RES_LO = 1  # their first tile index
TS = bool(int(__import__("os").environ.get("AMD_KERNELS_MARKOV_TS", "0")))  # per-workgroup timestamps into the scratch
TS_OFF = SCRATCH_DW
WG_COLS = V // NWG
ER = RANK // 2 + 4  # LDS row pitch in dwords
KSTEPS = RANK // 32
_C = [float(struct.unpack("<f", struct.pack("<f", 1.0 / k))[0]) for k in range(1, 9)]


def _fbits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


_LN2_HI, _LN2_LO = _fbits(0x3F317217), _fbits(0x3377D1CF)


def _i32(x):
    return x - (1 << 32) if x >= (1 << 31) else x


def _log_normal(x):
    """Triton's ``tl.log`` (LLVM's llvm.log.f32 expansion on gfx950) for finite x >= 2^-126."""
    y = fx.math.log2(x, fastmath=fx.FastMathFlags.afn)
    r0 = y * F(_LN2_HI)
    f0 = fx.math.fma(y, F(_LN2_HI), r0 * F(-1.0))
    f1 = fx.math.fma(y, F(_LN2_LO), f0)
    return fx.math.fma(y, F(_LN2_HI), f1)


def _lshr(x, s):
    return (x >> s) & ((1 << (32 - s)) - 1)


def _rotl(x, s):
    return (x << s) | _lshr(x, 32 - s)


def _mix(h, key, key_c1=None):
    key = key * _i32(0xCC9E2D51) if key_c1 is None else key_c1
    key = _rotl(key, 15)
    key = key * _i32(0x1B873593)
    h = h ^ key
    h = _rotl(h, 13)
    return (h << 2) + h + _i32(0xE6546B64)


def _fmix(h):
    h = h ^ _lshr(h, 16)
    h = h * _i32(0x85EBCA6B)
    h = h ^ _lshr(h, 13)
    h = h * _i32(0xC2B2AE35)
    return h ^ _lshr(h, 16)


def _noise(h0, col_c1):
    """fp32 Gumbel noise of the vocab id whose product with 0xCC9E2D51 is ``col_c1``."""
    r = _fmix(_mix(h0, None, col_c1) ^ 16)
    u = _lshr(r, 16).to(F) * F(1.52587890625e-05) + ((r & 0xFFFF).to(F) + F(0.5)) * F(2.3283064365386963e-10)
    p = F(_C[7])
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = fx.math.fma(u, p, F(_C[k - 1]))
    series = (u * F(-1.0)) * p
    one_m = F(1.0) - u
    direct = _log_normal((one_m > F(5.960464477539063e-08)).select(one_m, F(5.960464477539063e-08)))
    l1 = (u < F(0.25)).select(series, direct)
    return _log_normal(l1 * F(-1.0)) * F(-1.0)


def _dpp(src, ctrl, old=None, bank_mask=0xF):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return I(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(src), ctrl, 0xF, bank_mask, full))


def _swap(op, v):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False)
    return I(_llvm.extractvalue(T.i32, res, [0])), I(_llvm.extractvalue(T.i32, res, [1]))


def _xor_lane(v, m, upper):
    """Lane ``l ^ m`` of the wave as an i32; ``upper``: this lane's bit m is set (m = 16, 32)."""
    if m == 1:
        return _dpp(v, 0xB1)
    if m == 2:
        return _dpp(v, 0x4E)
    if m == 4:
        return _dpp(v, 0x104, old=_dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return _dpp(v, 0x128)
    a, b = _swap(_rocdl_ops.permlane16_swap if m == 16 else _rocdl_ops.permlane32_swap, v)
    return upper.select(a, b)


def _comb(v1, i1, v2, i2):
    """Keep (v1, i1) iff v1 > v2 or (v1 == v2 and i1 < i2)."""
    keep = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return keep.select(v1, v2), keep.select(i1, i2)


def _atomic_add(addr, v):
    """Returning scalar global add (one active lane): the old value."""
    low = I(fx.rocdl.readfirstlane(T.i32, fx.Int64(addr).to(fx.Int32)))
    high = I(fx.rocdl.readfirstlane(T.i32, (fx.Int64(addr) >> 32).to(fx.Int32)))
    pointer = (low.to(fx.Uint32).to(fx.Uint64) | (high.to(fx.Uint32).to(fx.Uint64) << 32)).bitcast(fx.Int64)
    inc = I(fx.rocdl.readfirstlane(T.i32, I(v)))
    return I(_llvm.inline_asm(T.i32, [fx.as_ir_value(inc), fx.as_ir_value(pointer)],
                              "s_atomic_add $0, $2, 0 glc\n\ts_waitcnt lgkmcnt(0)", "=s,0,s,~{memory}",
                              has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT))


def _xcc_id():
    return I(_llvm.inline_asm(T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)\n\ts_waitcnt lgkmcnt(0)", "=s",
                              has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT)) & 15


def _bf16_lo(w):
    return (w << 16).bitcast(F)


def _bf16_hi(w):
    return (w & -65536).bitcast(F)




def _dma(rsrc, lds_addr, voff, size):
    """``buffer_load ... lds``: ``size`` bytes per lane into LDS at ``lds_addr`` + lane * size."""
    lds = _llvm.IntToPtrOp(_ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(I(lds_addr)))
    fx.rocdl.raw_ptr_buffer_load_lds(rsrc, lds.result, fx.Int32(size), fx.Int32(voff), fx.Int32(0), fx.Int32(0), aux=NT)


def _by_wave(vals, w):
    out = I(vals[-1])
    for i in range(len(vals) - 2, -1, -1):
        out = (w == i).select(I(vals[i]), out)
    return out


def _wave_or(b, upper16, upper32):
    for mm in (1, 2, 4, 8):
        b = b | _xor_lane(b, mm, upper16)
    b = b | _xor_lane(b, 16, upper16)
    b = b | _xor_lane(b, 32, upper32)
    return I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(b)))


def compile_markov_chain(mt: int, ksteps: int):
    MT = mt
    K = ksteps
    T1 = TILES // 2  # tiles of the first half
    T2 = TILES - T1
    HMAXC = 2 * T2  # 8-column chunks per row of the larger half
    QMAX = 2 * TILES
    PBD = QMAX * 4  # dwords per row of the wave's staging buffer: QMAX 16-byte chunks
    ROWS = MT * 16
    SLAB_IT = (ROWS * QMAX + 63) // 64  # DMA instructions of a wave's slab
    SLAB_CH = SLAB_IT * 64  # chunks of its LDS region (whole instructions)
    assert 1 <= MT <= 3 and 1 <= K <= 8 and TILES * NWG * NW * 16 == V and TILES % 2 == 0
    RW = RES_TILES[MT]
    NSLOT = sum(RW)
    name = f"markov_gumbel_chain_v163840_r512_mt{MT}_k{K}"

    @fx.struct
    class Shared:
        e: fx.Array[fx.Int32, ROWS * ER, 16]
        buf: fx.Array[fx.Int32, NW * SLAB_CH * 4, 16]
        st: fx.Array[fx.Int32, K * 3 * 64]  # per step: h0, temperature bits, cache row offset (-1: none), by row
        pv: fx.Array[fx.Int32, NW * ROWS]
        pi: fx.Array[fx.Int32, NW * ROWS]
        tk: fx.Array[fx.Int32, 64]
        rk: fx.Array[fx.Int32, 4]
        rbuf: fx.Array[fx.Int32, max(NSLOT, 1) * 4096, 16]  # resident weight tiles: 16 KB each

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_base: fx.Int64,  # bf16 [n, k, V]: row stride i32_bs0 elements, step column s at s * V
        i32_bs0: fx.Int32,
        arg_w1: fx.Int64,  # bf16 [V, RANK]
        arg_w2: fx.Int64,  # bf16 [V, RANK]
        arg_ids: fx.Int64,  # int32 input_ids
        arg_anchor: fx.Int64,  # int64 [n]: step 0's previous token is input_ids[anchor[r]]
        arg_tokens: fx.Int64,  # int64 draft_tokens [*, i32_tok_stride]
        i32_tok_stride: fx.Int32,
        arg_idx: fx.Int64,  # int64 [n, i32_idx_stride]: request state index, < 0 invalid
        i32_idx_stride: fx.Int32,
        arg_pos: fx.Int64,  # int64 [n, i32_pos_stride]: sample_pos
        i32_pos_stride: fx.Int32,
        arg_temp: fx.Int64,  # fp32 [slots]
        arg_seed: fx.Int64,  # int64 [slots]
        arg_cache: fx.Int64,  # bf16 [rows, cache_cols, V]
        i32_cs0: fx.Int32,  # cache strides in elements
        i32_cs1: fx.Int32,
        arg_rowmap: fx.Int64,  # int32 [slots] cache row of a request, 0: the slot itself
        arg_sync: fx.Int64,  # generation, per-row result records, per-workgroup candidate records
        i32_n: fx.Int32,
        i32_pos_add: fx.Int32,  # low word added to sample_pos: 2^30 (vLLM's draft noise salt) - 1
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        ids_r = buffer_ops.create_buffer_resource_from_addr(arg_ids, num_records_bytes=0x7FFFFFFF)
        anc_r = buffer_ops.create_buffer_resource_from_addr(arg_anchor, num_records_bytes=i32_n * 8)
        tok_r = buffer_ops.create_buffer_resource_from_addr(arg_tokens, num_records_bytes=0x7FFFFFFF)
        w1_r = buffer_ops.create_buffer_resource_from_addr(arg_w1, num_records_bytes=V * RANK * 2)
        idx_r = buffer_ops.create_buffer_resource_from_addr(arg_idx, num_records_bytes=0x7FFFFFFF)
        pos_r = buffer_ops.create_buffer_resource_from_addr(arg_pos, num_records_bytes=0x7FFFFFFF)
        tmp_r = buffer_ops.create_buffer_resource_from_addr(arg_temp, num_records_bytes=0x7FFFFFFF)
        sed_r = buffer_ops.create_buffer_resource_from_addr(arg_seed, num_records_bytes=0x7FFFFFFF)
        rmp_r = buffer_ops.create_buffer_resource_from_addr(arg_rowmap, num_records_bytes=0x7FFFFFFF)
        base_r = buffer_ops.create_buffer_resource_from_addr(arg_base, num_records_bytes=0x7FFFFFFF)
        cache_r = buffer_ops.create_buffer_resource_from_addr(arg_cache, num_records_bytes=0x7FFFFFFF)
        w2_r = buffer_ops.create_buffer_resource_from_addr(arg_w2, num_records_bytes=V * RANK * 2)
        sy_r = buffer_ops.create_buffer_resource_from_addr(arg_sync, num_records_bytes=0x7FFFFFFF)
        has_map = arg_rowmap != 0

        def stamp(s, stage):
            if TS:  # noqa: SIM102
                if tx == 0:
                    t64 = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                    buffer_ops.buffer_store(t64.to(fx.Int32), sy_r, TS_OFF + (bid * K + s) * 24 + stage)

        stamp(0, 19)
        okl = lane < i32_n
        xcc = _xcc_id()
        upper16 = (lane & 16) != 0
        upper32 = (lane & 32) != 0

        def load_tile(t, n0v, mask=None):
            wi = n0v + t * 16 + lane % 16
            return [Vec(buffer_ops.buffer_load(w2_r, wi * (RANK // 2) + k * 16 + (lane // 16) * 4, vec_width=4,
                                               dtype=I, mask=mask)) for k in range_constexpr(KSTEPS)]

        def st_load1(s):
            slot = I(buffer_ops.buffer_load(idx_r, (lane * i32_idx_stride + s) * 2, vec_width=1, dtype=I, mask=okl))
            pos_lo = I(buffer_ops.buffer_load(pos_r, (lane * i32_pos_stride + s) * 2, vec_width=1, dtype=I,
                                              mask=okl)) + i32_pos_add
            return slot, pos_lo

        def st_load2(l1):
            slot = l1[0]
            valid = okl & (slot >= 0)
            slot0 = valid.select(slot, I(0))
            tmp = I(buffer_ops.buffer_load(tmp_r, slot0, vec_width=1, dtype=I, mask=valid)).bitcast(F)
            sd = Vec(buffer_ops.buffer_load(sed_r, slot0 * 2, vec_width=2, dtype=I, mask=valid))
            crow = has_map.select(I(buffer_ops.buffer_load(rmp_r, slot0, vec_width=1, dtype=I, mask=valid & has_map)),
                                  slot)
            return tmp, sd, crow

        def st_write(s, l1, l2):
            """Row ``lane``'s Gumbel seed state, temperature and cache offset of step ``s`` into the LDS table."""
            slot, pos_lo = l1
            tmp, sd, crow = l2
            valid = okl & (slot >= 0)
            cv = valid & (crow >= 0)
            smem.st.ptr[(s * 3 + 0) * 64 + lane] = _mix(_mix(_mix(I(0), I(sd[0])), I(sd[1]) ^ I(s)), pos_lo)
            smem.st.ptr[(s * 3 + 1) * 64 + lane] = valid.select(tmp, F(0.0)).bitcast(I)
            smem.st.ptr[(s * 3 + 2) * 64 + lane] = cv.select(crow * i32_cs0 + I(s) * i32_cs1, I(-1))

        def gather_issue(tokens):
            ws = []
            for i in range_constexpr(ROWS // NW):
                row = wave + NW * i
                tok = I(fx.rocdl.readlane(T.i32, fx.as_ir_value(tokens), fx.as_ir_value(row)))
                ws.append(Vec(buffer_ops.buffer_load(w1_r, tok * (RANK // 2) + lane * 4, vec_width=4, dtype=I)))
            return ws

        def gather_write(ws):
            for i in range_constexpr(ROWS // NW):
                row = wave + NW * i
                for j in range_constexpr(4):
                    smem.e.ptr[row * ER + lane * 4 + j] = ws[i][j]

        def dma_slab(step, n0v):
            """This wave's base-logit slab of ``step`` straight into LDS (row pitch QMAX 16-byte chunks)."""
            base_addr = lds_buf + wave * (SLAB_CH * 16)
            for it_ in range(fx.Index(0), fx.Index(SLAB_IT), fx.Index(1)):
                it = I(it_)
                j = lane + 64 * it
                r, q = j // QMAX, j % QMAX
                ok = (j < ROWS * QMAX) & (r < i32_n)
                voff = ok.select((r * i32_bs0 + step * V + n0v + q * 8) * 2, I(0x7FFFFFFF))
                _dma(base_r, base_addr + it * 1024, voff, 16)

        lds_buf = I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(I(fx.ptrtoint(smem.buf.ptr)))))

        # start: everything that does not wait on anything is issued first
        gen = I(buffer_ops.buffer_load(sy_r, 0, vec_width=1, dtype=I, cache_modifier=SC1))
        a_idx = I(buffer_ops.buffer_load(anc_r, lane * 2, vec_width=1, dtype=I, mask=okl))
        l1 = [st_load1(s) for s in range_constexpr(K)]
        if wave == NW - 1:  # this workgroup's index in its XCC
            smem.rk.ptr[0] = _atomic_add(arg_sync + fx.Int64(CNT_OFF * 4) + fx.Int64(xcc) * 128, 1) & 31
        tok_l = I(buffer_ops.buffer_load(ids_r, a_idx, vec_width=1, dtype=I, mask=okl))
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        rkw = I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(I(smem.rk.ptr[0]))))
        n0 = ((xcc * 32 + rkw) * NW + wave) * TILES * 16  # each XCC streams one contiguous eighth of the columns
        rw = _by_wave(RW, wave)  # resident tiles of this wave, from slot ``rslot0``
        rslot0 = _by_wave([sum(RW[:w]) for w in range(NW)], wave)
        fx.rocdl.sched_barrier(0)
        ws0 = gather_issue(tok_l)
        fx.rocdl.sched_barrier(0)
        l2 = [st_load2(l1[s]) for s in range_constexpr(K)]
        fx.rocdl.sched_barrier(0)
        gather_write(ws0)
        fx.rocdl.sched_barrier(0)
        nxt = load_tile(0, n0)
        fx.rocdl.sched_barrier(0)
        dma_slab(I(0), n0)
        fx.rocdl.sched_barrier(0)
        for s in range_constexpr(K):
            st_write(s, l1[s], l2[s])

        def fetch_next(from_lds, slot, nwi, gmask):
            out = [Vec.filled(4, 0, I) for _ in range(KSTEPS)]
            if from_lds:
                out = res_read(slot)
            else:
                out = [Vec(buffer_ops.buffer_load(w2_r, nwi * (RANK // 2) + k * 16 + (lane // 16) * 4, vec_width=4,
                                                  dtype=I, mask=gmask)) for k in range_constexpr(KSTEPS)]
            return out

        def finish_step(smem, s, noisy, divide, afr, temps, h0s, cur):
            def afr_at(m, k):
                return a_frag(m, k) if AFR_LDS[MT] else afr[m * KSTEPS + k]

            best_v = [F(NEG_INF) for _ in range(MT)]
            best_i = [I(0) for _ in range(MT)]
            rev = s & 1  # odd steps sweep the tiles backwards: the last tiles read (still in L2) are the first needed
            for h in range_constexpr(2):
                init = [v.ir_value() for v in best_v] + [v.ir_value() for v in best_i] + [v.ir_value() for v in cur]
                for t, old in range(fx.Index(0), fx.Index(T1 if h == 0 else T2), fx.Index(1), init=init):
                    best_v = [F(x) for x in old[:MT]]
                    best_i = [I(x) for x in old[MT:2 * MT]]
                    wv = [Vec(x) for x in old[2 * MT:]]
                    it = I(t) + (T1 if h == 1 else 0)  # iteration of the step; ``ti``: the tile of the wave's columns
                    ti = (rev == 1).select(TILES - 1 - it, it)
                    nit = it + 1
                    more = nit < TILES
                    nt = more.select((rev == 1).select(TILES - 1 - nit, nit), (rev == 1).select(I(0), I(TILES - 1)))
                    nres = more & (nt >= RES_LO) & (nt < RES_LO + rw)  # the next tile is resident
                    nwi = n0 + nt * 16 + lane % 16
                    first = s == 0
                    gmask = more.select(nres.select(first, more), s < K - 1)
                    nwv = fetch_next(nres & (s > 0), rslot0 + nt - RES_LO, nwi, gmask)
                    if first & (ti >= RES_LO) & (ti < RES_LO + rw):
                        res_write(smem, rslot0 + ti - RES_LO, wv)
                    acc = [Vec.filled(4, 0.0, F) for _ in range(MT)]
                    cur_a = [afr_at(m, 0) for m in range_constexpr(MT)]
                    for k in range_constexpr(KSTEPS):
                        nxt_a = [afr_at(m, k + 1) for m in range_constexpr(MT)] if k + 1 < KSTEPS else None
                        for m in range_constexpr(MT):
                            acc[m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                                Vec.make_type(4, F), [wv[k].bitcast(fx.BFloat16),
                                                      cur_a[m].bitcast(fx.BFloat16), acc[m], 0, 0, 0]))
                        cur_a = nxt_a
                    cl = ti * 16 + (lane // 16) * 4  # this lane's first column inside the wave's slab
                    c0 = n0 + cl
                    for m in range_constexpr(MT):
                        off = wave * SLAB_CH * 4 + (lane % 16 + 16 * m) * PBD + cl // 2
                        bw = [I(smem.buf.ptr[off]), I(smem.buf.ptr[off + 1])]
                        bias = [I(Vec.from_elements([F(acc[m][2 * e]), F(acc[m][2 * e + 1])], F).to(fx.BFloat16)
                                  .bitcast(I)[0]) for e in range_constexpr(2)]
                        ysum = []
                        for e in range_constexpr(2):
                            ysum.append(_bf16_lo(bw[e]) + _bf16_lo(bias[e]))
                            ysum.append(_bf16_hi(bw[e]) + _bf16_hi(bias[e]))
                        pk = [I(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(ysum[2 * i]), fx.as_ir_value(ysum[2 * i + 1])))
                              for i in range_constexpr(2)]
                        smem.buf.ptr[off] = pk[0]
                        smem.buf.ptr[off + 1] = pk[1]
                        tdiv = (temps[m] == F(0.0)).select(F(1.0), temps[m])
                        noisy_row = temps[m] != F(0.0)
                        kc = c0 * _i32(0xCC9E2D51)
                        for e in range_constexpr(4):
                            y = _bf16_lo(pk[e // 2]) if e % 2 == 0 else _bf16_hi(pk[e // 2])
                            v = (y / tdiv) if divide else y
                            if noisy:
                                v = v + noisy_row.select(_noise(h0s[m], kc + _i32((e * 0xCC9E2D51) & 0xFFFFFFFF)),
                                                         F(0.0))
                            gt = v > best_v[m]
                            gt = gt | ((v == best_v[m]) & ((c0 + e) < best_i[m]))  # the sweep may run backwards: first column wins ties
                            best_v[m] = gt.select(v, best_v[m])
                            best_i[m] = gt.select(c0 + e, best_i[m])
                    result = yield ([v.ir_value() for v in best_v] + [v.ir_value() for v in best_i] +
                                    [v.ir_value() for v in nwv])
                best_v = [F(x) for x in result[:MT]]
                best_i = [I(x) for x in result[MT:2 * MT]]
                cur = [Vec(x) for x in result[2 * MT:]]
                # the half's cache slab, coalesced 16 B per lane
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                for it_ in range(fx.Index(0), fx.Index((ROWS * HMAXC + 63) // 64), fx.Index(1)):
                    j = lane + 64 * I(it_)
                    r, q = j // HMAXC, j % HMAXC
                    co = I(smem.st.ptr[(s * 3 + 2) * 64 + r % ROWS])
                    rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    hp = rev if h == 0 else 1 - rev  # the physical half this iteration half covers
                    fx.copy(atom, fx.slice(buf16, (None, (wave * SLAB_CH * 4 + r * PBD + hp * T1 * 8 + q * 4) // 4)), rm)
                    buffer_ops.buffer_store(Vec(fx.memref_load_vec(rm)), cache_r,
                                            (co + n0 + hp * T1 * 16 + q * 8) // 2,
                                            mask=(j < ROWS * HMAXC) & (q < 2 * (T1 if h == 0 else T2)) & (co >= 0),
                                            cache_modifier=NT)
            for m in range_constexpr(MT):
                av, ai = best_v[m], best_i[m]
                for mm, up in ((16, upper16), (32, upper32)):
                    ov = _xor_lane(av.bitcast(I), mm, up).bitcast(F)
                    oi = _xor_lane(ai, mm, up)
                    av, ai = _comb(av, ai, ov, oi)
                if lane < 16:
                    smem.pv.ptr[(wave * MT + m) * 16 + lane] = av.bitcast(I)
                    smem.pi.ptr[(wave * MT + m) * 16 + lane] = ai
            return cur

        atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        lds16 = fx.logical_divide(fx.make_view(fx.recast_iter(fx.Int32, smem.e.ptr),
                                               fx.make_layout(ROWS * ER, 1)), fx.make_layout(4, 1))
        buf16 = fx.logical_divide(fx.make_view(fx.recast_iter(fx.Int32, smem.buf.ptr),
                                               fx.make_layout(NW * SLAB_CH * 4, 1)), fx.make_layout(4, 1))

        rbuf16 = fx.logical_divide(fx.make_view(fx.recast_iter(fx.Int32, smem.rbuf.ptr),
                                                fx.make_layout(max(NSLOT, 1) * 4096, 1)), fx.make_layout(4, 1))

        def res_read(slot):
            out = []
            for kk in range_constexpr(KSTEPS):
                rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(atom, fx.slice(rbuf16, (None, slot * 1024 + kk * 64 + lane)), rm)
                out.append(Vec(fx.memref_load_vec(rm)))
            return out

        def res_write(smem, slot, wv):
            for kk in range_constexpr(KSTEPS):
                for e in range_constexpr(4):
                    smem.rbuf.ptr[slot * 4096 + (kk * 64 + lane) * 4 + e] = wv[kk][e]

        def a_frag(m, k):
            rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(atom, fx.slice(lds16, (None, ((lane % 16 + 16 * m) * ER + k * 16 + (lane // 16) * 4) // 4)), rm)
            return Vec(fx.memref_load_vec(rm))

        for s, state in range(fx.Index(0), fx.Index(K), fx.Index(1), init=[v.ir_value() for v in nxt] + [tok_l.ir_value()]):
            s = I(s)
            nxt = [Vec(x) for x in state[:KSTEPS]]
            tok_l = I(state[KSTEPS])
            tag = gen + s + 1
            stamp(s, 0)
            # ---- prologue: w1[prev] rows into LDS (the base-logit slab is already on its way)
            if s > 0:
                gather_write(gather_issue(tok_l))
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            stamp(s, 1)
            afr = None if AFR_LDS[MT] else [a_frag(m, k) for m in range_constexpr(MT) for k in range_constexpr(KSTEPS)]
            temps = [I(smem.st.ptr[(s * 3 + 1) * 64 + lane % 16 + 16 * m]).bitcast(F) for m in range_constexpr(MT)]
            h0s = [I(smem.st.ptr[(s * 3 + 0) * 64 + lane % 16 + 16 * m]) for m in range_constexpr(MT)]
            flag = I(0)
            for m in range_constexpr(MT):
                gen_t = (temps[m] != F(0.0)) & (temps[m] != F(1.0))
                flag = flag | gen_t.select(I(3), (temps[m] != F(0.0)).select(I(1), I(0)))
            for mm in (1, 2, 4, 8):
                flag = flag | _xor_lane(flag, mm, upper16)
            flag = I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(flag)))

            nxt_out = nxt
            if flag >= 2:
                nxt_out = finish_step(smem, s, True, True, afr, temps, h0s, nxt)
            else:
                if flag == 1:
                    nxt_out = finish_step(smem, s, True, False, afr, temps, h0s, nxt)
                else:
                    nxt_out = finish_step(smem, s, False, False, afr, temps, h0s, nxt)
            nxt = nxt_out
            if s < K - 1:
                dma_slab(s + 1, n0)
            fx.gpu.barrier()
            stamp(s, 2)

            # ---- publish this workgroup's candidate of every row; reduce a row; read the result
            if wave == 0:
                if tx < i32_n:
                    m_t, l_t = tx // 16, tx % 16
                    pv = [I(smem.pv.ptr[(w * MT + m_t) * 16 + l_t]).bitcast(F) for w in range_constexpr(NW)]
                    pi = [I(smem.pi.ptr[(w * MT + m_t) * 16 + l_t]) for w in range_constexpr(NW)]
                    v, i = pv[0], pi[0]
                    for w in range_constexpr(1, NW):
                        v, i = _comb(v, i, pv[w], pi[w])
                    buffer_ops.buffer_store(Vec.from_elements([v.bitcast(I), i, tag, I(0)], I), sy_r,
                                            REC_OFF + (tx * NWG + bid) * REC_DW, cache_modifier=SC1)
                stamp(s, 3)
                if bid < i32_n:
                    def rd():
                        return [Vec(buffer_ops.buffer_load(sy_r, REC_OFF + (bid * NWG + lane * 4 + j) * REC_DW,
                                                           vec_width=4, dtype=I, cache_modifier=SC1))
                                for j in range_constexpr(4)]

                    def bad_of(rs):
                        b = I(0)
                        for r_ in rs:
                            b = b | (I(r_[2]) != tag).select(I(1), I(0))  # noqa: B023
                        return b
                    rs = rd()
                    bad = _wave_or(bad_of(rs), upper16, upper32)
                    spins = I(0)
                    while (bad != 0) & (spins < SPIN_LIMIT):
                        fx.rocdl.s_sleep(1)
                        rs = rd()
                        bad = _wave_or(bad_of(rs), upper16, upper32)
                        spins = spins + 1
                    if (bad != 0) & (lane == 0):
                        buffer_ops.buffer_store(I(1), sy_r, ERR_OFF, cache_modifier=SC1)
                    bv, bi = F(NEG_INF), I(0)
                    for r_ in rs:
                        bv, bi = _comb(bv, bi, I(r_[0]).bitcast(F), I(r_[1]))
                    for mm, up in ((1, upper16), (2, upper16), (4, upper16), (8, upper16), (16, upper16), (32, upper32)):
                        ov = _xor_lane(bv.bitcast(I), mm, up).bitcast(F)
                        oi = _xor_lane(bi, mm, up)
                        bv, bi = _comb(bv, bi, ov, oi)
                    if lane == 0:
                        buffer_ops.buffer_store(Vec.from_elements([bv.bitcast(I), bi, tag, I(0)], I), sy_r,
                                                RES_OFF + bid * REC_DW, cache_modifier=SC1)
                stamp(s, 4)
                if (s < K - 1) | (bid == 0):
                    def rdres():
                        return Vec(buffer_ops.buffer_load(sy_r, RES_OFF + lane * REC_DW, vec_width=4, dtype=I,
                                                          cache_modifier=SC1, mask=okl))

                    def bad_res(rv):
                        return (okl & (I(rv[2]) != tag)).select(I(1), I(0))  # noqa: B023
                    rv = rdres()
                    bad2 = _wave_or(bad_res(rv), upper16, upper32)
                    spins2 = I(0)
                    while (bad2 != 0) & (spins2 < SPIN_LIMIT):
                        fx.rocdl.s_sleep(1)
                        rv = rdres()
                        bad2 = _wave_or(bad_res(rv), upper16, upper32)
                        spins2 = spins2 + 1
                    if (bad2 != 0) & (lane == 0):
                        buffer_ops.buffer_store(I(1), sy_r, ERR_OFF, cache_modifier=SC1)
                    tok_v = ((I(rv[1]) >= 0) & (I(rv[1]) < V)).select(I(rv[1]), I(0))
                    smem.tk.ptr[lane] = tok_v
                    if bid == 0:  # noqa: SIM102
                        if okl:
                            buffer_ops.buffer_store(Vec.from_elements([tok_v, I(0)], I), tok_r,
                                                    (lane * i32_tok_stride + s) * 2)
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                stamp(s, 5)
            fx.gpu.barrier()
            tok_l = I(smem.tk.ptr[lane])
            yield [v.ir_value() for v in nxt] + [tok_l.ir_value()]
        if bid == 0:  # noqa: SIM102
            if tx == 0:
                buffer_ops.buffer_store(gen + K, sy_r, 0, cache_modifier=SC1)

    @flyc.jit
    def launch_markov_chain(
        arg_base: fx.Int64, i32_bs0: fx.Int32, arg_w1: fx.Int64, arg_w2: fx.Int64, arg_ids: fx.Int64,
        arg_anchor: fx.Int64, arg_tokens: fx.Int64, i32_tok_stride: fx.Int32, arg_idx: fx.Int64,
        i32_idx_stride: fx.Int32, arg_pos: fx.Int64, i32_pos_stride: fx.Int32, arg_temp: fx.Int64,
        arg_seed: fx.Int64, arg_cache: fx.Int64, i32_cs0: fx.Int32, i32_cs1: fx.Int32, arg_rowmap: fx.Int64,
        arg_sync: fx.Int64, i32_n: fx.Int32, i32_pos_add: fx.Int32, stream: fx.Stream,
    ):
        kernel(arg_base, i32_bs0, arg_w1, arg_w2, arg_ids, arg_anchor, arg_tokens, i32_tok_stride, arg_idx,
               i32_idx_stride, arg_pos, i32_pos_stride, arg_temp, arg_seed, arg_cache, i32_cs0, i32_cs1,
               arg_rowmap, arg_sync, i32_n, i32_pos_add).launch(grid=(NWG, 1, 1), block=(64 * NW, 1, 1),
                                                                stream=stream)

    return launch_markov_chain
