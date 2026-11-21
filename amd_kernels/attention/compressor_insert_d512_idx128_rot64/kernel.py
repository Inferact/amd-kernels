# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compressor tail and indexer key store, FlyDSL (gfx950): from the compressor's fp32 ``[kv | score]`` rows to the
fp8 main-cache rows and the MXFP4 indexer-cache rows, in one launch.

A workgroup of 16 waves takes 16 consecutive tokens. Phase 1, wave r the row t0 + r (lane l: dims 8l..8l+7): a
group-closing row (position + 1 divisible by the ratio) is pooled (ratio 2: the softmax-weighted pair of the previous
row and this one; the previous row of a request's first token is the ring row, the others' the raw row above),
RMS-normed to a bf16 latent that goes to LDS, rotated (GPT-J, dims 448..511, rotary position of the group's first
token) and stored as e4m3 (value * 1 / scale). Every token of a request shorter than the ring writes its raw row to its
ring slot; a longer request's first token writes the last ring-size rows after its ring read, so no row is written
where a first token still has to read. Phase 2: the latents times the indexer's wk (MFMA 16x16x32 bf16, K ascending;
wave w the output columns 16 (w % 8) .. + 15 and K half w // 8, the halves added through LDS, fp32) are rounded to
bf16, RMS-normed (per-tile sums of squares exchanged through LDS), rotated on columns 64..127 and quantized to MXFP4
(two tiles are one MX group, the pair exchanging its amax through LDS) into the paged indexer cache in the layout of
aiter's MQA-logits kernel (values [chunk, token, byte], scales [scale % 2, token, scale // 2]).

Loads that need only the token index (metadata, raw rows, weights) issue first, those that need the position and slots
(RoPE tables, ring rows) next, so a wave waits for two memory round trips. ``v_cvt_pk_bf16_f32`` is inline assembly
that the hazard recognizer does not see: nothing may convert an MFMA result with it (``.to(BFloat16)`` instead).

Order of the fp32 sums (``reference.py`` repeats it): the squares of a lane's dims in order, then quad, half-row and
row mirrors, then lanes L ^ 32 and L ^ 16; the wk norm's squares per lane in column order, then lanes ^ 32, ^ 16, then
the tiles ((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7)).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

NW = 16
TILE = 16  # tokens per workgroup
D = 512  # latent
KN = 128  # indexer key dim (wk output)
ROT = 64
LAT_STRIDE = D + 8  # bf16 elements of an LDS latent row (a 1040-byte stride: conflict-free 16-byte reads)
LOG2E = 0x3FB8AA3B
HWREG_MODE_FP16_OVFL = 1 | (23 << 6)
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _dpp_f(v, ctrl):
    o = _llvm.mlir_undef(T.i32)
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(_bits(v)), ctrl, 0xF, 0xF, True)).bitcast(fx.Float32)


def _permlane_swap(op, a, b):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(a), fx.as_ir_value(b), False, False)
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(_llvm.extractvalue(T.i32, res, [1]))


def _xlane4(x, op):
    """``op`` over lanes L, L ^ 16, L ^ 32, L ^ 48 (^ 32 first), broadcast to all four."""
    a, b = _permlane_swap(_rocdl_ops.permlane32_swap, _bits(x), _bits(x))
    m = op(_f32(a), _f32(b))
    a, b = _permlane_swap(_rocdl_ops.permlane16_swap, _bits(m), _bits(m))
    return op(_f32(a), _f32(b))


def _wave_sum(x):
    """Sum of the 64 lanes: quad (^ 1, ^ 2), half-row and row mirrors, then ^ 32, ^ 16."""
    a = x + _dpp_f(x, 0xB1)
    a = a + _dpp_f(a, 0x4E)
    a = a + _dpp_f(a, 0x141)
    a = a + _dpp_f(a, 0x140)
    return _xlane4(a, lambda p, q: p + q)


def _unpack2(w):
    w = fx.Int32(w)
    return [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]


def _unpack8(v):
    out = []
    for i in range(4):
        out += _unpack2(v[i])
    return out


def _pk(a, b):
    """Two fp32 -> two bf16 in a dword (RNE), ``a`` low."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def _rn(x):
    """fp32 -> bf16 (RNE) -> fp32."""
    return _unpack2(_pk(x, fx.Float32(0.0)))[0]


def _rsq(x):
    return fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(fx.Float32(x))))


def _exp(x):
    """exp(x) as the Triton ``tl.exp`` lowers: v_exp_f32 of x * log2(e)."""
    return fx.Float32(_rocdl_ops.exp2(T.f32, fx.as_ir_value(x * _f32(LOG2E))))


def _set_fp16_ovfl():
    _llvm.call_intrinsic(
        None, "llvm.amdgcn.s.setreg", [fx.Int32(HWREG_MODE_FP16_OVFL).ir_value(), fx.Int32(1).ir_value()], [], []
    )


def _cvt_pk_fp8(old, a, b, hi: bool):
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def _e4m3x8(y):
    lo = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), y[0], y[1], False), y[2], y[3], True)
    hi = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), y[4], y[5], False), y[6], y[7], True)
    return lo, hi


def _e2m1x2(a, b, old, sel):
    return fx.Int32(
        _rocdl_ops.cvt_scalef32_pk_fp4_f32(
            T.i32,
            fx.as_ir_value(fx.Int32(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def _raw(v):
    if not isinstance(v, _ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _lds_ptr(byte_addr):
    return _llvm.inttoptr(_ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def _ld_i32(addr, idx, nrec, uniform=True):
    """i32 ``idx`` of the array at ``addr`` (``nrec`` bytes); ``uniform``: the index is the same in every lane."""
    r = buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nrec)
    return fx.Int32(buffer_ops.buffer_load(r, idx, vec_width=1, dtype=fx.Int32, is_scalar=uniform))


def compile_compressor_insert(ratio: int, ring_rows: int, main_rows: int, idx_rows: int, cs_idx_bf16: bool):
    """``ratio``: compress ratio (1 or 2). ``ring_rows``: rows of a ring block (ratio 2). ``main_rows`` / ``idx_rows``:
    entries of a main-cache / indexer-cache page. ``cs_idx_bf16``: the indexer's cos | sin cache is bf16 (else fp32);
    the main cache's is fp32."""
    CR = ratio
    RS = D * CR  # fp32 columns of a raw row: kv (| score)
    CAP = ring_rows
    BSM = main_rows
    E = idx_rows
    name = f"compressor_insert_d{D}_idx{KN}_rot{ROT}_cr{CR}_r{CAP}_m{BSM}_e{E}" + ("_csbf16" if cs_idx_bf16 else "")
    LAT_B = TILE * LAT_STRIDE * 2
    X_OFF = LAT_B  # K-half partial sums: [8 tiles][64 lanes][4] fp32
    S_OFF = X_OFF + 8 * 64 * 16  # per-tile sums of squares [8][16] fp32
    A_OFF = S_OFF + 8 * TILE * 4  # per-tile amax [8][16] fp32
    LDS_B = A_OFF + 8 * TILE * 4

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS_B, 16]

    def _cs_pairs_main(cs_r, pg, jl):
        cv = Vec(buffer_ops.buffer_load(cs_r, pg * ROT + jl * 4, vec_width=4, dtype=T.f32))
        sv = Vec(buffer_ops.buffer_load(cs_r, pg * ROT + 32 + jl * 4, vec_width=4, dtype=T.f32))
        return cv, sv

    def _cs_pairs_idx(cs_r, pg, j0):
        """cos and sin of the two RoPE pairs j0, j0 + 1."""
        if cs_idx_bf16:
            cw = fx.Int32(buffer_ops.buffer_load(cs_r, pg * (ROT // 2) + j0 // 2, vec_width=1, dtype=fx.Int32))
            sw = fx.Int32(buffer_ops.buffer_load(cs_r, pg * (ROT // 2) + 16 + j0 // 2, vec_width=1, dtype=fx.Int32))
            return _unpack2(cw), _unpack2(sw)
        cvv = Vec(buffer_ops.buffer_load(cs_r, pg * ROT + j0, vec_width=2, dtype=T.f32))
        svv = Vec(buffer_ops.buffer_load(cs_r, pg * ROT + 32 + j0, vec_width=2, dtype=T.f32))
        return [cvv[0], cvv[1]], [svv[0], svv[1]]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_main_rows: fx.Int32,
        i32_cs_idx_rows: fx.Int32,
        arg_raw: fx.Int64,  # [n, RS] fp32: kv | score
        arg_pos: fx.Int64,  # [n] i64
        arg_ring: fx.Int64,  # [blocks, CAP, 1024] fp32
        arg_rslot: fx.Int64,  # [n] i64: ring slots (ratio 2) / main slots (ratio 1)
        arg_qsl: fx.Int64,  # [reqs + 1] i32
        arg_t2r: fx.Int64,  # [n] i32
        arg_nw: fx.Int64,  # [512] bf16 latent norm weight
        arg_main: fx.Int64,  # [pages, BSM, 512] e4m3
        arg_mslot: fx.Int64,  # [n] i64
        arg_cs_main: fx.Int64,  # [rows, 64] fp32
        arg_kscale: fx.Int64,  # fp32 scalar
        arg_wk: fx.Int64,  # wk in the MFMA fragment order (host.shuffle_wk)
        arg_kw: fx.Int64,  # [128] bf16
        arg_idx: fx.Int64,  # u8 pages
        arg_islot: fx.Int64,  # [n] i64
        arg_cs_idx: fx.Int64,  # [rows, 64]
        i32_raw_st: fx.Int32,  # raw row stride (elements)
        i32_ring_blk: fx.Int32,  # ring block stride (elements)
        i32_ring_row: fx.Int32,  # ring row stride (elements)
        i64_main_page: fx.Int64,  # main page stride (bytes)
        i64_idx_page: fx.Int64,  # indexer page stride (bytes)
        i32_eps: fx.Int32,  # fp32 bits, latent norm
        i32_keps: fx.Int32,  # fp32 bits, key norm
    ):
        _set_fp16_ovfl()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        t0 = bx * TILE
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lds = fx.Int32(fx.ptrtoint(smem.buf.ptr))

        def rl(v, r):
            return fx.Int32(fx.rocdl.readlane(T.i32, fx.Int32(v), fx.Int32(r)))

        def ld4(res, off, n=2):
            return [Vec(buffer_ops.buffer_load(res, off + 4 * h, vec_width=4, dtype=T.f32)) for h in range(n)]

        g = lane // 16
        l16 = lane % 16
        nt = wave % 8  # phase 2: output columns 16 nt .. + 15
        kh = wave // 8  # phase 2: K half
        # loads that need nothing computed: the metadata of token t0 + lane % 16 (phase 1's row is the wave: it takes
        # it by readlane), the weights, the raw rows
        tkn = t0 + l16
        m_ok = tkn < i32_n
        tk0 = m_ok.select(tkn, fx.Int32(0))
        pos_v = _ld_i32(arg_pos, tk0 * 2, fx.Int64(i32_n) * 8, False)
        rsl_v = _ld_i32(arg_rslot, tk0 * 2, fx.Int64(i32_n) * 8, False)
        msl_v = _ld_i32(arg_mslot, tk0 * 2, fx.Int64(i32_n) * 8, False)
        isl_v = _ld_i32(arg_islot, tk0 * 2, fx.Int64(i32_n) * 8, False)
        req_v = _ld_i32(arg_t2r, tk0, fx.Int64(i32_n) * 4, False) if CR == 2 else tk0
        nw = Vec(buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource_from_addr(arg_nw, num_records_bytes=D * 2),
            lane * 4, vec_width=4, dtype=fx.Int32))
        ks_r = buffer_ops.create_buffer_resource_from_addr(arg_kscale, num_records_bytes=4)
        kscale = _f32(fx.Int32(buffer_ops.buffer_load(ks_r, 0, vec_width=1, dtype=fx.Int32, is_scalar=True)))
        kw_r = buffer_ops.create_buffer_resource_from_addr(arg_kw, num_records_bytes=KN * 2)
        kwv = Vec(buffer_ops.buffer_load(kw_r, (nt * 16 + g * 4) // 2, vec_width=2, dtype=fx.Int32))
        wk_r = buffer_ops.create_buffer_resource_from_addr(arg_wk, num_records_bytes=KN * D * 2)
        wa = [Vec(buffer_ops.buffer_load(
            wk_r, ((nt * 16 + kh * (D // 64) + j) * 64 + lane) * 4, vec_width=4, dtype=fx.Int32
        )).bitcast(fx.BFloat16) for j in range_constexpr(D // 64)]
        r = wave
        t = t0 + r
        tt = (t < i32_n).select(t, fx.Int32(0))
        cur = buffer_ops.create_buffer_resource_from_addr(
            arg_raw + fx.Int64(tt) * fx.Int64(i32_raw_st * 4), num_records_bytes=RS * 4)
        kc = ld4(cur, lane * 8)
        if const_expr(CR == 2):
            sc = ld4(cur, D + lane * 8)
            prev_t = (tt > 0).select(tt - 1, fx.Int32(0))
            prow = buffer_ops.create_buffer_resource_from_addr(
                arg_raw + fx.Int64(prev_t) * fx.Int64(i32_raw_st * 4), num_records_bytes=RS * 4)
            kpr, spr = ld4(prow, lane * 8), ld4(prow, D + lane * 8)
        # then what needs the position and the slots: the RoPE tables, the ring row
        cs_main_r = buffer_ops.create_buffer_resource_from_addr(
            arg_cs_main, num_records_bytes=fx.Int64(i32_cs_main_rows) * (ROT * 4))
        cs_idx_r = buffer_ops.create_buffer_resource_from_addr(
            arg_cs_idx, num_records_bytes=fx.Int64(i32_cs_idx_rows) * (ROT * (2 if cs_idx_bf16 else 4)))
        is_rope = nt >= 4
        cs_idx = _cs_pairs_idx(cs_idx_r, (pos_v // CR) * CR, is_rope.select(nt - 4, fx.Int32(0)) * 8 + g * 2)
        pos = rl(pos_v, r)
        rslot = rl(rsl_v, r)
        mslot = rl(msl_v, r)
        cv, sv = _cs_pairs_main(cs_main_r, (pos // CR) * CR, lane % 8)
        if const_expr(CR == 2):
            rs = (rslot < 0).select(fx.Int32(0), rslot)
            ring_row = (
                fx.Int64(arg_ring) + fx.Int64(rs // CAP) * fx.Int64(i32_ring_blk * 4)
                + fx.Int64((pos + CAP - 1) % CAP) * fx.Int64(i32_ring_row * 4)
            )
            rr = buffer_ops.create_buffer_resource_from_addr(ring_row, num_records_bytes=RS * 4)
            kpg, spg = ld4(rr, lane * 8), ld4(rr, D + lane * 8)
            qs_v = _ld_i32(arg_qsl, req_v, (fx.Int64(i32_n) + 1) * 4, False)
            qe_v = _ld_i32(arg_qsl, req_v + 1, (fx.Int64(i32_n) + 1) * 4, False)
        else:
            qs_v, qe_v = tk0, tk0 + 1
        leader_v = tk0 == qs_v
        live_v = m_ok & (((pos_v + 1) % CR) == 0) & (rsl_v >= 0)

        nwf = _unpack8(nw)
        eps = _f32(i32_eps)
        kinv = fx.Float32(1.0) / kscale
        lim = fx.Float32(448.0)

        def ring_tail(qs, qe):
            """A request's last CAP raw rows into the ring (after this wave's ring read; chunks of CAP rows or more)."""
            fx.rocdl.s_waitcnt(vmcnt=0)
            tails = []
            for k in range_constexpr(CAP):
                tk = qe - CAP + k
                ok = tk >= qs
                tk = ok.select(tk, qs)
                tr = buffer_ops.create_buffer_resource_from_addr(
                    arg_raw + fx.Int64(tk) * fx.Int64(i32_raw_st * 4), num_records_bytes=RS * 4)
                vals = [Vec(buffer_ops.buffer_load(tr, lane * 16 + 4 * q, vec_width=4, dtype=T.f32))
                        for q in range_constexpr(4)]
                sl = _ld_i32(arg_rslot, tk * 2, fx.Int64(i32_n) * 8)
                tails.append((ok & (sl >= 0), sl, vals))
            for ok, sl, vals in tails:
                s0 = (sl < 0).select(fx.Int32(0), sl)
                dst = (fx.Int64(arg_ring) + fx.Int64(s0 // CAP) * fx.Int64(i32_ring_blk * 4)
                       + fx.Int64(s0 % CAP) * fx.Int64(i32_ring_row * 4))
                rr2 = buffer_ops.create_buffer_resource_from_addr(dst, num_records_bytes=RS * 4)
                for q in range_constexpr(4):
                    buffer_ops.buffer_store(vals[q], rr2, lane * 16 + 4 * q, mask=ok)

        # phase 1: the wave's row
        leader = rl(leader_v.select(fx.Int32(1), fx.Int32(0)), r) != 0
        live = rl(live_v.select(fx.Int32(1), fx.Int32(0)), r) != 0
        if const_expr(CR == 2):
            kp = [[leader.select(kpg[h][i], kpr[h][i]) for i in range(4)] for h in range(2)]
            sp = [[leader.select(spg[h][i], spr[h][i]) for i in range(4)] for h in range(2)]
            pooled = []
            for h in range_constexpr(2):
                for i in range_constexpr(4):
                    peak = _maxnum(sp[h][i], sc[h][i])
                    w_p, w_c = _exp(sp[h][i] - peak), _exp(sc[h][i] - peak)
                    pooled.append((kp[h][i] * w_p + kc[h][i] * w_c) / (w_p + w_c))
        else:
            pooled = [kc[h][i] for h in range_constexpr(2) for i in range_constexpr(4)]
        ss = pooled[0] * pooled[0]
        for i in range_constexpr(1, 8):
            ss = ss + pooled[i] * pooled[i]
        rstd = _rsq(_wave_sum(ss) * fx.Float32(1.0 / D) + eps)
        pk = [_pk((pooled[2 * i] * rstd) * nwf[2 * i], (pooled[2 * i + 1] * rstd) * nwf[2 * i + 1])
              for i in range_constexpr(4)]
        _llvm.store(_raw(Vec.from_elements(pk, fx.Int32)), _lds_ptr(lds + r * (LAT_STRIDE * 2) + lane * 16),
                    alignment=16)
        lat = _unpack8(pk)
        rope = lane >= 56
        y = []
        for i in range_constexpr(4):
            e, o = lat[2 * i], lat[2 * i + 1]
            y += [rope.select(_rn(e * cv[i] - o * sv[i]), e), rope.select(_rn(o * cv[i] + e * sv[i]), o)]
        y = [_minnum(_maxnum(v * kinv, -lim), lim) for v in y]
        lo, hi = _e4m3x8(y)
        ms = (mslot < 0).select(fx.Int32(0), mslot)
        rec = fx.Int64(arg_main) + fx.Int64(ms // BSM) * i64_main_page + fx.Int64(ms % BSM) * D
        mr = buffer_ops.create_buffer_resource_from_addr(rec, num_records_bytes=D)
        buffer_ops.buffer_store(Vec.from_elements([lo, hi], fx.Int32), mr, lane * 2, mask=live & (mslot >= 0))
        if const_expr(CR == 2):
            ring_ok = (t < i32_n) & (rslot >= 0)
            s0 = (rslot < 0).select(fx.Int32(0), rslot)
            dst = (fx.Int64(arg_ring) + fx.Int64(s0 // CAP) * fx.Int64(i32_ring_blk * 4)
                   + fx.Int64(s0 % CAP) * fx.Int64(i32_ring_row * 4))
            rw = buffer_ops.create_buffer_resource_from_addr(dst, num_records_bytes=RS * 4)
            nlen = rl(qe_v - qs_v, r)
            short = nlen < CAP
            for h in range_constexpr(2):
                buffer_ops.buffer_store(kc[h], rw, lane * 8 + 4 * h, mask=ring_ok & short)
                buffer_ops.buffer_store(sc[h], rw, D + lane * 8 + 4 * h, mask=ring_ok & short)
            if leader & ring_ok & (nlen >= CAP):
                ring_tail(rl(qs_v, r), rl(qe_v, r))
        fx.gpu.barrier()

        # phase 2: key = latent @ wk^T, columns 16 nt .. + 15, K half kh
        acc = Vec.filled(4, 0.0, fx.Float32)
        for j in range_constexpr(D // 64):
            b = Vec(_llvm.load(
                T.vec(4, T.i32),
                _lds_ptr(lds + l16 * (LAT_STRIDE * 2) + ((kh * (D // 64) + j) * 32 + g * 8) * 2), alignment=16,
            )).bitcast(fx.BFloat16)
            acc = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(Vec.make_type(4, fx.Float32), [wa[j], b, acc, 0, 0, 0]))
        xp = _lds_ptr(lds + X_OFF + (nt * 64 + lane) * 16)
        if kh == 1:
            _llvm.store(_raw(Vec.from_elements([fx.Float32(acc[i]) for i in range(4)], fx.Float32)), xp, alignment=16)
        fx.gpu.barrier()
        other = Vec(_llvm.load(T.vec(4, T.f32), xp, alignment=16))
        kp = [fx.Float32(fx.Float32(acc[i]) + fx.Float32(other[i])).to(fx.BFloat16).to(fx.Float32) for i in range(4)]
        ss = kp[0] * kp[0]
        for i in range_constexpr(1, 4):
            ss = ss + kp[i] * kp[i]
        part = _xlane4(ss, lambda p, q: p + q)
        if kh == 0:
            _llvm.store(part.ir_value(), _lds_ptr(lds + S_OFF + (nt * TILE + l16) * 4), alignment=4)
        fx.gpu.barrier()
        p = [fx.Float32(_llvm.load(T.f32, _lds_ptr(lds + S_OFF + (w * TILE + l16) * 4), alignment=4)) for w in range(8)]
        total = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))
        krstd = _rsq(total * fx.Float32(1.0 / KN) + _f32(i32_keps))
        kwf = _unpack2(kwv[0]) + _unpack2(kwv[1])
        yk = [_rn((kp[i] * krstd) * kwf[i]) for i in range(4)]
        cvk, svk = cs_idx
        rot = []
        for pr in range_constexpr(2):
            e, o = yk[2 * pr], yk[2 * pr + 1]
            rot += [is_rope.select(_rn(e * cvk[pr] - o * svk[pr]), e), is_rope.select(_rn(o * cvk[pr] + e * svk[pr]), o)]
        a = fx.math.absf(rot[0])
        for v in rot[1:]:
            a = _maxnum(a, fx.math.absf(v))
        a = _xlane4(a, _maxnum)
        if kh == 0:
            _llvm.store(a.ir_value(), _lds_ptr(lds + A_OFF + (nt * TILE + l16) * 4), alignment=4)
        fx.gpu.barrier()
        a_pair = fx.Float32(_llvm.load(T.f32, _lds_ptr(lds + A_OFF + ((nt ^ 1) * TILE + l16) * 4), alignment=4))
        a = _maxnum(a, a_pair)
        ratio_ = _maxnum(a, fx.Float32(6 * 2.0**-126)) * _f32(0x3E2AAAAB)
        e8 = (_bits(ratio_) + 0x7FFFFF) >> 23
        qsc = (fx.Int32(254) - e8 << 23).bitcast(fx.Float32)
        q = [v * qsc for v in rot]
        b01 = _e2m1x2(q[2], q[3], _e2m1x2(q[0], q[1], 0, 0), 1)
        ok = live_v & (isl_v >= 0) & (kh == 0)
        islot = isl_v
        pgs = fx.Int64(arg_idx) + fx.Int64(islot // E) * i64_idx_page
        ip = islot % E
        vbase = pgs + fx.Int64((ip // 32) * (32 * 64) + (nt // 2) * (32 * 16) + (ip % 32) * 16)
        vr = buffer_ops.create_buffer_resource_from_addr(vbase, num_records_bytes=KN)
        buffer_ops.buffer_store(b01.to(fx.Int16), vr, (nt % 2) * 4 + g, mask=ok)
        sg = nt // 2
        sbase = pgs + fx.Int64(E * 64 + (ip // 32) * (32 * 4) + (sg % 2) * (32 * 2) + (ip % 32) * 2 + sg // 2)
        sr = buffer_ops.create_buffer_resource_from_addr(sbase, num_records_bytes=4)
        buffer_ops.buffer_store(e8.to(fx.Int8), sr, 0, mask=ok & (g == 0) & ((nt % 2) == 0), offset_is_bytes=True)

    @flyc.jit
    def launch(
        arg_raw: fx.Int64,
        arg_pos: fx.Int64,
        arg_ring: fx.Int64,
        arg_rslot: fx.Int64,
        arg_qsl: fx.Int64,
        arg_t2r: fx.Int64,
        arg_nw: fx.Int64,
        arg_main: fx.Int64,
        arg_mslot: fx.Int64,
        arg_cs_main: fx.Int64,
        arg_kscale: fx.Int64,
        arg_wk: fx.Int64,
        arg_kw: fx.Int64,
        arg_idx: fx.Int64,
        arg_islot: fx.Int64,
        arg_cs_idx: fx.Int64,
        i32_n: fx.Int32,
        i32_cs_main_rows: fx.Int32,
        i32_cs_idx_rows: fx.Int32,
        i32_raw_st: fx.Int32,
        i32_ring_blk: fx.Int32,
        i32_ring_row: fx.Int32,
        i64_main_page: fx.Int64,
        i64_idx_page: fx.Int64,
        i32_eps: fx.Int32,
        i32_keps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_n, i32_cs_main_rows, i32_cs_idx_rows, arg_raw, arg_pos, arg_ring, arg_rslot, arg_qsl, arg_t2r, arg_nw,
            arg_main, arg_mslot, arg_cs_main, arg_kscale, arg_wk, arg_kw, arg_idx, arg_islot, arg_cs_idx,
            i32_raw_st, i32_ring_blk, i32_ring_row, i64_main_page, i64_idx_page, i32_eps, i32_keps,
        ).launch(grid=(fx.Int64((i32_n + (TILE - 1)) // TILE), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
