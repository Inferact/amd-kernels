# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MLA decode of the DSA layers for few rows: one workgroup per (row, 16-head group) holds the row's keys in LDS at once.

Record (656 B): 512 e4m3 latent | 4 f32 per-128 pow2 scales | 64 bf16 rope. Row q attends ``slots[indptr[q] : indptr[q + 1]]``
(physical slot = page * block_stride + offset, all >= 0); the cache is read as flat 656 B rows with 64-bit addresses.

Grid: ``ceil(rows * 6 / 8) * 8`` workgroups of 8 waves, workgroup p on XCD p % 8 handling unit ``(p % 8) * ceil(units / 8) +
p // 8``, so the six head groups of a row and the rows of a request share an XCD's L2. A pass covers up to 288 keys (the
whole row in practice; longer rows run further passes with a running max):

  * DMA: wave w gathers subtiles w, w + 8, w + 16 (16 keys, 512 B latent each) into LDS with ``global_load_lds`` (16 B per
    lane, 8 tokens per 4 instructions, swizzled so the K and V reads are bank-conflict free) and loads the per-block scales
    and the rope row of its own keys straight into registers. Index lookups go through ``ds_bpermute``.
  * q: the 16 heads are quantized once per workgroup, each wave quantizing 2 of the 64 (head, 128-dim block) units
    (8 lanes each) into an LDS staging record (fp8 hi, then fp8 lo = (x - hi) * 32, E8M0 block scale = max(exponent - 7, 1)),
    from which every wave reads its MFMA operands. A pre-quantized q (``reference.quantize_q``) skips the quantization.
  * QK: a wave computes S for its own subtiles, all 16 heads: scaled MFMA 16x16x128 on the raw fp8 latent (the record scale
    as E8M0 scale_a) with q hi and lo, plus the bf16 MFMA for the rope part.
  * Softmax: the max is exchanged through LDS; P = exp2(s - m + 8) in fp16 goes to LDS in the PV B-operand layout.
  * PV: wave w owns dims [64w, 64w + 64): P' = P * token scale of the value block (fp16), V stays raw fp8 in LDS
    (ds_read_b64_tr_b8, exact fp8 -> f16 conversion, f16 MFMA), fp32 accumulation.
  * Output: bf16 latent o / l and the natural-log LSE of scale * q.k per (row, head); a row with no key gives o = 0, lse = -inf.

LDS: the latent tile (288 x 512 B) first, then the P fragments, the f16 token scales and the max / sum exchange; the q staging
record overlays the P region until the first QK.

Preconditions the host cannot check: every per-128 scale is a power of two <= 2^7 (vLLM's writer: pow2ceil(max(amax, 1e-4) /
448), so |latent| <= 57344; a larger one is clamped, finite but inexact), and every slot is in the cache.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.dpp_utils import dpp_xor_f32
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, buf_base_i64
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.warm_up import launcher

from ._primitives import (
    HEADS,
    HG_HEADS,
    LN2,
    N_HG,
    RECORD_BYTES,
    SCALE_CAP_BITS,
    XCDS,
    _BoundedI32,
    _L,
    _cat,
    _cvt_v8,
    _e8m0_for,
    _fmax,
    _global_i32,
    _global_llvm_ptr,
    _guarded,
    _imin,
    _lds_llvm_ptr,
    _lds_load,
    _lds_store,
    _lo_scale,
    _mfma_bf16,
    _mfma_f16,
    _mfma_fp8,
    _q_load,
    _q_prologue,
    _quant,
    _raw,
    _readfirstlane,
    _tr8,
    _xmax,
    _xsum,
)

RT = 288
NSUB = RT // 16
KKN = RT // 32
NW = 8
NSLOT = (NSUB + NW - 1) // NW
CPW = 32 // NW
P_SHIFT = 8.0
P_OFF = RT * 512
SC_OFF = P_OFF + KKN * 1024
EX_OFF = SC_OFF + 4 * RT * 2
LDS_BYTES = EX_OFF + 1024
Q_SCALE_OFF = 8192


def _now():
    return fx.Int32(fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])))


def _f32_ptr(addr):
    return fx.inttoptr(fx.PointerType.get(T.f32, fx.AddressSpace.Shared, 4), fx.Int32(addr))


def q_prequant_load(q_t, q_row, g):
    """Pre-quantized q record (``reference.quantize_q`` layout) of this lane's head at i32 offset ``q_row``:
    (hi operands, hi E8M0, lo operands, lo E8M0, rope vectors)."""
    q8, q8l = [], []
    for j in range(4):
        c = 2 * j + g % 2
        for lo, dst in ((0, q8), (128, q8l)):
            a = Vec(q_t.load(q_row + lo + (g // 2) * 32 + c * 4, vec_size=4))
            b = Vec(q_t.load(q_row + lo + (2 + g // 2) * 32 + c * 4, vec_size=4))
            dst.append(_cat(a, b, 4))
    sw = fx.Int32(q_t[q_row + 288])
    qs = (sw >> (8 * g)) & 0xFF
    qr = [Vec(q_t.load(q_row + 256 + 16 * i + g * 4, vec_size=4)) for i in range(2)]
    return q8, qs, q8l, _lo_scale(qs), qr


def staged_q_frags(base, off, t, g):
    """MFMA operands of this lane's head from 512-B records staged at ``off`` (head t at 512 t, 16-B chunk c at c ^ t)."""
    a0 = t * 512 + (((8 * (g // 2) + g % 2) ^ t) << 4)
    out = []
    for j in range(4):
        a = base + off + (a0 ^ (32 * j))
        out.append(_cat(_lds_load(T.i32, 4, a, 16), _lds_load(T.i32, 4, a + 256, 16), 4))
    return out


@launcher
def compile_decode_row(pre_quantized: bool = False, timing: bool = False):
    """The decode launcher (rows is a runtime argument); ``timing`` stores s_memrealtime stamps at the phase ends into
    ``lse[rows * 96 + unit * 16 + k]`` (debug: ``lse`` then needs rows * 6 * 16 extra words)."""
    H = HG_HEADS
    PQ = pre_quantized
    name = "fp8_sparse_mla_decode_row" + ("_pq" if PQ else "")

    @fx.struct
    class Smem:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kern(
        Q: fx.Tensor,
        KV: fx.Tensor,
        IDX: fx.Tensor,
        IPTR: fx.Tensor,
        OUT: fx.Tensor,
        LSE: fx.Tensor,
        stride_q: fx.Int32,
        stride_qh: fx.Int32,
        stride_o: fx.Int32,
        stride_oh: fx.Int32,
        c_log2: fx.Float32,
        n_q: fx.Int32,
        idx_bytes: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        w = _readfirstlane(tid // 64)
        lane = tid % 64
        t = lane % 16
        g = lane // 16
        pid = fx.Int32(fx.block_idx.x)
        units = n_q * N_HG
        per_xcd = (units + (XCDS - 1)) // XCDS
        u = (pid % XCDS) * per_xcd + pid // XCDS
        u = (u < units).select(u, units - 1)
        qi = u // N_HG
        hg = u % N_HG

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_base = buf_base_i64(KV)
        idx_t = _BoundedI32(IDX, idx_bytes)
        ip_t = GTensor(IPTR, dtype=T.i32, shape=(-1,))
        o_t = GTensor(OUT, dtype=T.i32, shape=(-1,))
        lse_t = GTensor(LSE, dtype=T.i32, shape=(-1,))

        def stamp(k):
            if const_expr(timing):

                def _s(_k=k):
                    lse_t.store(n_q * HEADS + u * 16 + _k, _now())

                _guarded(tid == 0, _s)

        stamp(0)
        lds = fx.SharedAllocator().allocate(Smem).peek()
        base = _readfirstlane(fx.ptrtoint(lds.raw.ptr))

        r_beg = _readfirstlane(ip_t[qi])
        r_end = _readfirstlane(ip_t[qi + 1])
        r_len = r_end - r_beg
        r_len = (r_len < 0).select(fx.Int32(0), r_len)
        npass = _readfirstlane((r_len + (RT - 1)) // RT)

        hq = hg * H + t
        if const_expr(PQ):
            q8, qs, q8l, qsl, qr = q_prequant_load(q_t, (qi * stride_q + hq * stride_qh) // 4, g)
        else:
            q_row = (qi * stride_q + hq * stride_qh) // 2
            qr = [Vec(q_t.load(q_row + (_L + 32 * i) // 2 + g * 4, vec_size=4)) for i in range_constexpr(2)]

        raw0 = fx.Int32(idx_t[r_beg + lane])
        first_slot = _readfirstlane(raw0)
        # an empty row (CUDA-graph padding, or a rank with no keys) still issues its first DMA pass: slots[r_beg]
        # is then past the CSR end (stale / uninitialized), so read the always-valid record 0 instead
        first_slot = (r_len > 0).select(first_slot, fx.Int32(0))

        def ld_idx(p):
            ix = fx.Int32(idx_t[r_beg + p])
            return (p < r_len).select(ix, first_slot)

        stamp(1)
        neg_inf = fx.Float32(-1.0e30)

        # per-lane LDS offsets: swizzle f(token) = (token & 15) ^ ((token >> 4 & 1) << 3) moves a 16-B chunk c to c ^ f
        c0 = 8 * (g // 2) + g % 2
        vi = t // 2
        vtok = 16 * (vi // 4) + 4 * g + vi % 4
        fv = (4 * g + vi % 4) ^ (8 * (vi // 4))
        v_off = vtok * 512 + (((CPW * w) ^ fv) << 4) + (t % 2) * 8

        ssl = [w + NW * k for k in range(NSLOT)]
        valid_last = (w + NW * (NSLOT - 1)) < NSUB

        def dma_addrs(s, ixs):
            par = s & 1
            out = []
            for m in range_constexpr(8):
                tok_lo = 2 * m + (lane >> 5)
                f = tok_lo ^ (par << 3)
                ix = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(tok_lo * 4), _raw(ixs)))
                out.append(kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(((lane & 31) ^ f) * 16))
            return out

        def dma_group(s, addrs):
            for m in range_constexpr(8):
                rocdl.global_load_lds(_global_llvm_ptr(addrs[m]), _lds_llvm_ptr(base + (16 * s + 2 * m) * 512), 16, 0)

        def side_loads(ixs):
            row = kv_base + fx.Int64(ixs) * RECORD_BYTES
            scl = fx.Int32(_global_i32(row + fx.Int64(_L + 4 * g), 1))
            rp = [_global_i32(row + fx.Int64(_L + 16 + 16 * g + 64 * i), 4) for i in range(2)]
            return scl, rp

        def qk_subtile(s, scl, rp):
            tokl = 16 * s + t
            fk = t ^ ((s & 1) << 3)
            koff = tokl * 512 + ((c0 ^ fk) << 4)
            ses = (scl >> 23) & 0xFF
            sacc = Vec.filled(4, 0.0, fx.Float32)
            for j in range_constexpr(4):
                a1 = base + (koff ^ (32 * j))
                kf = _cat(_lds_load(T.i32, 4, a1, 16), _lds_load(T.i32, 4, a1 + 256, 16), 4)
                acc = _mfma_fp8(kf, q8[j], sacc, ses, qs)
                sacc = _mfma_fp8(kf, q8l[j], acc, ses, qsl)
            for i in range_constexpr(2):
                sacc = _mfma_bf16(rp[i], qr[i], sacc)
            return sacc

        def issue_pass(cb):
            """DMA of the pass's keys into LDS plus the scale / rope loads; returns (scales, rope vectors)."""
            ixs_l = [ld_idx(cb + 16 * ssl[k] + t) for k in range(NSLOT)]
            addrs = [dma_addrs(ssl[k], ixs_l[k]) for k in range(NSLOT)]
            for k in range_constexpr(NSLOT - 1):
                dma_group(ssl[k], addrs[k])
            _guarded(valid_last, lambda _s=ssl[NSLOT - 1], _a=addrs[NSLOT - 1]: dma_group(_s, _a))
            scs, rps = [], []
            for k in range_constexpr(NSLOT):
                scl_k = fx.Int32(0x3F800000)
                rp_k = [Vec.filled(4, 0, fx.Int32), Vec.filled(4, 0, fx.Int32)]
                if const_expr(k < NSLOT - 1):
                    scl_k, rp_k = side_loads(ixs_l[k])
                else:
                    if valid_last:
                        scl_k, rp_k = side_loads(ixs_l[k])
                scs.append(scl_k)
                rps += rp_k
            return scs, rps

        def quant_stage():
            """Quantize this wave's (head, block) units of q (8 lanes each) into the staged hi record (lo stays in regs)."""
            held = []
            for r in range_constexpr(8 // NW):
                uid = (w + NW * r) * 8 + (lane >> 3)
                head = uid >> 2
                blk = uid & 3
                sub = lane & 7
                off = (qi * stride_q + (hg * H + head) * stride_qh) // 2 + blk * 64 + sub * 8
                vals = []
                for v in (Vec(q_t.load(off, vec_size=4)), Vec(q_t.load(off + 4, vec_size=4))):
                    vals += [fx.Float32(x) for x in Vec(v.bitcast(fx.BFloat16)).to(fx.Float32)]
                am = fx.Float32(0.0)
                for x in vals:
                    am = _fmax(am, _fmax(x, fx.Float32(0.0) - x))
                for o_ in (1, 2, 4):
                    am = _fmax(am, dpp_xor_f32(am, o_))
                s8, inv = _e8m0_for(am)
                hi, lo = _quant(vals, inv)
                _lds_store(
                    Vec.from_elements(hi, fx.Int32),
                    T.i32,
                    base + P_OFF + head * 512 + (((blk * 8 + sub) ^ head) << 4),
                    16,
                )
                _guarded(
                    sub == 0,
                    lambda _h=head, _b=blk, _s=s8: _lds_store(
                        Vec.from_elements([fx.Int32(_s)], fx.Int32).to(fx.Int8),
                        T.i8,
                        base + P_OFF + Q_SCALE_OFF + _h * 4 + _b,
                        1,
                    ),
                )
                held.append((lo, head, blk, sub))
            return held

        sc0, rp0 = issue_pass(fx.Int32(0))
        stamp(2)
        if const_expr(not PQ):
            held = quant_stage()
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            q8 = staged_q_frags(base, P_OFF, t, g)
            sw = fx.Int32(fx.ptr_load(_i32_ptr(base + P_OFF + Q_SCALE_OFF + t * 4), result_type=T.i32))
            qs = (sw >> (8 * g)) & 0xFF
            qsl = _lo_scale(qs)
            gpu.barrier()
            for lo, head, blk, sub in held:
                _lds_store(
                    Vec.from_elements(lo, fx.Int32),
                    T.i32,
                    base + P_OFF + head * 512 + (((blk * 8 + sub) ^ head) << 4),
                    16,
                )
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            q8l = staged_q_frags(base, P_OFF, t, g)
        stamp(3)

        init = [_raw(neg_inf.ir_value()), _raw(fx.Float32(0.0).ir_value())]
        init += [_raw(Vec.filled(4, 0.0, fx.Float32).ir_value()) for _ in range(CPW)]
        init += [_raw(fx.Int32(x).ir_value()) for x in sc0]
        init += [_raw(Vec(x).ir_value()) for x in rp0]
        for pas, st in range(0, npass, 1, init=init):
            m_run = fx.Float32(st[0])
            l_run = fx.Float32(st[1])
            o = [Vec(st[2 + i]) for i in range(CPW)]
            scs = [fx.Int32(st[2 + CPW + i]) for i in range(NSLOT)]
            rps = [Vec(st[2 + CPW + NSLOT + i]) for i in range(2 * NSLOT)]
            cb = fx.Int32(pas) * RT

            rocdl.s_waitcnt(vmcnt=0)
            stamp(4)

            sacc_all = []
            for k in range_constexpr(NSLOT):
                sacc_k = Vec.filled(4, 0.0, fx.Float32)
                if const_expr(k < NSLOT - 1):
                    sacc_k = qk_subtile(ssl[k], scs[k], rps[2 * k : 2 * k + 2])
                else:
                    if valid_last:
                        sacc_k = qk_subtile(ssl[k], scs[k], rps[2 * k : 2 * k + 2])
                sacc_all.append(sacc_k)
            stamp(5)

            tmax = neg_inf
            sl = []
            for k in range_constexpr(NSLOT):
                for r in range_constexpr(4):
                    x = fx.Float32(Vec(sacc_all[k])[r]) * c_log2
                    ok = (cb + 16 * ssl[k] + 4 * g + r) < r_len
                    if k == NSLOT - 1:
                        ok = ok & valid_last
                    x = ok.select(x, neg_inf)
                    sl.append(x)
                    tmax = _fmax(tmax, x)
            tmax = _xmax(tmax)
            _lds_store(Vec.from_elements([tmax], fx.Float32), T.f32, base + EX_OFF + (w * 16 + t) * 4, 4)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            m_g = fx.Float32(fx.ptr_load(_f32_ptr(base + EX_OFF + t * 4), result_type=T.f32))
            for j in range_constexpr(1, NW):
                m_g = _fmax(m_g, fx.Float32(fx.ptr_load(_f32_ptr(base + EX_OFF + (j * 16 + t) * 4), result_type=T.f32)))
            m_new = _fmax(m_run, m_g)
            alpha = (m_run - m_new).exp2(fastmath="afn")
            m_cl = _fmax(m_new - P_SHIFT, fx.Float32(-1.0e20))
            pl = [(sl[i] - m_cl).exp2(fastmath="afn") for i in range(4 * NSLOT)]
            psum = fx.Float32(0.0)
            for x in pl:
                psum = psum + x
            psum = _xsum(psum)
            _lds_store(Vec.from_elements([psum], fx.Float32), T.f32, base + EX_OFF + 16 * 4 * NW + (w * 16 + t) * 4, 4)

            def store_ps(k):
                s = ssl[k]
                hv = Vec.from_elements([pl[4 * k + r] for r in range(4)], fx.Float32).to(fx.Float16).bitcast(fx.Int32)
                _lds_store(hv, T.i32, base + P_OFF + (s // 2) * 1024 + lane * 16 + (s % 2) * 8, 8)
                scb = _imin(scs[k], fx.Int32(SCALE_CAP_BITS)).bitcast(fx.Float32)
                _lds_store(
                    Vec.from_elements([scb], fx.Float32).to(fx.Float16), T.f16, base + SC_OFF + (g * RT + 16 * s + t) * 2, 2
                )

            for k in range_constexpr(NSLOT - 1):
                store_ps(k)
            _guarded(valid_last, lambda: store_ps(NSLOT - 1))
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            stamp(6)

            l_pass = fx.Float32(fx.ptr_load(_f32_ptr(base + EX_OFF + 16 * 4 * NW + t * 4), result_type=T.f32))
            for j in range_constexpr(1, NW):
                l_pass = l_pass + fx.Float32(
                    fx.ptr_load(_f32_ptr(base + EX_OFF + 16 * 4 * NW + (j * 16 + t) * 4), result_type=T.f32)
                )
            l_new = l_run * alpha + l_pass
            o = [
                Vec.from_elements([fx.Float32(Vec(o[i])[j]) * alpha for j in range(4)], fx.Float32) for i in range(CPW)
            ]
            blk_w = (CPW * w) // 8
            for kk in range_constexpr(KKN):
                pf = _lds_load(T.i32, 4, base + P_OFF + kk * 1024 + lane * 16, 16)
                sv = [
                    _lds_load(T.i32, 2, base + SC_OFF + (blk_w * RT + 32 * kk + 16 * sp + 4 * g) * 2, 8)
                    for sp in range_constexpr(2)
                ]
                bfr = (Vec(pf).bitcast(fx.Float16) * Vec(_cat(sv[0], sv[1], 2)).bitcast(fx.Float16)).bitcast(fx.Int32)
                for d in range_constexpr(CPW):
                    o[d] = _mfma_f16(_cvt_v8(_tr8(base + (v_off ^ (d * 16)), kk * 16384)), bfr, o[d])
            stamp(7)
            scs_n = [fx.Int32(0x3F800000) for _ in range(NSLOT)]
            rps_n = [Vec.filled(4, 0, fx.Int32) for _ in range(2 * NSLOT)]
            if fx.Int32(pas) + 1 < npass:
                gpu.barrier()
                scs_n, rps_n = issue_pass(cb + RT)
            fin = yield [m_new, l_new] + o + scs_n + rps_n

        l_fin = fx.Float32(fin[1])
        has_l = l_fin > fx.Float32(0.0)
        inv_l = has_l.select(fx.Float32(1.0) / l_fin, fx.Float32(0.0))
        o_row = (qi * stride_o + (hg * H + t) * stride_oh) >> 1
        for d in range_constexpr(CPW):
            ov = Vec(fin[2 + d])
            bf = Vec.from_elements([fx.Float32(ov[j]) * inv_l for j in range(4)], fx.Float32).to(fx.BFloat16)
            o_t.store(
                o_row + 8 * (CPW * w + d) + 2 * g,
                bf.bitcast(fx.Int32),
                vec_size=2,
            )
        lse2 = fx.Float32(fin[0]) + fx.Float32(fx.log2(has_l.select(l_fin, fx.Float32(1.0)))) - P_SHIFT
        lse = has_l.select(lse2 * LN2, fx.Float32(float("-inf")))
        lse_t.store(qi * HEADS + hg * H + t, lse.bitcast(fx.Int32))
        stamp(8)

    @flyc.jit
    def launch_decode(
        Q,
        KV,
        IDX,
        IPTR,
        OUT,
        LSE,
        n_q,
        stride_q,
        stride_qh,
        stride_o,
        stride_oh,
        c_log2,
        idx_bytes,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kern(Q, KV, IDX, IPTR, OUT, LSE, stride_q, stride_qh, stride_o, stride_oh, c_log2, n_q, idx_bytes).launch(
            grid=(fx.Int64(((n_q * N_HG + XCDS - 1) // XCDS) * XCDS), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_decode.compile_hints = {"waves_per_eu": 1, "fast_fp_math": False}
    return launch_decode


def _i32_ptr(addr):
    return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4), fx.Int32(addr))
