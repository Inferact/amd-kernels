# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MLA decode of the DSA layers for many rows: one workgroup per (row, 3 head groups), a role-split tile loop.

Record (656 B): 512 e4m3 latent | 4 f32 per-128 pow2 scales | 64 bf16 rope. Row q attends ``slots[indptr[q] : indptr[q + 1]]``
(physical slot = page * block_stride + offset, all >= 0); the cache is read as flat 656 B rows with 64-bit addresses.

Grid: ``ceil(rows * 2 / 8) * 8`` workgroups of 8 waves on XCD p % 8; the two workgroups of a row and the rows of a request share
an XCD's L2. The row's keys stream through a ring of three LDS tiles of 64 keys; the waves run a tile apart:

  * wave 3 (DMA wave) alone gathers each tile with ``global_load_lds`` (44 instructions: 32 latent, 8 rope, 4 scale; index
    lookups through ``ds_bpermute``) one tile ahead. No other wave has a DMA pending, so their LDS reads never wait for one.
  * waves 0-2 (QK waves, one 16-head group each) quantize q to an fp8 hi plus an fp8 residual (E8M0 block scale) or load it
    pre-quantized, then per tile: scaled MFMA 16x16x128 on the raw fp8 latent with q hi and lo, the bf16 MFMA for the rope
    part, an online softmax whose reference max moves only when a tile exceeds it by 3 (and then to the tile max + 6, so the
    O rescale is rare), P = exp2(s - m + 5) in fp16 to LDS.
  * waves 4-7 (PV waves, one 128-dim value block each, all three head groups) run the previous tile: P' = P * token scale of
    the block (fp16), V raw fp8 in LDS (ds_read_b64_tr_b8, exact fp8 -> f16 conversion, f16 MFMA), fp32 accumulation, O
    rescaled by exp2(m_old - m_new) only when some head moved its reference max (a ballot).
  * One barrier per tile (all waves) publishes P and the landed tile. Output: bf16 latent o / l and the natural-log LSE of
    scale * q.k per (row, head); a row with no key gives o = 0, lse = -inf.

Preconditions the host cannot check: every per-128 scale is a power of two <= 2^7 (vLLM's writer: pow2ceil(max(amax, 1e-4) /
448), so |latent| <= 57344; a larger one is clamped, finite but inexact), and every slot is in the cache.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
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
    _fmax,
    _global_llvm_ptr,
    _guarded,
    _imin,
    _lds_llvm_ptr,
    _lds_load,
    _lds_store,
    _mfma_bf16,
    _mfma_f16,
    _mfma_fp8,
    _q_load,
    _q_prologue,
    _raw,
    _readfirstlane,
    _tr8,
    _xmax,
    _xsum,
)
from .kernel_row import q_prequant_load

NH = 3
NW = 8
NB = 3
TT = 64
THR = 3.0
MARGIN = 6.0
SHIFT = 5.0
LAT_B = TT * 512
SCL_OFF = LAT_B
ROPE_OFF = SCL_OFF + 4 * TT * 4
BUF = ROPE_OFF + TT * 128
CPW = 8
P_BUF = NH * 2 * 1024
AL_BUF = NH * 16 * 4
P_OFF = NB * BUF
AL_OFF = P_OFF + 2 * P_BUF
L_OFF = AL_OFF + 2 * AL_BUF
LDS_BYTES = L_OFF + NH * 16 * 4 + 64
SPLIT = N_HG // NH
DMA_WAVE = 3


def _now():
    return fx.Int32(fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])))


def _f32_ptr(addr):
    return fx.inttoptr(fx.PointerType.get(T.f32, fx.AddressSpace.Shared, 4), fx.Int32(addr))


def _i32_ptr(addr):
    return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4), fx.Int32(addr))


@launcher
def compile_decode_tile(pre_quantized: bool = False, timing: bool = False):
    """The decode launcher (rows is a runtime argument); ``timing`` stores s_memrealtime stamps into
    ``lse[rows * 96 + unit * 16 + k]`` (phase ends) and ``lse[rows * 96 + units * 16 + unit * 64 + tile * 8 + k]`` (tile
    phases), debug: ``lse`` then needs rows * 2 * 80 extra words."""
    H = HG_HEADS
    PQ = pre_quantized
    name = "fp8_sparse_mla_decode_tile" + ("_pq" if PQ else "")

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
        units = n_q * SPLIT
        per_xcd = (units + (XCDS - 1)) // XCDS
        u = (pid % XCDS) * per_xcd + pid // XCDS
        u = (u < units).select(u, units - 1)
        qi = u // SPLIT
        hg0 = (u % SPLIT) * NH

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

        def tstamp(tile, k, who=0):
            if const_expr(timing):

                def _s(_tile=tile, _k=k):
                    lse_t.store(n_q * HEADS + units * 16 + u * 64 + fx.Int32(_tile) * 8 + _k, _now())

                _guarded(tid == 64 * who, _s)

        stamp(0)
        lds = fx.SharedAllocator().allocate(Smem).peek()
        base = _readfirstlane(fx.ptrtoint(lds.raw.ptr))

        def row_bounds():
            r_beg = _readfirstlane(ip_t[qi])
            r_end = _readfirstlane(ip_t[qi + 1])
            r_len = r_end - r_beg
            r_len = (r_len < 0).select(fx.Int32(0), r_len)
            return r_beg, r_len, _readfirstlane((r_len + (TT - 1)) // TT)

        def dma_tile(ixs, bbase):
            """The 44 DMA operations of a tile (32 latent, 8 rope, 4 scale), all by the DMA wave."""
            ops = []
            for i in range_constexpr(32):
                x = 2 * i + (lane >> 5)
                ix = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(x * 4), _raw(ixs)))
                fs = (x & 15) ^ ((x >> 1) & 8)
                ops.append(
                    (kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(((lane & 31) ^ fs) * 16), bbase + i * 1024, 16)
                )
            for kk in range_constexpr(8):
                x = 8 * kk + (lane >> 3)
                ix = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(x * 4), _raw(ixs)))
                ops.append(
                    (
                        kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(_L + 16 + ((lane & 7) ^ (lane >> 3)) * 16),
                        bbase + ROPE_OFF + kk * 1024,
                        16,
                    )
                )
            for b in range_constexpr(4):
                ops.append(
                    (
                        kv_base + fx.Int64(ixs) * RECORD_BYTES + fx.Int64(_L) + fx.Int64(4 * b),
                        bbase + SCL_OFF + b * 256,
                        4,
                    )
                )
            for src, dst, size in ops:
                rocdl.global_load_lds(_global_llvm_ptr(src), _lds_llvm_ptr(dst), size, 0)

        def sync_tile():
            """Wait for the landed tile and sync all waves."""
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()

        neg_inf = fx.Float32(-1.0e30)

        def qk_role():
            hq = (hg0 + w) * H + t
            if const_expr(PQ):
                q8, qs, q8l, qsl, qr = q_prequant_load(q_t, (qi * stride_q + hq * stride_qh) // 4, g)
            else:
                q_row = (qi * stride_q + hq * stride_qh) // 2
                raw_q = _q_load(q_t, q_row, g)
                qr = [Vec(q_t.load(q_row + (_L + 32 * i) // 2 + g * 4, vec_size=4)) for i in range_constexpr(2)]
                q8, qs, q8l, qsl = _q_prologue(raw_q, g)
            stamp(3)
            _, r_len, nt = row_bounds()

            def kread(bb, s, j):
                c1 = 8 * (g // 2) + 2 * j + g % 2
                a1 = bb + (16 * s + t) * 512 + ((c1 ^ (t ^ ((s & 1) << 3))) << 4)
                return _cat(_lds_load(T.i32, 4, a1, 16), _lds_load(T.i32, 4, a1 + 256, 16), 4)

            def qk_softmax(bb, m_old, l_old, pos0):
                ses = [
                    (fx.Int32(fx.ptr_load(_i32_ptr(bb + SCL_OFF + (g * 64 + 16 * s + t) * 4), result_type=T.i32)) >> 23)
                    & 0xFF
                    for s in range_constexpr(4)
                ]
                sa = [Vec.filled(4, 0.0, fx.Float32) for _ in range(4)]
                for j in range_constexpr(4):
                    kr = [kread(bb, s, j) for s in range(4)]
                    for s in range_constexpr(4):
                        acc = _mfma_fp8(kr[s], q8[j], sa[s], ses[s], qs)
                        sa[s] = _mfma_fp8(kr[s], q8l[j], acc, ses[s], qsl)
                for s in range_constexpr(4):
                    tk = 16 * s + t
                    ra = bb + ROPE_OFF + (tk // 8) * 1024 + (t % 8) * 128
                    for i in range_constexpr(2):
                        sa[s] = _mfma_bf16(_lds_load(T.i32, 4, ra + (((g + 4 * i) ^ (t % 8)) << 4), 16), qr[i], sa[s])
                sl = []
                tmax = neg_inf
                for s in range_constexpr(4):
                    for r in range_constexpr(4):
                        x = fx.Float32(Vec(sa[s])[r]) * c_log2
                        x = ((pos0 + 16 * s + 4 * g + r) < r_len).select(x, neg_inf)
                        sl.append(x)
                        tmax = _fmax(tmax, x)
                tmax = _xmax(tmax)
                m_new = (tmax > m_old + THR).select(tmax + MARGIN, m_old)
                alpha = (m_old - m_new).exp2(fastmath="afn")
                m_cl = _fmax(m_new - SHIFT, fx.Float32(-1.0e20))
                pl = [(sl[k] - m_cl).exp2(fastmath="afn") for k in range(16)]
                psum = fx.Float32(0.0)
                for x in pl:
                    psum = psum + x
                return m_new, l_old * alpha + psum, alpha, pl

            def p_store(pb, pl, alpha):
                for kk in range_constexpr(2):
                    vals = [pl[(2 * kk + sp) * 4 + r] for sp in range(2) for r in range(4)]
                    hv = Vec.from_elements(vals, fx.Float32).to(fx.Float16).bitcast(fx.Int32)
                    _lds_store(hv, T.i32, base + P_OFF + pb * P_BUF + (w * 2 + kk) * 1024 + lane * 16, 16)
                _lds_store(
                    Vec.from_elements([alpha], fx.Float32), T.f32, base + AL_OFF + pb * AL_BUF + (w * 16 + t) * 4, 4
                )

            init = [_raw(neg_inf.ir_value()), _raw(fx.Float32(0.0).ir_value())]
            for tl, st in range(0, nt + 1, 1, init=init):
                m_a = fx.Float32(st[0])
                l_a = fx.Float32(st[1])
                i = fx.Int32(tl)
                sync_tile()
                tstamp(i, 0)
                if i < nt:
                    m_a, l_a, alpha_a, pl_a = qk_softmax(base + (i % NB) * BUF, m_a, l_a, i * TT)
                    p_store(i & 1, pl_a, alpha_a)
                tstamp(i, 2)
                fin = yield [m_a, l_a]
            l_tot = _xsum(fx.Float32(fin[1]))
            _lds_store(Vec.from_elements([l_tot], fx.Float32), T.f32, base + L_OFF + (w * 16 + t) * 4, 4)
            has_l = l_tot > fx.Float32(0.0)
            lse2 = fx.Float32(fin[0]) + fx.Float32(fx.log2(has_l.select(l_tot, fx.Float32(1.0)))) - SHIFT
            lse = has_l.select(lse2 * LN2, fx.Float32(float("-inf")))
            lse_t.store(qi * HEADS + (hg0 + w) * H + t, lse.bitcast(fx.Int32))
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        def pv_role():
            pw = w - 4
            _, _, nt = row_bounds()
            vi = t // 2
            vtok = 16 * (vi // 4) + 4 * g + vi % 4
            fv = (4 * g + vi % 4) ^ (8 * (vi // 4))
            v_off = vtok * 512 + (((CPW * pw) ^ fv) << 4) + (t % 2) * 8
            init = [_raw(Vec.filled(4, 0.0, fx.Float32).ir_value()) for _ in range(NH * CPW)]
            for tl, st in range(0, nt + 1, 1, init=init):
                o = [Vec(st[k]) for k in range(NH * CPW)]
                i = fx.Int32(tl)
                sync_tile()
                tstamp(i, 3, 4)
                if i >= 1:
                    tt_ = i - 1
                    bT = base + (tt_ % NB) * BUF
                    pb = tt_ & 1
                    alphas = [
                        fx.Float32(
                            fx.ptr_load(_f32_ptr(base + AL_OFF + pb * AL_BUF + (hl * 16 + t) * 4), result_type=T.f32)
                        )
                        for hl in range_constexpr(NH)
                    ]
                    scv = []
                    for kk in range_constexpr(2):
                        vals = []
                        for sp in range_constexpr(2):
                            v4 = _lds_load(T.i32, 4, bT + SCL_OFF + (pw * 64 + 32 * kk + 16 * sp + 4 * g) * 4, 16)
                            for r in range_constexpr(4):
                                vals.append(_imin(fx.Int32(v4[r]), fx.Int32(SCALE_CAP_BITS)).bitcast(fx.Float32))
                        scv.append(Vec.from_elements(vals, fx.Float32).to(fx.Float16))
                    chg = alphas[0] != fx.Float32(1.0)
                    for hl in range_constexpr(1, NH):
                        chg = chg | (alphas[hl] != fx.Float32(1.0))
                    any_chg = rocdl.ballot(T.i64, _raw(chg.ir_value()))
                    if (tt_ > 0) & (fx.Int64(any_chg) != 0):
                        o = [
                            Vec.from_elements([fx.Float32(Vec(o[k])[j]) * alphas[k // CPW] for j in range(4)], fx.Float32)
                            for k in range(NH * CPW)
                        ]
                    for kk in range_constexpr(2):
                        pfk = []
                        for hl in range_constexpr(NH):
                            pf = _lds_load(T.i32, 4, base + P_OFF + pb * P_BUF + (hl * 2 + kk) * 1024 + lane * 16, 16)
                            pfk.append((Vec(pf).bitcast(fx.Float16) * scv[kk]).bitcast(fx.Int32))
                        rd = [_tr8(bT + (v_off ^ (d * 16)), kk * 16384) for d in range(CPW)]
                        for d in range_constexpr(CPW):
                            a = _cvt_v8(rd[d])
                            for hl in range_constexpr(NH):
                                o[hl * CPW + d] = _mfma_f16(a, pfk[hl], o[hl * CPW + d])
                tstamp(i, 4, 4)
                fin = yield o
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            for hl in range_constexpr(NH):
                lv = fx.Float32(fx.ptr_load(_f32_ptr(base + L_OFF + (hl * 16 + t) * 4), result_type=T.f32))
                inv_l = (lv > fx.Float32(0.0)).select(fx.Float32(1.0) / lv, fx.Float32(0.0))
                o_row = (qi * stride_o + ((hg0 + hl) * H + t) * stride_oh) >> 1
                for d in range_constexpr(CPW):
                    ov = Vec(fin[hl * CPW + d])
                    bf = Vec.from_elements([fx.Float32(ov[j]) * inv_l for j in range(4)], fx.Float32).to(fx.BFloat16)
                    o_t.store(
                        o_row + 8 * (CPW * pw + d) + 2 * g,
                        bf.bitcast(fx.Int32),
                        vec_size=2,
                    )
            stamp(10)

        def dma_role():
            r_beg, r_len, nt = row_bounds()
            raw0 = fx.Int32(idx_t[r_beg + lane])
            first_slot = _readfirstlane(raw0)
            # an empty row (CUDA-graph padding, or a rank with no keys) still issues its first DMA pass: slots[r_beg]
            # is then past the CSR end (stale / uninitialized), so read the always-valid record 0 instead
            first_slot = (r_len > 0).select(first_slot, fx.Int32(0))

            def ld_idx(p):
                ix = fx.Int32(idx_t[r_beg + p])
                return (p < r_len).select(ix, first_slot)

            ix0 = (lane < r_len).select(raw0, first_slot)
            ix1 = ld_idx(TT + lane)
            stamp(1)
            _guarded(nt > 0, lambda: dma_tile(ix0, base))
            stamp(2)
            init = [_raw(fx.Int32(ix1).ir_value())]
            for tl, st in range(0, nt + 1, 1, init=init):
                ixn = fx.Int32(st[0])
                i = fx.Int32(tl)
                sync_tile()
                ixnn = ld_idx((i + 2) * TT + lane)
                _guarded(
                    i + 1 < nt,
                    lambda _i=ixn, _b=base + ((i + 1) % NB) * BUF: dma_tile(_i, _b),
                )
                fin = yield [ixnn]
            gpu.barrier()

        _guarded(w < NH, qk_role)
        _guarded(w >= 4, pv_role)
        _guarded(w == DMA_WAVE, dma_role)

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
            grid=(fx.Int64(((n_q * SPLIT + XCDS - 1) // XCDS) * XCDS), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_decode.compile_hints = {"waves_per_eu": 1, "fast_fp_math": False}
    return launch_decode
