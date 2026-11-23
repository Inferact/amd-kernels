# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MQA decode attention over per-tensor e4m3 KV rows (64 query heads, head dim 512, FlyDSL, gfx950).

One 8-wave workgroup per query row (optionally one per (row, key range) with a last-arriver combine). All 64 heads
attend the row's ragged key list (main list, then extra list) in tiles of 64 keys; a key row is K and V (512 e4m3
bytes). Tiles are gathered by global -> LDS DMA (``global_load_async_lds``) into a 4-slot ring, 16-byte pieces
XOR-swizzled (``swz``) so the K reads (16 B) and the V^T reads (``ds_read_b64_tr_b8``) are bank-conflict free. Row
addresses are staged once in LDS.

Waves 0-3 run QK (32x32x64 MFMA, S^T = K Q^T, wave = 32 heads x 32 keys) and the softmax; waves 4-7 run PV (scaled
32x32x64 MFMA, dims 128 w..+127, all 64 heads). PV of tile c overlaps QK + softmax of tile c + 1; one barrier per tile.
P is e4m3 in two terms (hi + e4m3 residual) via the E8M0 B-scale operand, so it carries ~8 mantissa bits; the softmax
reference moves lazily (only when a block's max grows by more than ``LAZY`` log2), so most tiles skip the O rescale.
The sum is over the fp32 P, the attention sink joins it as a key without a value. The reference's moves enter the
P x V MX scale (E8M0, at most 127) and the fp32 accumulator, which cannot carry a jump of 100+ log2 units: a tile whose
exponent is more than ``EMAX`` above the reference restarts the accumulator at its own exponent (the old sum, smaller
by 2^-EMAX, is dropped); the QK waves flag it through ``alpha`` (one ulp below 1), so the PV waves take their slow path
only then.

The epilogue rounds O / sum to bf16, applies the inverse GPT-J RoPE to the last 64 dims and quantizes MXFP8 (E8M0
per 32 dims, row-major) in registers, or stores bf16 rows (``mxfp8=False``). No atomics touch outputs: the split
combine runs in a fixed split order.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

NW = 8  # waves per workgroup: 4 QK / softmax waves, 4 PV waves
BN = 64  # keys per tile
D = 512  # head dim = row bytes = V width
H = 64  # query heads
ROPE = 64  # trailing dims the inverse RoPE turns
TILE_B = BN * D
MAXK = 1024  # keys per row (both lists)
MAX_KEYS = MAXK
PPITCH = 80  # P row bytes: 64 + 16 pad (8 lanes' 16 B pieces on distinct banks)
LAZY = 4.0  # headroom and lazy-reference step (log2)
EMAX = 40.0  # a tile whose exponent is more than this above the reference restarts the accumulator at its own
PQ = math.log2(448.0) - LAZY  # P = 2^(s - M + PQ) <= 448 for the integer M >= the block's max score - LAZY
LOG2E = 1.4426950408889634
# kernel arguments preloaded into SGPRs (the first 64 bytes) instead of s_load'ed
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}
TINY_F32 = 1.1754943508222875e-38
AGENT_SCOPE = 0x10
UNIT_B = 131072 + 512  # split partials per (row, split): acc quads, then (m, l) of the 64 heads
STAGES = 4
MAX_SPLITS = 8
MAX_UNITS = 512  # (row, split) workgroups with partials


def _raw(v):
    return v.ir_value() if (not isinstance(v, _ir.Value) and hasattr(v, "ir_value")) else v


def _lds_ptr(addr):
    return _llvm.inttoptr(_ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(addr)))


def _g_ptr(addr):
    return _llvm.inttoptr(_ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr)))


def _ld128(addr):
    return Vec(_llvm.load(T.vec(4, T.i32), _lds_ptr(addr), alignment=16))


def _st128(addr, v):
    _llvm.store(_raw(v), _lds_ptr(addr), alignment=16)


def _ld64(addr):
    return Vec(_llvm.load(T.vec(2, T.f32), _lds_ptr(addr), alignment=8))


def _st64(addr, v):
    _llvm.store(_raw(v), _lds_ptr(addr), alignment=8)


def _tr8(addr):
    return Vec(_rocdl_ods.ds_read_tr8_b64(T.vec(2, T.i32), _lds_ptr(addr)).result)


def _ld_f32(addr):
    return fx.Float32(_llvm.load(T.f32, _lds_ptr(addr), alignment=4))


def _st_f32(addr, v):
    _llvm.store(_raw(fx.Float32(v)), _lds_ptr(addr), alignment=4)


def _ld_inv_i32(addr):
    """A uniform, read-only i32 (an ``s_load``)."""
    p = _llvm.inttoptr(_ir.Type.parse("!llvm.ptr<4>"), _raw(fx.Int64(addr)))
    return fx.Int32(_llvm.LoadOp(T.i32, p, alignment=4, invariant=True).result)


def _ld_inv_f32(addr):
    p = _llvm.inttoptr(_ir.Type.parse("!llvm.ptr<4>"), _raw(fx.Int64(addr)))
    return fx.Float32(_llvm.LoadOp(T.f32, p, alignment=4, invariant=True).result)


def _global_atomic_add(addr_i64, value):
    """``old = *addr; *addr += value`` on a global i32 (agent scope, relaxed)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _g_ptr(addr_i64),
            fx.Int32(value).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def _barrier():
    _llvm.inline_asm(None, [], "s_waitcnt lgkmcnt(0)\ns_barrier", "~{memory}", has_side_effects=True)


def _waitcnt_vm(n):
    _llvm.inline_asm(None, [], f"s_waitcnt vmcnt({n})", "~{memory}", has_side_effects=True)


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _permlane_swap(op, d_a, d_b):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False)
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(_llvm.extractvalue(T.i32, res, [1]))


def _maxf_nn(a, b):
    fm = _ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result)


def _xlane4(x, op):
    """``op`` over lanes L, L ^ 16, L ^ 32, L ^ 48, broadcast to all four."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(x))
    m = op(_as_f32(a), _as_f32(b))
    a, b = _permlane_swap(_rocdl.permlane16_swap, _bits(m), _bits(m))
    return op(_as_f32(a), _as_f32(b))


def _valu(asm, *ops):
    """One unpacked fp32 VALU op as inline asm (LLVM pairs plain fp32 ops into v_pk_*, which cannot issue while the
    other wave's MFMA is in flight)."""
    return fx.Float32(
        _llvm.inline_asm(
            T.f32, [fx.as_ir_value(fx.Float32(o)) for o in ops], asm, "=v" + ",v" * len(ops), has_side_effects=False
        )
    )


def _afma(a, b, c):
    return _valu("v_fma_f32 $0, $1, $2, $3", a, b, c)


def _aadd(a, b):
    return _valu("v_add_f32 $0, $1, $2", a, b)


def _asub(a, b):
    return _valu("v_sub_f32 $0, $1, $2", a, b)


def _amul(a, b):
    return _valu("v_mul_f32 $0, $1, $2", a, b)


def _aadd_t(a, b):
    """``_aadd`` of a value a ``v_exp_f32`` may have just written (the trans forwarding hazard)."""
    return _valu("s_nop 0\n\tv_add_f32 $0, $1, $2", a, b)


def _asub_t(a, b):
    return _valu("s_nop 0\n\tv_sub_f32 $0, $1, $2", a, b)


def _afma_t(a, b, c):
    return _valu("s_nop 0\n\tv_fma_f32 $0, $1, $2, $3", a, b, c)


def _xlane2(x, op):
    """``op`` over lanes L and L ^ 32."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(x))
    return op(_as_f32(a), _as_f32(b))


def _exp2(x):
    return fx.Float32(_rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


def _pk_fp8(a, b, old, hi):
    return fx.Int32(
        _rocdl_ods.cvt_pk_fp8_f32(
            T.i32,
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Int32(old)),
            hi,
        )
    )


def _pack4_fp8(x0, x1, x2, x3):
    """Four fp32 -> a dword of four e4m3 bytes, ``x0`` the lowest."""
    lo = _pk_fp8(x0, x1, fx.Int32(_llvm.mlir_undef(T.i32)), False)
    return _pk_fp8(x2, x3, lo, True)


def _unpack2_fp8(word, hi):
    v = Vec(_rocdl.cvt_pk_f32_fp8(Vec.make_type(2, fx.Float32), fx.as_ir_value(word), hi))
    return v[0], v[1]


def _pk_bf16(a, b):
    """Two fp32 -> a dword of two bf16 (RNE), ``a`` low."""
    return fx.Int32(
        Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


def _mfma16(a8, b8, acc4):
    """v_mfma_scale_f32_16x16x128_f8f6f4, e4m3 x e4m3, unit scales. Lane (i, q) holds A[i][32 q + e], B[32 q + e][i],
    e = 0..31 (8 dwords), and gets D[4 q + r][i]."""
    zero = fx.as_ir_value(fx.Int32(0))
    e4m3 = _ir.Attribute.parse("#rocdl<matrix_format fp8_e4m3>")
    sel = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 0)
    return Vec(
        _rocdl_ods.mfma_scale_f32_16x16x128_f8f6f4(
            Vec.make_type(4, fx.Float32),
            fx.as_ir_value(a8), fx.as_ir_value(b8), fx.as_ir_value(acc4),
            e4m3, e4m3, sel, zero, sel, zero,
        ).result
    )  # fmt: skip


def _mfma32(a8, b8, acc16):
    """v_mfma_scale_f32_32x32x64_f8f6f4: lane (i, h) holds A[i][32 h + e], B[32 h + e][i] (8 dwords) and gets
    D[8 (r // 4) + 4 h + r % 4][i] in register r."""
    zero = fx.as_ir_value(fx.Int32(0))
    e4m3 = _ir.Attribute.parse("#rocdl<matrix_format fp8_e4m3>")
    sel = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 0)
    return Vec(
        _rocdl_ods.mfma_scale_f32_32x32x64_f8f6f4(
            Vec.make_type(16, fx.Float32),
            fx.as_ir_value(a8), fx.as_ir_value(b8), fx.as_ir_value(acc16),
            e4m3, e4m3, sel, zero, sel, zero,
        ).result
    )  # fmt: skip


def _mfma32s(a8, b8, acc16, sc_b):
    """``_mfma32`` with the E8M0 scale of the B operand's K block: lane (j, h) gives the scale (low byte of ``sc_b``)
    of block h of column j; a block is bytes 16 h' .. 16 h' + 15 of the lanes' 32 operand bytes (both lane halves).
    A has unit scales."""
    e4m3 = _ir.Attribute.parse("#rocdl<matrix_format fp8_e4m3>")
    sel = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 0)
    return Vec(
        _rocdl_ods.mfma_scale_f32_32x32x64_f8f6f4(
            Vec.make_type(16, fx.Float32),
            fx.as_ir_value(a8), fx.as_ir_value(b8), fx.as_ir_value(acc16),
            e4m3, e4m3, sel, fx.as_ir_value(fx.Int32(127)), sel, fx.as_ir_value(sc_b),
        ).result
    )  # fmt: skip


def _cat8(lo4, hi4):
    return Vec.from_elements([lo4[k] for k in range(4)] + [hi4[k] for k in range(4)], fx.Int32)


def swz(kappa):
    """The 16-byte piece index of a row is XORed with this: bits 0-1 of the row, bit 2 = row bit 2 ^ row bit 4."""
    return (kappa & 3) | ((((kappa >> 2) ^ (kappa >> 4)) & 1) << 2)


def compile_sparse_mla_decode(
    stages: int = 4, p_terms: int = 2, mxfp8: bool = True, has_extra: bool = True, splitk: bool = False
):
    """``stages``: LDS ring slots (32 KB each). ``p_terms``: 2 = P as hi + residual e4m3 (cosine > 0.99999 to the fp64
    reference), 1 = one e4m3 P. ``mxfp8``: the inverse-RoPE MXFP8 output, else bf16 rows. ``has_extra``: the extra
    key list. ``splitk``: a row's tiles split over ``i32_S`` workgroups whose partials the last one combines."""
    NS = stages
    assert NS >= 3 and p_terms in (1, 2)
    P_BUF = p_terms * H * PPITCH
    DMA_PER_WAVE = 8  # 1 KB DMA instructions per PV wave and tile
    name = (
        f"sparse_mla_decode_fp8kv_h64_d512_s{NS}_p{p_terms}"
        + ("_mxfp8" if mxfp8 else "_bf16")
        + ("_x" if has_extra else "")
        + ("_sk" if splitk else "")
    )

    @fx.struct
    class Shared:
        kv: fx.Array[fx.Uint8, NS * TILE_B, 16]
        p: fx.Array[fx.Uint8, 2 * P_BUF, 16]
        mu: fx.Array[fx.Float32, 2 * 2 * H]
        pr: fx.Array[fx.Int32, 2 * 2 * H * 2]
        fin: fx.Array[fx.Float32, 3 * H]
        ids: fx.Array[fx.Int64, MAXK]
        tk: fx.Array[fx.Int32, 4]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [T, 64, 512] e4m3
        arg_main: fx.Int64,  # main cache, e4m3 rows
        arg_xtra: fx.Int64,  # extra cache
        arg_mip: fx.Int64,  # [T + 1] int32
        arg_mid: fx.Int64,  # int32 slots
        arg_xip: fx.Int64,
        arg_xid: fx.Int64,
        arg_sink: fx.Int64,  # [64] fp32
        arg_pos: fx.Int64,  # [T] int32
        arg_cs: fx.Int64,  # [P, 64] fp32 cos | sin
        arg_out: fx.Int64,  # mxfp8: e4m3 [T, 32768]; else bf16 [T, 64, 512]
        arg_osc: fx.Int64,  # mxfp8: e8m0 [T, 1024]
        arg_qs: fx.Int64,  # fp32 q scale
        arg_ks: fx.Int64,  # fp32 kv scale
        arg_part: fx.Int64,  # splitk: partials, 128 KiB + 512 B per (row, split)
        arg_cnt: fx.Int64,  # splitk: [T] int32 tickets, zero
        i32_T: fx.Int32,
        i32_rows_x: fx.Int32,  # rows per XCD
        i32_S: fx.Int32,  # splits per row
        i32_m_lg: fx.Int32,  # main cache: log2 block, block stride in rows
        i32_m_st: fx.Int32,
        i32_x_lg: fx.Int32,
        i32_x_st: fx.Int32,
        i32_cs_st: fx.Int32,  # cos_sin row stride (elements)
        i32_pos_b: fx.Int32,  # bytes between positions (4: int32, 8: int64)
        f32_sm: fx.Float32,  # softmax scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bx = fx.block_idx.x
        lane = tx % 64
        w8 = fx.Int32(_rocdl.readfirstlane(T.i32, tx // 64))
        isq = w8 < 4
        l32, hh = lane % 32, lane // 32
        neg_inf = fx.Float32(float("-inf"))
        zero16 = Vec.filled(16, 0.0, fx.Float32)

        xcd = bx % 8
        j = bx // 8
        if const_expr(splitk):
            row = xcd * i32_rows_x + j // i32_S
            split = j % i32_S
        else:
            row = xcd * i32_rows_x + j
            split = fx.Int32(0)
        live = row < i32_T
        rowc = live.select(row, fx.Int32(0))

        lds_kv = fx.Int32(fx.ptrtoint(smem.kv.ptr))
        lds_p = fx.Int32(fx.ptrtoint(smem.p.ptr))
        lds_mu = fx.Int32(fx.ptrtoint(smem.mu.ptr))
        lds_pr = fx.Int32(fx.ptrtoint(smem.pr.ptr))
        lds_fin = fx.Int32(fx.ptrtoint(smem.fin.ptr))
        lds_ids = fx.Int32(fx.ptrtoint(smem.ids.ptr))
        lds_tk = fx.Int32(fx.ptrtoint(smem.tk.ptr))

        def uni(v):
            return fx.Int32(_rocdl.readfirstlane(T.i32, v))

        ms = uni(_ld_inv_i32(arg_mip + fx.Int64(rowc) * 4))
        n_main = uni(_ld_inv_i32(arg_mip + fx.Int64(rowc) * 4 + 4)) - ms
        if const_expr(has_extra):
            xs = uni(_ld_inv_i32(arg_xip + fx.Int64(rowc) * 4))
            n_x = uni(_ld_inv_i32(arg_xip + fx.Int64(rowc) * 4 + 4)) - xs
        else:
            xs = fx.Int32(0)
            n_x = fx.Int32(0)
        n_row = n_main + n_x
        nt_row = (n_row + (BN - 1)) // BN
        if const_expr(splitk):
            per = (nt_row + (i32_S - 1)) // i32_S
            per = (per > 0).select(per, fx.Int32(1))
            c0 = split * per
            c1 = (c0 + per < nt_row).select(c0 + per, nt_row)
            nt = (c1 > c0).select(c1 - c0, fx.Int32(0))
        else:
            per = nt_row
            c0 = fx.Int32(0)
            nt = nt_row
        nt = live.select(nt, fx.Int32(0))
        kbase = c0 * BN
        n_unit = ((n_row < c0 * BN + nt * BN).select(n_row, c0 * BN + nt * BN)) - kbase
        n_unit = (nt > 0).select(n_unit, fx.Int32(0))
        kmax = (n_unit > 0).select(n_unit - 1, fx.Int32(0))
        qs = _ld_inv_f32(arg_qs)
        ks = _ld_inv_f32(arg_ks)
        c_sc = f32_sm * qs * ks

        aux0 = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 0)

        # the cache row address of every list position of the unit, staged in LDS
        for k in range_constexpr((MAXK + 64 * NW - 1) // (64 * NW)):
            kp = tx + 64 * NW * k
            kpg = kbase + ((kp < n_unit).select(kp, kmax))
            if const_expr(has_extra):
                in_main = kpg < n_main
                sptr = in_main.select(arg_mid + fx.Int64(ms + kpg) * 4, arg_xid + fx.Int64(xs + kpg - n_main) * 4)
                lg = in_main.select(i32_m_lg, i32_x_lg)
                st = in_main.select(i32_m_st, i32_x_st)
                base = in_main.select(fx.Int64(arg_main), fx.Int64(arg_xtra))
            else:
                sptr = arg_mid + fx.Int64(ms + kpg) * 4
                lg, st, base = i32_m_lg, i32_m_st, fx.Int64(arg_main)
            slot = fx.Int32(_llvm.load(T.i32, _g_ptr(sptr), alignment=4))
            rowi = (slot >> lg) * st + (slot & ((fx.Int32(1) << lg) - 1))
            if kp < MAXK:
                _llvm.store(_raw(base + fx.Int64(rowi) * D), _lds_ptr(lds_ids + kp * 8), alignment=8)
        _barrier()

        # ---------------- QK waves: heads 32 hg .. + 31, keys 32 kg .. + 31 of every tile ----------------
        wq = w8 % 4
        hg, kg = wq % 2, wq // 2
        # K^T operand: lane (i = l32, hh) reads bytes 64 s + 32 hh .. + 31 of tile row kap(i): the 32 MFMA rows i = 8 a +
        # 4 b + c are the tile keys 32 kg + 16 b + 4 a + c, so lane (head, hh) gets the keys 32 kg + 16 hh + r in D[r]
        kap = 32 * kg + 16 * ((l32 >> 2) & 1) + 4 * (l32 >> 3) + (l32 & 3)
        fk = swz(kap)
        koff = [
            [kap * D + (((4 * sb + 2 * hh + hf) ^ fk) * 16) for hf in range_constexpr(2)] for sb in range_constexpr(2)
        ]
        head_q = 32 * hg + l32
        pw_off = [(term * H + head_q) * PPITCH + 32 * kg + 16 * hh for term in range_constexpr(p_terms)]

        def qk(slot, qfrag):
            base = lds_kv + slot * TILE_B
            ka = []
            for s in range_constexpr(8):
                lo = _ld128(base + koff[s % 2][0] + 128 * (s // 2))
                hi = _ld128(base + koff[s % 2][1] + 128 * (s // 2))
                ka.append(_cat8(lo, hi))
            _rocdl.sched_barrier(0)
            S = zero16
            for s in range_constexpr(8):
                S = _mfma32(ka[s], qfrag[s], S)
            return S

        def masked(S, c):
            sv = [S[r] for r in range_constexpr(16)]
            rem = n_unit - c * BN - 32 * kg - 16 * hh
            if c == nt - 1:
                sv = [(fx.Int32(r) < rem).select(sv[r], neg_inf) for r in range_constexpr(16)]
            return sv

        def maxchain(sv):
            t8 = [_maxf_nn(sv[2 * i], sv[2 * i + 1]) for i in range_constexpr(8)]
            t4 = [_maxf_nn(t8[2 * i], t8[2 * i + 1]) for i in range_constexpr(4)]
            t2 = [_maxf_nn(t4[0], t4[1]), _maxf_nn(t4[2], t4[3])]
            return _xlane2(_maxf_nn(t2[0], t2[1]), _maxf_nn)

        def exps(sv, m_use):
            msub = (m_use > neg_inf).select(m_use - PQ, fx.Float32(0.0))
            nms = -msub
            return [_exp2(_afma(x, c_sc, nms)) for x in sv]

        def sum16(ps):
            tot = [_aadd_t(ps[i], ps[i + 8]) for i in range_constexpr(8)]
            tot = [_aadd(tot[i], tot[i + 4]) for i in range_constexpr(4)]
            tot = [_aadd(tot[0], tot[2]), _aadd(tot[1], tot[3])]
            return _xlane2(_aadd(tot[0], tot[1]), _aadd)

        def write_p(ps, buf):
            ph = [_pack4_fp8(ps[4 * t], ps[4 * t + 1], ps[4 * t + 2], ps[4 * t + 3]) for t in range_constexpr(4)]
            pbase = lds_p + buf * P_BUF
            _st128(pbase + pw_off[0], Vec.from_elements(ph, fx.Int32))
            if const_expr(p_terms == 2):
                pl = []
                for t in range_constexpr(4):
                    h01 = _unpack2_fp8(ph[t], False)
                    h23 = _unpack2_fp8(ph[t], True)
                    pl.append(
                        _pack4_fp8(
                            _asub_t(ps[4 * t], h01[0]),
                            _asub_t(ps[4 * t + 1], h01[1]),
                            _asub_t(ps[4 * t + 2], h23[0]),
                            _asub_t(ps[4 * t + 3], h23[1]),
                        )
                    )
                _st128(pbase + pw_off[1], Vec.from_elements(pl, fx.Int32))

        def mu_addr(buf, kgi):
            return lds_mu + (((buf * 2 + kgi) * H + head_q) * 4)

        def softmax_first(S):
            """Tile 0: the block's own exponent (max score - LAZY, rounded up): P, its max exponent, (R, F, L)."""
            sv = masked(S, fx.Int32(0))
            mx = maxchain(sv)
            m_use = fx.math.ceil(mx * c_sc - LAZY)
            ps = exps(sv, m_use)
            lsum = sum16(ps)
            write_p(ps, fx.Int32(0))
            _st_f32(mu_addr(fx.Int32(0), kg), m_use)
            return neg_inf, m_use, lsum

        def softmax_next(S, c, buf, state):
            """Tile c >= 1: P relative to the reference R_c of the earlier tiles (the stats of tile c - 1), recomputed
            with the block's own exponent only where a score is more than LAZY above it."""
            R, F, L = state
            pb = (c - 1) % 2
            m0 = _ld_f32(mu_addr(pb, 0))
            m1 = _ld_f32(mu_addr(pb, 1))
            mmax = _maxf_nn(m0, m1)
            Rref = (mmax - R > EMAX).select(mmax, R)  # the accumulator restarts at a tile whose exponent jumped
            Rn = (mmax > Rref + LAZY).select(mmax, Rref)
            alpha = _exp2(Rref - Rn)
            sv = masked(S, c)
            ps = exps(sv, Rn)
            umax = maxchain(sv) * c_sc
            ovf = umax > Rn + LAZY
            m_use = Rn
            if fx.Int64(_rocdl.ballot(T.i64, ovf)) != fx.Int64(0):
                m_use = ovf.select(fx.math.ceil(umax - LAZY), Rn)
                ps = exps(sv, m_use)
            lsum = sum16(ps)
            write_p(ps, buf)
            x = m_use - Rn
            _st64(
                lds_pr + (((buf * 2 + kg) * H + head_q) * 2) * 4,
                Vec.from_elements([x.to(fx.Int32), alpha.bitcast(fx.Int32)], fx.Int32),
            )
            _st_f32(mu_addr(buf, kg), m_use)
            Fn = (x > EMAX).select(m_use, Rn)
            L_new = fx.math.fma(lsum, _exp2(m_use - Fn), L * _exp2(F - Fn))
            return Rn, Fn, L_new

        # ---------------- PV waves: dims 128 wp .. + 127 of all heads ----------------
        wp = w8 - 4
        vk = (lane % 16) // 2
        vb4 = (lane // 16) % 2
        fv = (vk & 3) | ((((vk >> 2) ^ hh) & 1) << 2)
        voff = [
            (16 * hh + vk) * D + ((8 * wp + ((2 * d + vb4) ^ fv)) * 16) + 8 * (lane % 2) for d in range_constexpr(4)
        ]
        pr_off = [
            [(term * H + 32 * ht + l32) * PPITCH + 16 * hh for term in range_constexpr(p_terms)]
            for ht in range_constexpr(2)
        ]
        dma_off = [((lane % 32) ^ swz(16 * wp + 2 * jj + hh)) * 16 for jj in range_constexpr(DMA_PER_WAVE)]
        kp_off = [16 * wp + 2 * jj + hh for jj in range_constexpr(DMA_PER_WAVE)]

        def dma_prep(c):
            """The tile's 8 row addresses of this lane, loaded from LDS."""
            addrs = []
            for jj in range_constexpr(DMA_PER_WAVE):
                kp = c * BN + kp_off[jj]
                kp = (kp > MAXK - 1).select(fx.Int32(MAXK - 1), kp)
                addrs.append(fx.Int64(_llvm.load(T.i64, _lds_ptr(lds_ids + kp * 8), alignment=8)))
            return addrs

        def dma_step(addrs, jj, slot):
            a = addrs[jj] + fx.Int64(dma_off[jj])
            dst = uni(lds_kv + slot * TILE_B + (8 * wp + jj) * 1024)
            _rocdl_ods.global_load_async_lds(_g_ptr(a), _lds_ptr(dst), 16, 0, aux=aux0)

        def issue_dma(c, slot):
            addrs = dma_prep(c)
            for jj in range_constexpr(DMA_PER_WAVE):
                dma_step(addrs, jj, slot)

        def wait_dma(later):
            """Own DMAs landed except the ``later`` newest tiles'."""
            _waitcnt_vm(DMA_PER_WAVE * later)

        def pv_mm(slot, buf, acc, scales, fill):
            """O^T += V^T P^T of one tile; ``scales``: the lane's e8m0 byte of each head tile."""
            base = lds_kv + slot * TILE_B
            pbase = lds_p + buf * P_BUF
            bf = []
            for ht in range_constexpr(2):
                for term in range_constexpr(p_terms):
                    lo = _ld128(pbase + pr_off[ht][term])
                    hi = _ld128(pbase + pr_off[ht][term] + 32)
                    bf.append(_cat8(lo, hi))
            out = []
            for d in range_constexpr(4):
                parts = []
                for u in range_constexpr(4):
                    r = _tr8(base + voff[d] + (u // 2) * 16384 + (u % 2) * 4096)
                    parts += [r[0], r[1]]
                a8 = Vec.from_elements(parts, fx.Int32)
                for ht in range_constexpr(2):
                    x = acc[2 * d + ht]
                    for term in range_constexpr(p_terms):
                        x = _mfma32s(a8, bf[ht * p_terms + term], x, scales[ht])
                        if fill is not None:
                            fill()
                    out.append(x)
            return out

        def e8m0i(e):
            byte = e + 127
            byte = (byte > 254).select(fx.Int32(254), byte)
            return (byte < 0).select(fx.Int32(0), byte)

        def e8m0(e):
            return e8m0i(e.to(fx.Int32))

        def pv_first(slot, buf, fill=None):
            """Tile 0 (acc = 0): the frame is the larger of the two blocks' exponents."""
            scales = []
            for ht in range_constexpr(2):
                m0 = _ld_f32(lds_mu + ((buf * 2 + 0) * H + 32 * ht + l32) * 4)
                m1 = _ld_f32(lds_mu + ((buf * 2 + 1) * H + 32 * ht + l32) * 4)
                e = (hh > 0).select(m1, m0) - _maxf_nn(m0, m1)
                scales.append(e8m0((e > fx.Float32(-200.0)).select(e, fx.Float32(-200.0))))
            return pv_mm(slot, buf, init_acc, scales, fill)

        def pv_next(slot, buf, acc, fill=None):
            """Tile c >= 1: the e8m0 byte and the rescale factor of the QK wave of this lane's block."""
            scales, alphas = [], []
            for ht in range_constexpr(2):
                rec = Vec(_llvm.load(T.vec(2, T.i32), _lds_ptr(lds_pr + (((buf * 2 + hh) * H + 32 * ht + l32) * 2) * 4), alignment=8))
                x_o = fx.Int32(_llvm.load(T.i32, _lds_ptr(lds_pr + (((buf * 2 + (1 - hh)) * H + 32 * ht + l32) * 2) * 4), alignment=4))
                x = fx.Int32(rec[0])
                xm = (x > x_o).select(x, x_o)
                big = xm > fx.Int32(int(EMAX))
                scales.append(e8m0i(big.select(x - xm, x)))
                alphas.append(big.select(fx.Float32(0.0), fx.Int32(rec[1]).bitcast(fx.Float32)))
            need = (alphas[0] < fx.Float32(1.0)) | (alphas[1] < fx.Float32(1.0))
            acc_s = acc
            if fx.Int64(_rocdl.ballot(T.i64, need)) != fx.Int64(0):
                acc_s = [
                    Vec.from_elements([_amul(acc[2 * d + ht][e], alphas[ht]) for e in range_constexpr(16)], fx.Float32)
                    for d in range_constexpr(4)
                    for ht in range_constexpr(2)
                ]
            return pv_mm(slot, buf, acc_s, scales, fill)

        def finalize(acc_f, m_f, l_f):
            """O / (sum + sink) of this PV wave's dims, as bf16 rows or inverse-RoPE MXFP8 (module doc)."""
            # O^T lane (head 32 ht + l32, hh): register r of tile d is dim 128 wp + 32 d + 8 (r // 4) + 4 hh + r % 4
            rv = []
            for ht in range_constexpr(2):
                rv.append(ks / (l_f[ht] + _exp2(sink2[ht] - m_f[ht] + PQ)))
            if const_expr(not mxfp8):
                orr = buffer_ops.create_buffer_resource_from_addr(
                    arg_out, num_records_bytes=fx.Int64(i32_T) * fx.Int64(H * D * 2)
                )
                for d in range_constexpr(4):
                    for ht in range_constexpr(2):
                        x = acc_f[2 * d + ht]
                        head = 32 * ht + l32
                        for a in range_constexpr(4):
                            lo = _pk_bf16(x[4 * a] * rv[ht], x[4 * a + 1] * rv[ht])
                            hi = _pk_bf16(x[4 * a + 2] * rv[ht], x[4 * a + 3] * rv[ht])
                            dim = 128 * wp + 32 * d + 8 * a + 4 * hh
                            off = ((rowc * H + head) * D + dim) // 2
                            buffer_ops.buffer_store(Vec.from_elements([lo, hi], fx.Int32), orr, off)
            else:
                ocr = buffer_ops.create_buffer_resource_from_addr(
                    arg_out, num_records_bytes=fx.Int64(i32_T) * fx.Int64(H * D)
                )
                osr = buffer_ops.create_buffer_resource_from_addr(
                    arg_osc, num_records_bytes=fx.Int64(i32_T) * fx.Int64(H * D // 32)
                )
                is_rope = wp == 3
                # the codes are staged in this wave's piece of the (free) key ring, 144-byte rows of 128 codes, and
                # stored as 128-byte runs; the scale bytes of a head's four blocks leave as one dword
                lds_o = lds_kv + wp * (H * 144)
                sbytes = [[None] * 4 for _ in range_constexpr(2)]
                for d in range_constexpr(4):
                    if const_expr(d >= 2):
                        cs = cs_all[d - 2]
                    for ht in range_constexpr(2):
                        x = acc_f[2 * d + ht]
                        head = 32 * ht + l32
                        packs = [_pk_bf16(x[2 * k] * rv[ht], x[2 * k + 1] * rv[ht]) for k in range_constexpr(8)]
                        vals = []
                        for k in range_constexpr(8):
                            vals += [_as_f32(packs[k] << 16), _as_f32(packs[k] & fx.Int32(-65536))]
                        if const_expr(d >= 2):
                            rot = []
                            for b in range_constexpr(8):
                                cb, sb_ = cs[b // 2][b % 2], cs[b // 2][2 + b % 2]
                                ev, od = vals[2 * b], vals[2 * b + 1]
                                rot += [fx.math.fma(ev, cb, od * sb_), fx.math.fma(od, cb, -(ev * sb_))]
                            vals = [is_rope.select(rot[i], vals[i]) for i in range_constexpr(16)]
                        amax = fx.Float32(TINY_F32)
                        for i in range_constexpr(16):
                            amax = _maxf_nn(amax, _as_f32(_bits(vals[i]) & fx.Int32(0x7FFFFFFF)))
                        amax = _xlane2(amax, _maxf_nn)
                        sbf = fx.math.ceil(fx.math.log2(amax / fx.Float32(448.0))) + fx.Float32(127.0)
                        sbf = fx.arith.minimumf(fx.arith.maximumf(sbf, fx.Float32(0.0)), fx.Float32(254.0))
                        rs = fx.Float32(fx.math.exp2(fx.Float32(127.0) - sbf))
                        for a in range_constexpr(4):
                            dw = _pack4_fp8(
                                vals[4 * a] * rs, vals[4 * a + 1] * rs, vals[4 * a + 2] * rs, vals[4 * a + 3] * rs
                            )
                            _llvm.store(
                                _raw(dw), _lds_ptr(lds_o + head * 144 + 32 * d + 8 * a + 4 * hh), alignment=4
                            )
                        sbytes[ht][d] = sbf.to(fx.Int32)
                _llvm.inline_asm(None, [], "s_waitcnt lgkmcnt(0)", "~{memory}", has_side_effects=True)
                for i in range_constexpr(8):
                    hrow = lane // 8 + 8 * i
                    v = _ld128(lds_o + hrow * 144 + (lane % 8) * 16)
                    buffer_ops.buffer_store(v, ocr, ((rowc * H + hrow) * D + 128 * wp + (lane % 8) * 16) // 4)
                for ht in range_constexpr(2):
                    dws = sbytes[ht][0] | (sbytes[ht][1] << 8) | (sbytes[ht][2] << 16) | (sbytes[ht][3] << 24)
                    buffer_ops.buffer_store(
                        dws, osr, (rowc * (H * D // 32) + (32 * ht + l32) * (D // 32) + 4 * wp) // 4, mask=hh == 0
                    )

        # split partials of a (row, split) unit: acc quads [wp][quad 32][lane][4 fp32], then (m, l) per head
        def part_acc_off(uid, qd):
            return ((uid * UNIT_B) // 16 + (wp * 32 + qd) * 64 + lane) * 4  # dwords

        def part_stat_off(uid, head):
            return (uid * UNIT_B + 131072 + head * 8) // 4

        def prsrc():
            return buffer_ops.create_buffer_resource_from_addr(arg_part, num_records_bytes=0xFFFFFFFF)

        # the epilogue's inputs (PV waves), loaded before the key loop so that their latency is not in the tail
        zero_cs = [(fx.Float32(0.0),) * 4 for _ in range_constexpr(4)]
        sink2 = [fx.Float32(0.0), fx.Float32(0.0)]
        cs_all = [zero_cs, zero_cs]
        if not isq:
            sink2 = [
                fx.Float32(_llvm.load(T.f32, _g_ptr(arg_sink + fx.Int64(32 * ht + l32) * 4), alignment=4)) * LOG2E
                for ht in range_constexpr(2)
            ]
            if const_expr(mxfp8):
                pos = uni(_ld_inv_i32(arg_pos + fx.Int64(rowc) * i32_pos_b))
                cs_row = arg_cs + fx.Int64(pos) * fx.Int64(i32_cs_st) * 4
                cs_all = []
                for d in range_constexpr(2):
                    row_cs = []
                    for a in range_constexpr(4):
                        pidx = 16 * d + 4 * a + 2 * hh
                        cv = Vec(_llvm.load(T.vec(2, T.f32), _g_ptr(cs_row + fx.Int64(pidx) * 4), alignment=8))
                        sv = Vec(_llvm.load(T.vec(2, T.f32), _g_ptr(cs_row + fx.Int64(32 + pidx) * 4), alignment=8))
                        row_cs.append((fx.Float32(cv[0]), fx.Float32(cv[1]), fx.Float32(sv[0]), fx.Float32(sv[1])))
                    cs_all.append(row_cs)
        init_acc = [zero16 for _ in range_constexpr(8)]
        m_f = [neg_inf, neg_inf]
        l_f = [fx.Float32(0.0), fx.Float32(0.0)]
        acc_f = init_acc
        if isq:
            if nt > 0:
                qr = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_T) * fx.Int64(H * D)
                )
                q_dw = (rowc * H + head_q) * (D // 4) + 8 * hh
                qfrag = []
                for s_ in range_constexpr(8):
                    lo = buffer_ops.buffer_load(qr, q_dw + 16 * s_, vec_width=4, dtype=fx.Int32)
                    hi = buffer_ops.buffer_load(qr, q_dw + 16 * s_ + 4, vec_width=4, dtype=fx.Int32)
                    qfrag.append(_cat8(Vec(lo), Vec(hi)))
                _barrier()
                S0 = qk(fx.Int32(0), qfrag)
                st0 = softmax_first(S0)
                _barrier()
                init = list(st0)
                results = [fx.as_ir_value(x) for x in init]
                for it, st_ in range(  # type: ignore[call-overload]
                    fx.Index(1), fx.Index(nt), fx.Index(1), init=[fx.as_ir_value(x) for x in init]
                ):
                    c = fx.Int32(it)
                    S = qk(c % NS, qfrag)
                    nst = softmax_next(S, c, c % 2, tuple(fx.Float32(st_[i]) for i in range(3)))
                    _barrier()
                    results = yield [fx.as_ir_value(x) for x in nst]
                R_, F_, L_ = (fx.Float32(results[i]) for i in range(3))
                pb = (nt - 1) % 2
                m0 = _ld_f32(mu_addr(pb, 0))
                m1 = _ld_f32(mu_addr(pb, 1))
                mm = _maxf_nn(m0, m1)
                Rl = (mm - R_ > EMAX).select(mm, R_)
                Lf = L_ * _exp2(F_ - Rl)
                _st_f32(lds_fin + (kg * H + head_q) * 4, Lf)
                _st_f32(lds_fin + (2 * H + head_q) * 4, Rl)
            _barrier()
        else:
            if nt > 0:
                for c0_ in range_constexpr(NS - 1):
                    issue_dma(fx.Int32(c0_), fx.Int32(c0_))
                wait_dma(NS - 2)
                _barrier()
                wait_dma(NS - 3)
                _barrier()
                acc_f = init_acc
                if nt > 1:
                    # iteration 1: PV of tile 0
                    addrs = dma_prep(fx.Int32(NS - 1))
                    cnt = [0]

                    def fill1():
                        k = cnt[0]
                        cnt[0] += 1
                        if k >= 2 and k - 2 < DMA_PER_WAVE:
                            dma_step(addrs, k - 2, fx.Int32((NS - 1) % NS))
                        _rocdl.sched_barrier(0)

                    acc_1 = pv_first(fx.Int32(0), fx.Int32(0), fill1)
                    for _ in range_constexpr(DMA_PER_WAVE + 2 - cnt[0]):
                        fill1()
                    wait_dma(NS - 3)
                    _barrier()
                    results = [fx.as_ir_value(x) for x in acc_1]
                    for it, st_ in range(  # type: ignore[call-overload]
                        fx.Index(2), fx.Index(nt), fx.Index(1), init=[fx.as_ir_value(x) for x in acc_1]
                    ):
                        acc_t = [Vec(st_[i]) for i in range_constexpr(8)]
                        c = fx.Int32(it)
                        addrs = dma_prep(c + (NS - 2))
                        dslot = (c + (NS - 2)) % NS
                        cnt = [0]

                        def fill():
                            k = cnt[0]
                            cnt[0] += 1
                            if k >= 2 and k - 2 < DMA_PER_WAVE:
                                dma_step(addrs, k - 2, dslot)
                            _rocdl.sched_barrier(0)

                        acc_n = pv_next((c - 1) % NS, (c - 1) % 2, acc_t, fill)
                        for _ in range_constexpr(DMA_PER_WAVE + 2 - cnt[0]):
                            fill()
                        wait_dma(NS - 3)
                        _barrier()
                        results = yield [fx.as_ir_value(x) for x in acc_n]
                    acc_l = [Vec(results[i]) for i in range_constexpr(8)]
                    acc_f = pv_next((nt - 1) % NS, (nt - 1) % 2, acc_l)
                else:
                    acc_f = pv_first(fx.Int32(0), fx.Int32(0))
                wait_dma(0)
            _barrier()
            if nt > 0:
                m_f = []
                l_f = []
                for ht in range_constexpr(2):
                    l0 = _ld_f32(lds_fin + (0 * H + 32 * ht + l32) * 4)
                    l1 = _ld_f32(lds_fin + (1 * H + 32 * ht + l32) * 4)
                    m_f.append(_ld_f32(lds_fin + (2 * H + 32 * ht + l32) * 4))
                    l_f.append(l0 + l1)
            if const_expr(not splitk):
                if live:
                    finalize(acc_f, m_f, l_f)
            else:
                if nt > 0:
                    uid = rowc * i32_S + split
                    pr = prsrc()
                    for d in range_constexpr(4):
                        for ht in range_constexpr(2):
                            for a in range_constexpr(4):
                                v4 = Vec.from_elements(
                                    [acc_f[2 * d + ht][4 * a + e].bitcast(fx.Int32) for e in range_constexpr(4)], fx.Int32
                                )
                                buffer_ops.buffer_store(
                                    v4, pr, part_acc_off(uid, (2 * d + ht) * 4 + a), cache_modifier=AGENT_SCOPE
                                )
                    if wp == 0:
                        if hh == 0:
                            for ht in range_constexpr(2):
                                buffer_ops.buffer_store(
                                    Vec.from_elements([_bits(m_f[ht]), _bits(l_f[ht])], fx.Int32),
                                    pr,
                                    part_stat_off(uid, 32 * ht + l32),
                                    cache_modifier=AGENT_SCOPE,
                                )
                _waitcnt_vm(0)

        if const_expr(splitk):
            if live:
                _barrier()
                if tx == 0:
                    _llvm.store(
                        _raw(_global_atomic_add(arg_cnt + fx.Int64(rowc) * 4, 1)), _lds_ptr(lds_tk), alignment=4
                    )
                _barrier()
                last = uni(fx.Int32(_llvm.load(T.i32, _lds_ptr(lds_tk), alignment=4)))
                if last == i32_S - 1:
                    if not isq:
                        n_s = (nt_row + (per - 1)) // per
                        res = [fx.as_ir_value(neg_inf), fx.as_ir_value(neg_inf)]
                        for sp, st_ in range(  # type: ignore[call-overload]
                            fx.Index(0), fx.Index(n_s), fx.Index(1), init=[fx.as_ir_value(neg_inf)] * 2
                        ):
                            uid_s = rowc * i32_S + fx.Int32(sp)
                            new_m = []
                            for ht in range_constexpr(2):
                                v = Vec(
                                    buffer_ops.buffer_load(
                                        prsrc(), part_stat_off(uid_s, 32 * ht + l32), vec_width=2, dtype=fx.Int32,
                                        cache_modifier=AGENT_SCOPE,
                                    )
                                ).bitcast(fx.Float32)
                                new_m.append(_maxf_nn(fx.Float32(st_[ht]), v[0]))
                            res = yield [fx.as_ir_value(x) for x in new_m]
                        mstar = [fx.Float32(res[0]), fx.Float32(res[1])]
                        init2 = init_acc + [fx.Float32(0.0), fx.Float32(0.0)]
                        res2 = [fx.as_ir_value(x) for x in init2]
                        for sp, st_ in range(  # type: ignore[call-overload]
                            fx.Index(0), fx.Index(n_s), fx.Index(1), init=[fx.as_ir_value(x) for x in init2]
                        ):
                            uid_s = rowc * i32_S + fx.Int32(sp)
                            acc_p = [Vec(st_[i]) for i in range_constexpr(8)]
                            wts, l_n2 = [], []
                            for ht in range_constexpr(2):
                                v = Vec(
                                    buffer_ops.buffer_load(
                                        prsrc(), part_stat_off(uid_s, 32 * ht + l32), vec_width=2, dtype=fx.Int32,
                                        cache_modifier=AGENT_SCOPE,
                                    )
                                ).bitcast(fx.Float32)
                                wt = _exp2(v[0] - mstar[ht])
                                wts.append(wt)
                                l_n2.append(fx.math.fma(wt, v[1], fx.Float32(st_[8 + ht])))
                            acc_n2 = []
                            for d in range_constexpr(4):
                                for ht in range_constexpr(2):
                                    elems = []
                                    for a in range_constexpr(4):
                                        pq = Vec(
                                            buffer_ops.buffer_load(
                                                prsrc(), part_acc_off(uid_s, (2 * d + ht) * 4 + a), vec_width=4,
                                                dtype=fx.Int32, cache_modifier=AGENT_SCOPE,
                                            )
                                        ).bitcast(fx.Float32)
                                        for e in range_constexpr(4):
                                            elems.append(fx.math.fma(wts[ht], pq[e], acc_p[2 * d + ht][4 * a + e]))
                                    acc_n2.append(Vec.from_elements(elems, fx.Float32))
                            res2 = yield [fx.as_ir_value(x) for x in acc_n2 + l_n2]
                        acc_c = [Vec(res2[i]) for i in range_constexpr(8)]
                        l_c = [fx.Float32(res2[8 + i]) for i in range_constexpr(2)]
                        finalize(acc_c, mstar, l_c)
                    if tx == 0:
                        _llvm.store(_raw(fx.Int32(0)), _g_ptr(arg_cnt + fx.Int64(rowc) * 4), alignment=4)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_main: fx.Int64,
        arg_xtra: fx.Int64,
        arg_mip: fx.Int64,
        arg_mid: fx.Int64,
        arg_xip: fx.Int64,
        arg_xid: fx.Int64,
        arg_sink: fx.Int64,
        arg_pos: fx.Int64,
        arg_cs: fx.Int64,
        arg_out: fx.Int64,
        arg_osc: fx.Int64,
        arg_qs: fx.Int64,
        arg_ks: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        i32_T: fx.Int32,
        i32_rows_x: fx.Int32,
        i32_S: fx.Int32,
        i32_m_lg: fx.Int32,
        i32_m_st: fx.Int32,
        i32_x_lg: fx.Int32,
        i32_x_st: fx.Int32,
        i32_cs_st: fx.Int32,
        i32_pos_b: fx.Int32,
        f32_sm: fx.Float32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q, arg_main, arg_xtra, arg_mip, arg_mid, arg_xip, arg_xid, arg_sink, arg_pos, arg_cs, arg_out,
            arg_osc, arg_qs, arg_ks, arg_part, arg_cnt, i32_T, i32_rows_x, i32_S, i32_m_lg, i32_m_st, i32_x_lg,
            i32_x_st, i32_cs_st, i32_pos_b, f32_sm,
        ).launch(grid=(i32_grid, 1, 1), block=(64 * NW, 1, 1), stream=stream)  # fmt: skip

    launch.kernel_name = name
    launch.compile_hints = {"waves_per_eu": 2, **PRELOAD_HINTS}
    return launch
