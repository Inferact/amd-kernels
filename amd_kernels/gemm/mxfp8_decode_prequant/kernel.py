# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""gfx950 MXFP8 x MXFP8 GEMM for decode-size M, one launch, no LDS staging of A or W:

    out[m, n] = sum_k A[m, k] * W[n, k]          (E4M3 codes, per-32 E8M0 scales, bf16 out)

W is the (16, 16) preshuffled weight with AITER's ``shuffle_scale_a16w4`` scales (one scale row per output
row). A comes in one of three forms:

  shuffled   A in the weight's (16, 16) layout (16-row tiles instead of 16-column tiles), scales shuffled the same
             way: one ``buffer_load`` brings a contiguous 1 KB block straight into the 64 lanes of a wave, lane
             (klane, r) getting K bytes ``[16*klane, +16)`` of row r of a 64-K block; two blocks (K ``+0`` and
             ``+64``) make one 8-VGPR operand of ``v_mfma_scale_f32_16x16x128_f8f6f4``.
  ``A_BF16`` the bf16 activation itself, row-major [M, K], quantized in registers right before the MFMA (the
             per-32 amax from the two 16-K segments a lane holds and its neighbours' over permlane swaps; the
             scale byte is the Triton MXFP8 quantizer's, the codes are x / 2^(sb - 127) rounded to nearest even).
  ``A_RM``   pre-quantized row-major E4M3 codes [M, K] and row-major E8M0 scales [M, K / 32] (the producer's
             output as it is): the same 16 B per lane per block from the row-major rows, and the scale dword of a
             lane assembled from four row-major dword loads (rows r and r + 16, the two 128-K halves, byte
             klane) at the step's first use, so the loads keep their look-ahead. Rows beyond M read zero.

A workgroup is ``KW`` K-waves over the same ``NI`` 16-column tiles and ``RT`` 16-row tiles: wave w covers K-steps
``[w*SW, (w+1)*SW)`` (128 K each); W (non-temporal loads) and A blocks are issued ahead of the MFMAs as far as the
registers allow. After the last step every K-wave parks the fp32 partial sums of the tiles it does not own in LDS;
the owner of a tile adds the KW partials in wave order (deterministic) and stores it: the MFMA is fed (W, A), so a
lane ends up with 4 consecutive columns of one row.

``EPI_LIMIT`` (gate_up weights, rows [gate; up] with N = 2 I): the epilogue instead of storing bf16 tiles. A
workgroup takes NI / 2 gate column tiles and the NI / 2 up tiles of the same columns; the owner of a block of two
gate and two up tiles rounds the sums to bf16, applies ``bf16(bf16(silu(min(gate, limit))) * clamp(up, +-limit))``
in fp32, quantizes the 32 columns (two tiles; the amax over the four lanes of a row by permlane swaps) as the
Triton quantizer does and stores E4M3 codes [M, I] and E8M0 bytes [M, I / 32], both row-major. No K split over
workgroups (``KS`` 1).

Layouts (bytes):
  A        [RT*16, K]          shuffle_weight(A padded to 16 rows, (16, 16))
  A_scale  [ceil(M/32)*32, K/32]  shuffle_scale_a16w4: [M/32, K/256, klane 4, r 16] dwords,
                                  bytes = (16-row half) + 2 * (128-K half)
  W        [N, K], W_scale     the same two shuffles of the weight
  OUT      [M, N] bf16 (``KS`` > 1: the fp32 partials [KS, M, N])

Odd RT (a 16*RT-row tile with RT % 2 == 1): the shuffled A_scale atom pairs two 16-row tiles per dword, so the
MFMA's half-select (op_sel, a trace-time immediate) would flip meaning with the m-block's parity. The kernel
re-pairs instead: scale dwords are addressed from the m-block's first tile rounded DOWN to even, and when the
block starts on the odd half, each dword consumed by the MFMAs is rebuilt at load time. RT=1 keeps its own shift
path; ``A_BF16`` builds its own scale dwords and stays even-RT; ``A_RM`` builds each dword from the rows of its
own tiles and needs no repair.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_4wave import _acc_read

W_CACHE_MOD = 2  # non-temporal W loads: a decode GEMM reads every weight byte once
# VGPRs the kernel may hold at once. An MFMA operand is 8 contiguous VGPRs, so the register
# allocator runs out well before all 256 are live (measured: clean at 199 live, parking
# operands in AGPRs at 202). A parked operand is copied back with v_accvgpr_read right
# before its MFMA and races the AGPR write: wrong sums, some of them only now and then.
VGPR_BUDGET = 192
# kernel arguments preloaded into SGPRs instead of s_load'ed before the first loads
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _addr(t):
    """A tensor argument's device address (in a launch body)."""
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def _lookahead(SW, RT, NI, RP, CP, vgpr_budget=None, a_vgprs=8, reserve=0, fold=1):
    """(W steps, A blocks) issued ahead of the MFMAs: the longest common horizon whose
    registers fit. Held at once: the operands in flight plus the one in use of each
    queue (8 VGPRs per 128-K operand), a scale dword per 32-row / 32-column group for
    every other W step, and the per-lane offset vector of every row / column tile.
    An A block is one (step, row tile) pair, or ``fold`` steps of the folded bf16 mode."""
    budget = (vgpr_budget or VGPR_BUDGET) - (RT + NI) - reserve
    best = None
    for la_a in range(1, SW * RT // fold + 1):
        la_w = min(SW, -(-la_a * fold // RT))
        regs = a_vgprs * (la_a + 1) + 8 * NI * (la_w + 1) + (CP + RP) * (la_w // 2 + 2)
        if regs > budget:
            break
        best = (la_w, la_a)
    assert best is not None, f"RT {RT} x NI {NI} leaves no VGPRs for the operands"
    return best


def _minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _pk_bf16(a, b):
    """Two fp32 -> two bf16 (round to nearest even) in a dword, ``a`` low."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def _unpack_bf16(dwords):
    out = []
    for w in dwords:
        w = fx.Int32(w)
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _swiglu_bf16(gate, up, limit):
    """4 fp32 sums of gate and of up -> 2 dwords of bf16 pairs of
    bf16(bf16(silu(min(bf16(gate), limit))) * clamp(bf16(up), -limit, limit))."""
    g = _unpack_bf16([_pk_bf16(gate[0], gate[1]), _pk_bf16(gate[2], gate[3])])
    u = _unpack_bf16([_pk_bf16(up[0], up[1]), _pk_bf16(up[2], up[3])])
    g = [_minnum(v, limit) for v in g]
    u = [_minnum(_maxnum(v, -limit), limit) + fx.Float32(0.0) for v in u]
    s = _unpack_bf16(
        [_pk_bf16(*[v / (fx.Float32(1.0) + fx.math.exp(-v)) for v in g[2 * i : 2 * i + 2]]) for i in range(2)]
    )
    return [_pk_bf16(s[2 * i] * u[2 * i], s[2 * i + 1] * u[2 * i + 1]) for i in range(2)]


SC1 = 0x10  # agent-scope (cross-XCD) stores and loads of the split-K slabs
CNT_STRIDE = 16  # dwords between the tile counters
EPOCH = 8  # what every fused launch adds to a tile's counter (>= KS, a power of two)


def _atomic_add(addr, v):
    """A returning scalar global add (one lane): takes a tile's ticket. The caller has drained its slab stores."""
    low = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int64(addr).to(fx.Int32)))
    high = fx.Int32(fx.rocdl.readfirstlane(T.i32, (fx.Int64(addr) >> 32).to(fx.Int32)))
    pointer = (low.to(fx.Uint32).to(fx.Uint64) | (high.to(fx.Uint32).to(fx.Uint64) << 32)).bitcast(fx.Int64)
    inc = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.as_ir_value(inc), fx.as_ir_value(pointer)],
            "s_atomic_add $0, $2, 0 glc\n\ts_waitcnt lgkmcnt(0)", "=s,0,s,~{memory}",
            has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )  # fmt: skip


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _pack8(h0, h1):
    """two 4 x i32 halves -> one 8 x i32 MFMA operand (VGPRs 0-3 = K [16*klane, +16),
    VGPRs 4-7 = K [64 + 16*klane, +16))."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements(
        [a[k] for k in range(4)] + [b[k] for k in range(4)], fx.Int32
    )


def _mfma(w_op, a_op, acc, sw, sa, jw, ia, k2):
    """One scaled 16x16x128 fp8 MFMA, accumulator pinned in AGPR (``=a`` / tied ``0``).
    Operand order (W, A): lane (q, r) accumulates out[row r, columns 4q .. 4q+3].
    Scale byte of a dword = op_sel + 2 * op_sel_hi (16-row / 16-col half, 128-K half)."""
    src2 = "$0" if acc is not None else "0"
    asm = (
        f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
        f"op_sel:[{jw},{ia},0] op_sel_hi:[{k2},{k2},0] cbsz:0 blgp:0"
    )
    ops = [
        fx.as_ir_value(w_op),
        fx.as_ir_value(a_op),
        fx.as_ir_value(sw),
        fx.as_ir_value(sa),
    ]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(fx.as_ir_value(acc))
        cons += ",0"
    return llvm.inline_asm(
        Vec.make_type(4, fx.Float32), ops, asm, cons, has_side_effects=True
    )


def _pk_max_u16(a, b):
    """per-16-bit-half unsigned max (v_pk_max_u16): on sign-cleared bf16 patterns that is
    the value order."""
    va = llvm.BitcastOp(T.i16x2, _raw(fx.Int32(a))).result
    vb = llvm.BitcastOp(T.i16x2, _raw(fx.Int32(b))).result
    return fx.Int32(llvm.BitcastOp(T.i32, llvm.intr_umax(va, vb)).result)


def _amax_bf16(dwords):
    """max |x| over the 2 * len(dwords) bf16 values, as a 16-bit bf16 pattern in an i32."""
    m = None
    for d in dwords:
        a = fx.Int32(d) & fx.Int32(0x7FFF7FFF)
        m = a if m is None else _pk_max_u16(m, a)
    lo, hi = m & fx.Int32(0xFFFF), m >> 16
    return (lo > hi).select(lo, hi)


def _i32_max(a, b):
    return (a > b).select(a, b)


def _swap16(v):
    """(even-row value, odd-row value) of this lane's 16-lane row pair (klane q, q ^ 1)."""
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _raw(fx.Int32(v))
    res = fx.rocdl.permlane16_swap(pair_ty, raw, raw, False, False)
    return fx.Int32(llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        llvm.extractvalue(T.i32, res, [1])
    )


def _swap32(v):
    """(low-half value, high-half value): the values of klanes 0/1 and 2/3."""
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _raw(fx.Int32(v))
    res = fx.rocdl.permlane32_swap(pair_ty, raw, raw, False, False)
    return fx.Int32(llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        llvm.extractvalue(T.i32, res, [1])
    )


def _e8m0_bf16(amax_bits):
    """(scale byte, conversion scale) of a block from the bf16 pattern of its maximum. The
    byte is the Triton MXFP8 quantizer's, clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0,
    254): with amax = 2^e * m, ceil(log2(amax / 448)) = e - 8 + [m > 1.75], so it is the
    exponent field - 8 + [mantissa > 0x60], which (a + 0x1F) >> 7 gives in one carry;
    zero, denormal and tiny maxima land on 0 as in that quantizer. The conversion divides by
    2^(byte - 127); byte 0 would make that a denormal, so such a block (maximum below
    2^-119) is converted with 2^-126 instead."""
    t = (fx.Int32(amax_bits) + fx.Int32(0x1F)) >> 7
    sb = _i32_max(t, fx.Int32(8)) - fx.Int32(8)
    scale = ((_i32_max(t, fx.Int32(9)) - fx.Int32(8)) << 23).bitcast(fx.Float32)
    return sb, scale


def _e8m0_bf16_pp2(amax_bits):
    """NVIDIA-consistent variant of ``_e8m0_bf16``: the stored
    byte IS the conversion exponent (``max(t, 9) - 8``, never 0), so the stored
    scale always equals the 2^(byte - 127) its consumer decodes -- ``_e8m0_bf16``
    stores 0 for a block maximum below ~1.75*2^-119 while converting with 2^-126
    (a 2x error under an OCP consumer, a zeroed block under a ``(byte << 23)``
    one). Identical output for every bf16 block maximum above 1.75*2^-119, which
    includes every nonzero normal activation."""
    t = (fx.Int32(amax_bits) + fx.Int32(0x1F)) >> 7
    sb = _i32_max(t, fx.Int32(9)) - fx.Int32(8)
    scale = (sb << 23).bitcast(fx.Float32)
    return sb, scale


def _row_shl(vals, n):
    """Each lane takes the value of lane + n of its 16-lane row (v_mov_b32 dpp row_shl;
    the last n lanes of a row get 0)."""
    zero = _raw(fx.Int32(0))
    return [
        fx.Int32(
            fx.rocdl.update_dpp(
                T.i32, zero, _raw(fx.Int32(v)), 0x100 + n, 0xF, 0xF, True
            )
        )
        for v in vals
    ]


def _quant_bf16(dwords, scale):
    """dwords of bf16 pairs -> half as many dwords of fp8 codes: x / scale, round to nearest
    even (v_cvt_scalef32_pk_fp8_bf16, the scale is 2^(sb - 127))."""
    out = []
    for i in range(0, len(dwords), 2):
        pair0 = llvm.BitcastOp(T.bf16x2, _raw(fx.Int32(dwords[i]))).result
        pair1 = llvm.BitcastOp(T.bf16x2, _raw(fx.Int32(dwords[i + 1]))).result
        # the first conversion's other half is overwritten by the second
        w = fx.rocdl.cvt_scalef32_pk_fp8_bf16(
            T.i16x2, llvm.mlir_undef(T.i16x2), pair0, _raw(scale), False
        )
        w = fx.rocdl.cvt_scalef32_pk_fp8_bf16(T.i16x2, w, pair1, _raw(scale), True)
        out.append(fx.Int32(llvm.BitcastOp(T.i32, w).result))
    return Vec.from_elements(out, fx.Int32)


def _pin_accumulators(values):
    """Route the accumulators through an inline asm that outputs them again (tied
    ``=a`` / ``0``) with the XDL wait states inside: the reads that follow depend on
    this fence and cannot be hoisted right behind the last inline-asm MFMA (the
    compiler sees no hazard through the asm; without it the last MFMA of a wave is
    missing from half of each accumulator)."""
    one = len(values) == 1  # an inline asm with one output returns it bare
    ty = ir.Type.parse(
        "vector<4xf32>"
        if one
        else "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(values)) + ")>"
    )
    constraints = ",".join(["=a"] * len(values) + [str(i) for i in range(len(values))])
    r = llvm.inline_asm(
        ty,
        [_raw(v) for v in values],
        "s_nop 15\ns_nop 15",
        constraints,
        has_side_effects=True,
    )
    if one:
        return [r]
    return [llvm.extractvalue(T.vec(4, T.f32), r, [i]) for i in range(len(values))]


def compile_mxfp8_gemm_decode(
    *,
    N: int,
    K: int,
    RT: int,
    NI: int,
    KW: int,
    KS: int = 1,
    vgpr_budget: int | None = None,
    sched_barriers: bool = True,
    LA: tuple[int, int] | None = None,
    A_BF16: bool = False,
    A_RM: bool = False,
    EPI_LIMIT: float | None = None,
    SK_FUSE: bool = False,
    FOLD: int = 1,
    W_CM: int | None = None,
    PARK_ALL: bool | None = None,
    OWNER_READ: bool | None = None,
    OUT_CM: int | None = None,
    INTERLEAVE_PRIME: bool | None = None,
):
    """Kernel for one (N, K) weight, ``RT`` 16-row tiles per workgroup (M <= 16*RT per
    m-block), ``NI`` 16-column tiles per workgroup, ``KW`` K-waves. Grid =
    ``KS * ceil(M / (16*RT)) * launch.n_blocks`` workgroups of ``64*KW`` threads.

    ``KS`` > 1 splits K over workgroups as well (a workgroup then reads only its
    ``K / KS`` of A: what bounds a workgroup at M >= 64 is the A it has to ingest);
    OUT is then the fp32 ``[KS, M, N]`` partial slabs for the split-K reduce kernel.

    ``A_BF16``: A is the bf16 activation itself, row-major [M, K] (rows beyond M read
    zero), quantized in registers right before the MFMA -- the per-32 amax needs the two
    16-K segments a lane holds and their neighbours' (klane q ^ 1 through a 16-lane row
    swap, q ^ 2 through a 32-lane swap), the scale byte is the Triton quantizer's, the
    codes are x / 2^(sb - 127) rounded to nearest even (v_cvt_scalef32_pk_fp8_bf16, the
    per-32 maxima on packed bf16 patterns); A_scale is unused. Saves the quantizer launch at
    decode M (2.1 us of 7.5 at M 4). Every workgroup quantizes its rows again, so the cost
    is the VALU work per step (~70 instructions per lane, for the 16 rows of a tile);
    ``FOLD`` 2 / 4 (M <= 8 / 4, ``RT`` 1): a tile holds 16 / FOLD rows and the lanes of
    MFMA row r + (16 / FOLD) * j quantize step j of every FOLD-step block for row r --
    one load and a FOLD-th of the VALU work per step, the codes and the scale byte move
    to MFMA row r with a DPP row shift right before the MFMA; MFMA rows beyond 16 / FOLD
    compute junk that the store masks.

    ``A_RM`` and ``EPI_LIMIT``: the module docstring. With ``EPI_LIMIT`` OUT is the E4M3 codes [M, N / 2]
    and ``SC_OUT`` the E8M0 bytes [M, N / 64].

    ``SK_FUSE`` (``KS`` 2-8): no separate split-K reduce launch. A workgroup stores its fp32 partial tile in the slab
    [KS, M, N] (agent-scope stores), drains the stores, and takes a ticket on its tile's counter (a scalar atomic
    add; the counters are epoch counters: each launch adds ``EPOCH`` in all, so they return to a multiple of it and
    need no reset, and one launch at a time runs on a counter array). The workgroup that takes the last ticket sums
    the KS partials in split order and writes the bf16 output (or runs the epilogue); the others are done. Arguments
    ``CNT`` (the counters) and ``SLAB`` are device memory the host keeps; OUT is the final output.

    ``LA`` = (W steps, A blocks) in flight ahead of the MFMAs, by default the deepest
    pair that fits ``vgpr_budget`` VGPRs; ``sched_barriers`` keeps the loads where the
    source issues them (off, the scheduler hoists them: a little faster while the
    registers last). The tuned rows carry the values the sweep found clean and fastest.

    ``INTERLEAVE_PRIME=False`` runs the preceding W-first burst prologue for A/B tests.
    Measured rows otherwise alternate one W step and its A fragments (A first for o_proj).

    ``OWNER_READ=False`` runs the preceding owner-copy epilogue for A/B tests.

    ``OUT_CM`` is the output stores' cache modifier (default: non-temporal, 2, on the rows of
    ``_NT_OUT_ROWS``, else 0 = the default policy); it changes no byte of the output.

    ``PARK_ALL`` (default: the measured rows of ``_PARK_ALL_ROWS``) is the K-reduce form
    with ``KW`` > 1: every K-wave parks every tile in LDS straight from its AGPRs, those of a
    row tile right after the next row tile's MFMAs of the last step, and wave w then sums
    tiles w, w + KW, ... from LDS in wave order (all LDS reads of a group of tiles before
    its first store; the tile's part of the store offset in the scalar offset) -- the sums
    of the owner form below, bit for bit, without its per-wave code copies and the
    AGPR-to-VGPR copies of the owned tiles. ``False``: the owner form (the owner of tile
    i = K-wave i % KW keeps its own partial in registers and adds the others' from LDS)."""
    a_first_prime = False
    interleave_prime = bool(INTERLEAVE_PRIME)
    assert not interleave_prime or (not A_BF16 and not A_RM and FOLD == 1)
    w_cm = W_CACHE_MOD if W_CM is None else W_CM
    assert w_cm in (0, 1, 2), w_cm
    assert not (A_BF16 and A_RM)
    assert not SK_FUSE or (KS > 1 and not PARK_ALL and FOLD == 1)
    assert not SK_FUSE or KS <= EPOCH
    assert not (EPI_LIMIT is not None and PARK_ALL)
    EPI = EPI_LIMIT is not None
    EPI_DIRECT = EPI and not SK_FUSE  # the epilogue runs where the K-waves are summed
    HP = NI // 2  # EPI: gate tiles (and as many up tiles) per workgroup
    if EPI:
        assert (KS == 1 or SK_FUSE) and NI % 4 == 0 and (N // 32) % HP == 0 and FOLD == 1, (N, NI, KS)
    assert K % 256 == 0 and N % (16 * NI) == 0
    KSTEPS = K // 128  # 128-K steps
    assert KSTEPS % (KS * KW) == 0 and (KSTEPS // (KS * KW)) % 2 == 0, (K, KS, KW)
    SW = KSTEPS // (KS * KW)  # steps per K-wave (even: starts on a 256-K scale block)
    # Odd RT is legal only on the quantized-A path, where the A-scale dwords are
    # re-paired to the m-block's parity at load time (module docstring). A_BF16 builds
    # each scale dword itself with the byte at (rt % 2) and stays even-RT.
    assert RT == 1 or RT % 2 == 0 or not A_BF16
    BM = 16 * RT // FOLD  # rows per m-block
    NB = N // (16 * NI)
    SCB = K // 256  # scale blocks (64 dwords) per 32-row / 32-column group
    RP = max(1, RT // 2)  # 32-row scale groups per workgroup
    if RT > 1 and RT % 2 == 1:
        RP = (RT + 1) // 2  # the parity-straddling block needs one group more
    # W scale dwords per step: one per 32-column group, or one per 16-column tile when
    # NI is odd (a group can then straddle two workgroups: each tile shifts its half down)
    CP = NI if NI % 2 else NI // 2
    LA_W, LA_A = LA or _lookahead(
        SW, RT, NI, 4 * RP if A_RM else RP, CP, vgpr_budget, a_vgprs=16 if A_BF16 else 8,
        reserve=(24 + (8 if FOLD > 1 else 0)) if A_BF16 else 0, fold=FOLD,
    )  # fmt: skip
    assert FOLD in (1, 2, 4) and (
        FOLD == 1 or (A_BF16 and RT == 1 and SW % FOLD == 0)
    ), (FOLD, RT, SW)
    RPT = 16 // FOLD  # rows per 16-row tile
    NA = SW * RT // FOLD  # A blocks per K-wave
    assert 1 <= LA_W <= SW and 1 <= LA_A <= NA, (LA_W, LA_A, SW, RT, FOLD)
    RED_SLOT = RT * NI * 64 * 16  # bytes parked per K-wave
    park_all = bool(PARK_ALL) and KW > 1
    owner_read = bool(OWNER_READ) and KW > 1 and not park_all
    out_cm = 0 if OUT_CM is None else OUT_CM
    assert out_cm in (0, 1, 2, 16, 17, 18), out_cm
    out_kw = dict(cache_modifier=out_cm) if out_cm else {}
    LDS_BYTES = KW * RED_SLOT if park_all else max(16, (KW - 1) * RED_SLOT)
    assert LDS_BYTES <= 160 * 1024, "K-reduce scratch exceeds the LDS"
    assert N * K <= 0xFFFFFFFF

    @fx.struct
    class Shared:
        red: fx.Array[fx.Uint8, LDS_BYTES, 16]
        ticket: fx.Array[fx.Int32, 4]

    name = (
        f"flydsl_mxfp8_dec_n{N}_k{K}_rt{RT}_ni{NI}_kw{KW}_ks{KS}_la{LA_W}x{LA_A}"
        + ("sb" if sched_barriers else "")
        + (("_fq" + (str(FOLD) if FOLD > 1 else "")) if A_BF16 else "")
        + ("_rm" if A_RM else "")
        + ("_swiglu" if EPI else "")
        + ("_skf" if SK_FUSE else "")
        + (f"_wcm{w_cm}" if w_cm != W_CACHE_MOD else "")
        + ("_ip" if interleave_prime else "")
        + ({0: "", W_CACHE_MOD: "_ont"}.get(out_cm, f"_ocm{out_cm}"))
        + ("_pk" if park_all else "")
        + ("_ownread" if owner_read else "")
    )

    # addresses, not tensors (a tensor argument ends the preload): every argument
    # arrives preloaded in SGPRs
    @flyc.kernel(name=name, known_block_size=[64 * KW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        n_mb: fx.Int32,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_as: fx.Int64,
        arg_a: fx.Int64,
        arg_out: fx.Int64,
        arg_sc_out: fx.Int64,
        arg_cnt: fx.Int64,
        arg_slab: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        if const_expr(KS > 1):
            kz, wg = bx // (n_mb * NB), bx % (n_mb * NB)
            kwave = kz * KW + wave
        else:
            kz, wg, kwave = 0, bx, wave
        mb, nb = wg // NB, wg % NB
        lane_off = q16 * 256 + l16 * 16  # a lane's 16 B inside a 1 KB block
        k_base = kwave * (SW * 2048)  # this K-wave's first block pair (bytes)

        if const_expr(A_BF16):
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_a, num_records_bytes=i32_m * (2 * K)
            )
        elif const_expr(A_RM):
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_a, num_records_bytes=i32_m * K
            )
        else:
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_a, num_records_bytes=n_mb * (BM * K)
            )
        w_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_w, num_records_bytes=N * K
        )
        as_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_as, num_records_bytes=a_scale_bytes
        )
        ws_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_ws, num_records_bytes=(N // 32) * SCB * 256
        )
        if const_expr(EPI):
            out_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_out, num_records_bytes=i32_m * (N // 2)
            )
            sc_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_sc_out, num_records_bytes=i32_m * (N // 64)
            )
            # the gate tiles of this workgroup, then the up tiles of the same columns
            n16 = [
                nb * HP + ni if ni < HP else N // 32 + nb * HP + (ni - HP)
                for ni in range_constexpr(NI)
            ]
        else:
            out_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_out, num_records_bytes=i32_m * (N * (2 if KS == 1 or SK_FUSE else 4 * KS))
            )
            n16 = [nb * NI + ni for ni in range_constexpr(NI)]
        if const_expr(SK_FUSE):
            slab_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_slab, num_records_bytes=i32_m * (N * 4 * KS)
            )
        r16 = [mb * RT + rt for rt in range_constexpr(RT)]
        w_dw = [
            (n16[ni] * (16 * K) + k_base + lane_off) // 4 for ni in range_constexpr(NI)
        ]
        if const_expr(A_BF16):
            # row-major bf16: this lane's row, its 32-byte K segment, this K-wave's first
            # step (folded: the lane's row r and step j of each block)
            a_row, a_fold = l16 % RPT, l16 // RPT
            a_dw = [
                (
                    (r16[rt] * RPT + a_row) * (2 * K)
                    + kwave * (SW * 256)
                    + a_fold * 256
                    + q16 * 32
                )
                // 4
                for rt in range_constexpr(RT)
            ]
            sa_shift = (
                ((a_fold % 2) * 16) if FOLD > 1 else None
            )  # the block step's parity
        elif const_expr(A_RM):
            a_dw = [
                ((r16[rt] * 16 + l16) * K + kwave * (SW * 128) + q16 * 16) // 4
                for rt in range_constexpr(RT)
            ]
        else:
            a_dw = [
                (r16[rt] * (16 * K) + k_base + lane_off) // 4
                for rt in range_constexpr(RT)
            ]
        sc_lane = q16 * 16 + l16
        sc_k = kwave * (SW // 2 * 64)  # this K-wave's first 256-K scale block (dwords)
        ws_dw = [
            ((n16[cp if NI % 2 else 2 * cp] // 2) * SCB) * 64 + sc_k + sc_lane
            for cp in range_constexpr(CP)
        ]
        if const_expr(A_RM):
            # per pair of row tiles: the dword offsets of its two tiles' rows in the row-major scales
            rm_dw = [
                [
                    ((r16[t] * 16 + l16) * (K // 32) + kwave * (SW // 2) * 8) // 4
                    for t in (2 * rp, min(2 * rp + 1, RT - 1))
                ]
                for rp in range_constexpr(RP)
            ]
            as_dw = None
        elif const_expr(RT > 1 and RT % 2 == 1):
            # odd RT: address the 32-row groups from the m-block's first tile rounded
            # down to even (r16[0] // 2 floors); the parity mix below re-pairs them
            # (const_expr: a bare if is rewritten to a dynamic scf.if whose branch
            # assignments never bind in this scope -> as_dw NameError)
            as_dw = [
                ((r16[0] // 2 + rp) * SCB) * 64 + sc_k + sc_lane
                for rp in range_constexpr(RP)
            ]
        else:
            as_dw = [
                ((r16[2 * rp] // 2) * SCB) * 64 + sc_k + sc_lane
                for rp in range_constexpr(RP)
            ]
        a_par = r16[0] % 2 if RT > 1 and RT % 2 == 1 and not A_RM else None  # m-block parity
        # a 16-column (row) tile of an odd count sits in either half of its 32 group:
        # move its bytes (half, half + 2) down to (0, 2) and select half 0
        w_shift = [(n16[ni] % 2) * 8 for ni in range_constexpr(NI)] if NI % 2 else None
        a_shift = (r16[0] % 2) * 8 if RT == 1 and not A_RM else None

        def load_pair(rsrc, dw, s, nt):
            kw = dict(vec_width=4, dtype=fx.Int32)
            if const_expr(nt):
                kw["cache_modifier"] = w_cm
            return _pack8(
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048, **kw),
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048 + 1024, **kw),
            )

        def load_scales(rsrc, dws, s, shift):
            vals = [
                fx.Int32(
                    buffer_ops.buffer_load(
                        rsrc,
                        dw,
                        vec_width=1,
                        dtype=fx.Int32,
                        soffset_bytes=(s // 2) * 256,
                    )
                )
                for dw in dws
            ]
            if const_expr(shift is None):
                return vals
            if const_expr(isinstance(shift, list)):
                return [v.shrui(sh) for v, sh in zip(vals, shift)]
            return [v.shrui(shift) for v in vals]

        def load_scales_rm(s):
            """Raw dwords of the row-major A scales of 256-K block s // 2, in the order (pair of row tiles,
            128-K half, tile of the pair); ``combine_rm`` builds the MFMA dwords from them."""
            return [
                fx.Int32(
                    buffer_ops.buffer_load(
                        as_rsrc,
                        rm_dw[rp][t],
                        vec_width=1,
                        dtype=fx.Int32,
                        soffset_bytes=(s // 2) * 8 + kh * 4,
                    )
                )
                for rp in range_constexpr(RP)
                for kh in range_constexpr(2)
                for t in range_constexpr(2)
            ]

        def combine_rm(raw, rp):
            """The scale dword of row-tile pair rp: byte t + 2 * kh = klane q's byte of the dword of
            tile t of the pair, half kh."""
            sh = q16 * 8
            out = None
            for i in range_constexpr(4):
                b = (raw[4 * rp + i].shrui(sh)) & fx.Int32(0xFF)
                out = b if out is None else out | (b << (8 * i))
            return out

        def repair_a_scale(vals, p):
            """Odd RT: re-pair scale dword p to the even-parity layout the MFMA immediates
            assume -- dword p must carry block tiles 2p (bytes 0/2) and 2p+1 (bytes 1/3).
            An odd-parity block shifts every atom by one half: bytes 0/2 come from half 1
            of group p, bytes 1/3 from half 0 of group p+1 (the last pair's byte 1/3 name a
            tile past the block and are never consumed). Called at the dword's first MFMA,
            not at its load, so the load keeps its full look-ahead (VMLEAD)."""
            nxt = vals[p + 1] if p + 1 < RP else vals[p]
            return (a_par == 1).select(
                (vals[p].shrui(fx.Int32(8)) & fx.Int32(0x00FF00FF))
                | ((nxt & fx.Int32(0x00FF00FF)) << 8),
                vals[p],
            )

        def load_w_step(s):
            bb = [load_pair(w_rsrc, w_dw[ni], s, True) for ni in range_constexpr(NI)]
            if const_expr(s % 2 == 0):
                return bb, (
                    load_scales(ws_rsrc, ws_dw, s, w_shift),
                    None
                    if A_BF16
                    else (load_scales_rm(s) if A_RM else load_scales(as_rsrc, as_dw, s, a_shift)),
                )
            return bb, None

        def load_a(t):
            rt, s = t % RT, (t // RT) * FOLD  # block t: row tile, first step
            if const_expr(A_BF16):
                # the two 16-K segments (K 16q and 64 + 16q of the step), 32 B each, bf16
                return [
                    buffer_ops.buffer_load(
                        a_rsrc,
                        a_dw[rt],
                        vec_width=4,
                        dtype=fx.Int32,
                        soffset_bytes=s * 256 + off,
                    )
                    for off in (0, 16, 128, 144)
                ]
            if const_expr(A_RM):
                kw = dict(vec_width=4, dtype=fx.Int32)
                return _pack8(
                    buffer_ops.buffer_load(a_rsrc, a_dw[rt], soffset_bytes=s * 128, **kw),
                    buffer_ops.buffer_load(a_rsrc, a_dw[rt], soffset_bytes=s * 128 + 64, **kw),
                )
            return load_pair(a_rsrc, a_dw[rt], s, False)

        def quantize(raw, rt, s):
            """raw bf16 of (row tile rt, step s) -> (fp8 operand, this MFMA's scale dword)."""
            seg0 = [Vec(raw[0])[i] for i in range(4)] + [
                Vec(raw[1])[i] for i in range(4)
            ]
            seg1 = [Vec(raw[2])[i] for i in range(4)] + [
                Vec(raw[3])[i] for i in range(4)
            ]
            # per-32 groups of this row: pair (q, q ^ 1) holds group q // 2 in segment 0 and
            # group 2 + q // 2 in segment 1; the other pair's values come with the 32-swap
            e0, o0 = _swap16(_amax_bf16(seg0))
            e1, o1 = _swap16(_amax_bf16(seg1))
            own0, own1 = _i32_max(e0, o0), _i32_max(e1, o1)
            (sb0, sc0), (sb1, sc1) = _e8m0_bf16(own0), _e8m0_bf16(own1)
            # the scale byte the MFMA reads from this lane is group q's
            lo0, hi0 = _swap32(sb0)
            lo1, hi1 = _swap32(sb1)
            sbm = (q16 == 0).select(
                sb0, (q16 == 1).select(hi0, (q16 == 2).select(lo1, sb1))
            )
            a_op = _pack8(_quant_bf16(seg0, sc0), _quant_bf16(seg1, sc1))
            if const_expr(FOLD > 1):
                return a_op, sbm << sa_shift
            return a_op, sbm << (8 * (2 * (s % 2) + (rt % 2)))

        # Both operands run ahead of the MFMAs: W by LA_W steps, A by LA_A (step, row
        # tile) blocks in the order the MFMAs consume them; with everything in flight
        # from the start a wave ends when its last block lands, not after one memory
        # latency per step.
        acc = [[None] * NI for _ in range(RT)]
        w_q = [None] * SW
        a_q = [None] * NA
        if const_expr(park_all):
            # K-wave `wave` parks tile i at (wave * RT * NI + i) KB, a lane's 16 B at lane * 16
            park_base = fx.Int32(fx.ptrtoint(smem.red.ptr)) + lane * 16

            def park_tile(i, v):
                llvm.StoreOp(
                    _raw(Vec(v).bitcast(fx.Int32)),
                    llvm.inttoptr(
                        ir.Type.parse("!llvm.ptr<3>"),
                        _raw(fx.Int64(park_base + (wave * (RT * NI) + i) * 1024)),
                    ),
                    alignment=16,
                )
        if const_expr(interleave_prime):
            for p in range_constexpr(max(LA_W, (LA_A + RT - 1) // RT)):
                if const_expr(not a_first_prime and p < LA_W):
                    w_q[p] = load_w_step(p)
                    fx.rocdl.sched_barrier(0)
                for rt in range_constexpr(RT):
                    if const_expr(p * RT + rt < LA_A):
                        a_q[p * RT + rt] = load_a(p * RT + rt)
                fx.rocdl.sched_barrier(0)
                if const_expr(a_first_prime and p < LA_W):
                    w_q[p] = load_w_step(p)
                    fx.rocdl.sched_barrier(0)
        else:
            for s in range_constexpr(LA_W):
                w_q[s] = load_w_step(s)
            for t in range_constexpr(LA_A):
                a_q[t] = load_a(t)
        if const_expr(sched_barriers):
            fx.rocdl.sched_barrier(0)
        sw = sa = None
        held = None  # folded mode: the block's quantized operand and scale dword
        for s in range_constexpr(SW):
            if const_expr(s + LA_W < SW):
                w_q[s + LA_W] = load_w_step(s + LA_W)
            bb, sc = w_q[s]
            if const_expr(s % 2 == 0):
                sw, sa = sc
                if const_expr(A_RM or (RT > 1 and RT % 2 == 1)):
                    sa_raw, sa = sa, [None] * RP  # built / repaired at first use (both steps)
            for rt in range_constexpr(RT):
                if const_expr(FOLD == 1):
                    t = s * RT + rt
                    if const_expr(t + LA_A < NA):
                        a_q[t + LA_A] = load_a(t + LA_A)
                    if const_expr(A_BF16):
                        a_op, sa_v = quantize(a_q[t], rt, s)
                    else:
                        if const_expr(A_RM and sa[rt // 2] is None):
                            sa[rt // 2] = combine_rm(sa_raw, rt // 2)
                        elif const_expr(RT > 1 and RT % 2 == 1 and sa[rt // 2] is None):
                            sa[rt // 2] = repair_a_scale(sa_raw, rt // 2)
                        a_op, sa_v = a_q[t], sa[rt // 2]
                    a_q[t] = None
                else:
                    b, j = s // FOLD, s % FOLD
                    if const_expr(j == 0):
                        if const_expr(b + LA_A < NA):
                            a_q[b + LA_A] = load_a(b + LA_A)
                        held = quantize(a_q[b], rt, s)
                        a_q[b] = None
                    if const_expr(j == 0):
                        a_op, sa_v = held
                    else:
                        moved = _row_shl(
                            [Vec(held[0])[i] for i in range(8)] + [held[1]], RPT * j
                        )
                        a_op, sa_v = Vec.from_elements(moved[:8], fx.Int32), moved[8]
                for ni in range_constexpr(NI):
                    acc[rt][ni] = _mfma(
                        bb[ni],
                        a_op,
                        acc[rt][ni],
                        sw[ni if NI % 2 else ni // 2],
                        sa_v,
                        0 if NI % 2 else ni % 2,
                        rt % 2,
                        s % 2,
                    )
                # the loads depend on nothing, so the scheduler would pull later steps'
                # loads up to here: more operands alive than _lookahead budgeted
                if const_expr(sched_barriers):
                    fx.rocdl.sched_barrier(0)
                if const_expr(park_all and s == SW - 1 and rt >= 1):
                    # row tile rt - 1 is final: park it behind row tile rt's NI MFMAs (the
                    # >= 2 * 32 cycles they issue in cover the XDL-write -> LDS-read wait
                    # states, which the compiler cannot see through the asm); the fences
                    # keep the writes there
                    if const_expr(not sched_barriers):
                        fx.rocdl.sched_barrier(0)
                    for ni in range_constexpr(NI):
                        park_tile((rt - 1) * NI + ni, acc[rt - 1][ni])
                    fx.rocdl.sched_barrier(0)
            w_q[s] = None

        def own(i):
            """The K-wave that sums and stores tile i (EPI: the four tiles of a quant block share one)."""
            return (i % NI % HP // 2) % KW if EPI else i % KW

        def epi_block(rt, c2, vals):
            """Quantized swiglu of quant block c2 of row tile rt: gate tiles 2 * c2 and 2 * c2 + 1 with their up
            tiles (``vals``: tile index -> fp32 sums); stores the codes and, from klane 0, the scale byte."""
            limit = fx.Float32(EPI_LIMIT)
            y = []
            for jj in range_constexpr(2):
                ni = 2 * c2 + jj
                y += _swiglu_bf16(Vec(vals[rt * NI + ni]), Vec(vals[rt * NI + HP + ni]), limit)
            e, o = _swap16(_amax_bf16(y))
            lo, hi = _swap32(_i32_max(e, o))
            sb, scale = _e8m0_bf16(_i32_max(lo, hi))
            codes = Vec(_quant_bf16(y, scale))
            row = mb * BM + rt * 16 + l16
            valid = row < i32_m
            for jj in range_constexpr(2):
                col = (nb * HP + 2 * c2 + jj) * 16 + q16 * 4
                buffer_ops.buffer_store(
                    fx.Int32(codes[jj]), out_rsrc, row * (N // 2) + col, offset_is_bytes=True, mask=valid, **out_kw
                )
            buffer_ops.buffer_store(
                sb.to(fx.Int8), sc_rsrc, row * (N // 64) + nb * (HP // 2) + c2,
                offset_is_bytes=True, mask=valid & (q16 == 0), **out_kw,
            )  # fmt: skip

        def store_tile(i, v):
            rt, ni = i // NI, i % NI
            row = mb * BM + rt * 16 + l16
            valid = row < i32_m if FOLD == 1 else (row < i32_m) & (l16 < RPT)
            col = n16[ni] * 16 + q16 * 4
            if const_expr(SK_FUSE):
                buffer_ops.buffer_store(
                    v.bitcast(fx.Int32),
                    slab_rsrc,
                    ((kz * i32_m + row) * N + col) * 4,
                    offset_is_bytes=True,
                    mask=valid,
                    cache_modifier=SC1,
                )
            elif const_expr(KS == 1):
                buffer_ops.buffer_store(
                    v.to(fx.BFloat16).bitcast(fx.Int32),
                    out_rsrc,
                    (row * N + col) * 2,
                    offset_is_bytes=True,
                    mask=valid,
                    **out_kw,
                )
            else:
                buffer_ops.buffer_store(
                    v.bitcast(fx.Int32),
                    out_rsrc,
                    ((kz * i32_m + row) * N + col) * 4,
                    offset_is_bytes=True,
                    mask=valid,
                    **out_kw,
                )

        def store_tile_dyn(i, v, live):
            """store_tile for a wave-uniform runtime tile index (``live``: i < RT * NI): the
            lane's part of the offset is fixed, the tile's part goes in the scalar offset"""
            rt_d, ni_d = i // NI, i % NI
            esz = 2 if KS == 1 else 4
            # rows past M: the lane offset is masked out of the buffer (as store_tile's)
            valid = l16 < i32_m - (mb * BM + rt_d * 16)
            if const_expr(FOLD > 1):
                valid = valid & (l16 < RPT)
            if const_expr(live is not None):
                valid = valid & live
            data = v.to(fx.BFloat16).bitcast(fx.Int32) if KS == 1 else v.bitcast(fx.Int32)
            buffer_ops.buffer_store(
                data,
                out_rsrc,
                lane_out,
                offset_is_bytes=True,
                mask=valid,
                soffset_bytes=(rt_d * (16 * N) + ni_d * 16) * esz,
                **out_kw,
            )

        if const_expr(park_all):
            last = _pin_accumulators([acc[RT - 1][ni] for ni in range(NI)])
            for ni in range_constexpr(NI):
                park_tile((RT - 1) * NI + ni, last[ni])
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            if const_expr(KS == 1):
                lane_out = ((mb * BM + l16) * N + nb * (NI * 16) + q16 * 4) * 2
            else:
                lane_out = (((kz * i32_m) + mb * BM + l16) * N + nb * (NI * 16) + q16 * 4) * 4
            n_t = RT * NI
            n_j = -(-n_t // KW)  # tiles per wave (the last one only for waves < n_t % KW)
            red_base = park_base + wave * 1024  # this wave's first tile, K-wave 0's slot
            # a wave whose last tile index runs past the tiles reads tile n_t - 1 instead
            # and masks the store
            last_i = (wave + (n_j - 1) * KW < n_t).select(
                wave + (n_j - 1) * KW, fx.Int32(n_t - 1)
            )

            def lds_vec(addr):
                return Vec(
                    llvm.load(
                        T.vec(4, T.i32),
                        llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(addr))),
                        alignment=16,
                    )
                ).bitcast(fx.Float32)

            # all LDS reads of a group of tiles go out before its first store (the stores
            # would otherwise order every read behind the previous tile's store)
            GROUP = max(1, 24 // KW)
            for j0 in range_constexpr(0, n_j, GROUP):
                js = list(range(j0, min(n_j, j0 + GROUP)))
                parts = []
                for j in js:
                    if const_expr(j * KW + KW <= n_t):
                        base = red_base + j * KW * 1024
                    else:
                        base = park_base + last_i * 1024
                    parts.append([lds_vec(base + w * n_t * 1024) for w in range_constexpr(KW)])
                for j, ps in zip(js, parts):
                    v = ps[0]
                    for w in range_constexpr(1, KW):
                        v = v + ps[w]
                    full = j * KW + KW <= n_t
                    store_tile_dyn(
                        wave + j * KW if full else last_i,
                        v,
                        None if full else (wave + j * KW < n_t),
                    )
        else:
            flat = _pin_accumulators([acc[rt][ni] for rt in range(RT) for ni in range(NI)])
            if const_expr(KW > 1):
                # K-reduce: tile i belongs to K-wave i % KW; the other K-waves park their
                # partial sums of it in LDS, the owner adds the KW partials in wave order
                # (the sums one reducing wave made) and stores the tile
                lds_base = fx.Int32(fx.ptrtoint(smem.red.ptr))

                def lds_ptr(i, w):
                    """where K-wave w parks tile i (w != the owner)"""
                    o = own(i)
                    off = (w if w < o else w - 1) * RED_SLOT + i * 1024
                    return llvm.inttoptr(
                        ir.Type.parse("!llvm.ptr<3>"),
                        _raw(fx.Int64(lds_base + off + lane * 16)),
                    )

                def park(w):
                    for i in range_constexpr(RT * NI):
                        if const_expr(own(i) != w):
                            llvm.StoreOp(
                                _raw(Vec(flat[i]).bitcast(fx.Int32)),
                                lds_ptr(i, w),
                                alignment=16,
                            )

                def owned_value(v):
                    if const_expr(owner_read):
                        # Side-effecting AGPR reads stay after the ownership branch.
                        # Other rows retain their compiler-selected copies and ISA.
                        return Vec.from_elements([_acc_read(v, z) for z in range(4)], fx.Float32)
                    return Vec(v)

                def reduce_store(o):
                    sums = {}
                    for i in range_constexpr(RT * NI):
                        if const_expr(own(i) == o):
                            v = None
                            for w in range_constexpr(KW):
                                p = (
                                    owned_value(flat[i])
                                    if w == o
                                    else Vec(
                                        llvm.load(
                                            T.vec(4, T.i32), lds_ptr(i, w), alignment=16
                                        )
                                    ).bitcast(fx.Float32)
                                )
                                v = p if v is None else v + p
                            if const_expr(EPI_DIRECT):
                                sums[i] = v
                            else:
                                store_tile(i, v)
                    if const_expr(EPI_DIRECT):
                        for rt in range_constexpr(RT):
                            for c2 in range_constexpr(HP // 2):
                                if const_expr(c2 % KW == o):
                                    epi_block(rt, c2, sums)

                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                for w in range_constexpr(KW):
                    if wave == w:
                        park(w)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                for w in range_constexpr(KW):
                    if wave == w:
                        reduce_store(w)
            elif const_expr(EPI_DIRECT):
                for rt in range_constexpr(RT):
                    for c2 in range_constexpr(HP // 2):
                        epi_block(rt, c2, {i: Vec(flat[i]) for i in range(RT * NI)})
            else:
                for i in range_constexpr(RT * NI):
                    store_tile(i, Vec(flat[i]))

        if const_expr(SK_FUSE):
            def final_reduce(o):
                sums = {}
                for i in range_constexpr(RT * NI):
                    if const_expr(own(i) == o):
                        rt, ni = i // NI, i % NI
                        row = mb * BM + rt * 16 + l16
                        valid = row < i32_m
                        col = n16[ni] * 16 + q16 * 4
                        v = None
                        for z in range_constexpr(KS):
                            p = Vec(
                                buffer_ops.buffer_load(
                                    slab_rsrc, (z * i32_m + row) * N + col, vec_width=4, dtype=fx.Float32,
                                    mask=valid, cache_modifier=SC1,
                                )
                            )
                            v = p if v is None else v + p
                        if const_expr(EPI):
                            sums[i] = v
                        else:
                            buffer_ops.buffer_store(
                                v.to(fx.BFloat16).bitcast(fx.Int32), out_rsrc, (row * N + col) * 2,
                                offset_is_bytes=True, mask=valid, **out_kw,
                            )  # fmt: skip
                if const_expr(EPI):
                    for rt in range_constexpr(RT):
                        for c2 in range_constexpr(HP // 2):
                            if const_expr(c2 % KW == o):
                                epi_block(rt, c2, sums)

            # every slab store is acknowledged before the ticket is taken
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx == 0:
                inc = (kz == 0).select(fx.Int32(EPOCH - (KS - 1)), fx.Int32(1))
                old = _atomic_add(arg_cnt + fx.Int64(wg) * (CNT_STRIDE * 4), inc)
                smem.ticket.ptr[0] = (old & (EPOCH - 1)) + inc
            fx.gpu.barrier()
            last = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.ticket.ptr[0])))
            if last == EPOCH:
                for w in range_constexpr(KW):
                    if wave == w:
                        final_reduce(w)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        W: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W_scale: fx.Tensor,
        SC_OUT: fx.Tensor,
        CNT: fx.Tensor,
        SLAB: fx.Tensor,
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        n_mb: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, a_scale_bytes, n_mb, _addr(W), _addr(W_scale), _addr(A_scale), _addr(A),
            _addr(OUT), _addr(SC_OUT), _addr(CNT), _addr(SLAB),
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(64 * KW, 1, 1), stream=stream)  # fmt: skip

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    launch.k_split = KS
    launch.n_blocks = NB
    launch.block_m = BM
    launch.fold = FOLD
    return launch
