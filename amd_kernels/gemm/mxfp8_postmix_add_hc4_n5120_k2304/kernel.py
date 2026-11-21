# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""gfx950 MXFP8 x MXFP8 decode GEMM (N 5120, K 2304: the shared-expert down projection) whose epilogue adds the
routed experts' output and applies the mHC post-mix, one launch:

    y[m, n]      = bf16(bf16(sum_k A[m, k] * W[n, k]) + routed[m, n])    A: E4M3 codes + row-major E8M0 scales
    R'[m, j, n]  = bf16(p[m, j] * y + sum_h c[m, h, j] * R[m, h, n])       one fma chain, h = 0 ... 3

The mainloop, K-reduce, split-K hand-off and the post-mix / gate-partial epilogues are those of
``gemm/mxfp8_postmix_hc4_n5120_k8192`` (read its kernel docstring): W is the (16, 16) preshuffled weight with AITER's
shuffled scales, A is read as the producer wrote it (row-major codes and scales), a lane ends up with 4 consecutive
columns of one row of a tile and the post-mix is lane-local. K = 18 steps of 128: ``KS * KW`` is 1, 3 or 9. With ``GATES``
the epilogue also writes the collapse x1 and the per-wave records of the gate projection for
``norm/mhc_gate_finish_rmsnorm_h5120``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

N = 5120
K = 2304
H = N
NS = 4
PSLOT = 32  # floats of a partial record: [0:24] mixes, 24 sum R'^2, 25 sum x1^2
W_CACHE_MOD = 2  # non-temporal W loads: a decode GEMM reads every weight byte once
# VGPRs the kernel may hold at once. An MFMA operand is 8 contiguous VGPRs, so the register allocator runs out well
# before all 256 are live (clean at 199 live, parking operands in AGPRs at 202: a parked operand is copied back right
# before its MFMA and races the AGPR write).
VGPR_BUDGET = 192
EPI_GROUP = 3  # tiles per epilogue group: 28 VGPRs of operands each
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def _addr(t):
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def _lookahead(SW, RT, NI, RP, CP, budget):
    """(W steps, A blocks) issued ahead of the MFMAs: the longest common horizon whose registers fit (8 VGPRs per
    128-K operand, a scale dword per 32-row / 32-column group for every other W step, an offset vector per tile)."""
    budget = budget - (RT + NI)
    best = None
    for la_a in range(1, SW * RT + 1):
        la_w = min(SW, -(-la_a // RT))
        regs = 8 * (la_a + 1) + 8 * NI * (la_w + 1) + (CP + RP) * (la_w // 2 + 2)
        if regs > budget:
            break
        best = (la_w, la_a)
    assert best is not None, f"RT {RT} x NI {NI} leaves no VGPRs for the operands"
    return best


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _pack8(h0, h1):
    """two 4 x i32 halves -> one 8 x i32 MFMA operand (VGPRs 0-3 = K [16*klane, +16), 4-7 = K [64 + 16*klane, +16))."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements([a[k] for k in range(4)] + [b[k] for k in range(4)], fx.Int32)


def _mfma(w_op, a_op, acc, sw, sa, jw, ia, k2):
    """One scaled 16x16x128 fp8 MFMA, accumulator pinned in AGPR. Operand order (W, A): lane (q, r) accumulates
    out[row r, columns 4q .. 4q+3]. Scale byte of a dword = op_sel + 2 * op_sel_hi (16-row / 16-col half, 128-K half)."""
    src2 = "$0" if acc is not None else "0"
    asm = (
        f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
        f"op_sel:[{jw},{ia},0] op_sel_hi:[{k2},{k2},0] cbsz:0 blgp:0"
    )
    ops = [fx.as_ir_value(w_op), fx.as_ir_value(a_op), fx.as_ir_value(sw), fx.as_ir_value(sa)]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(fx.as_ir_value(acc))
        cons += ",0"
    return llvm.inline_asm(Vec.make_type(4, fx.Float32), ops, asm, cons, has_side_effects=True)


def _pin_accumulators(values):
    """Route the accumulators through an inline asm that outputs them again (tied ``=a`` / ``0``) with the XDL wait
    states inside: the reads that follow cannot be hoisted right behind the last inline-asm MFMA."""
    one = len(values) == 1
    ty = ir.Type.parse(
        "vector<4xf32>" if one else "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(values)) + ")>"
    )
    constraints = ",".join(["=a"] * len(values) + [str(i) for i in range(len(values))])
    r = llvm.inline_asm(ty, [_raw(v) for v in values], "s_nop 15\ns_nop 15", constraints, has_side_effects=True)
    if one:
        return [r]
    return [llvm.extractvalue(T.vec(4, T.f32), r, [i]) for i in range(len(values))]


def _swap16(v):
    """(this lane's, its neighbour's (lane ^ 16)) value of an i32 over the 16-lane rows."""
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _raw(fx.Int32(v))
    res = fx.rocdl.permlane16_swap(pair_ty, raw, raw, False, False)
    return fx.Int32(llvm.extractvalue(T.i32, res, [0])), fx.Int32(llvm.extractvalue(T.i32, res, [1]))


def _swap32(v):
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _raw(fx.Int32(v))
    res = fx.rocdl.permlane32_swap(pair_ty, raw, raw, False, False)
    return fx.Int32(llvm.extractvalue(T.i32, res, [0])), fx.Int32(llvm.extractvalue(T.i32, res, [1]))


def _pk_bf16(a, b):
    """Two fp32 -> two bf16 (round to nearest even) in a dword, ``a`` low."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def _unpack_bf16(dwords):
    out = []
    for w in dwords:
        w = fx.Int32(w)
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _units(count, kw, n_u, wave):
    """Epilogue unit indices of a wave over ``count`` units and their live masks (None: always live): a unit
    repeating the last one past the end is masked."""
    idx, live = [], []
    for u in range(n_u):
        i = wave + u * kw if kw > 1 else u
        lv = None if u * kw + kw <= count else (i < count)
        if lv is not None:
            i = lv.select(i, fx.Int32(count - 1))
        idx.append(i)
        live.append(lv)
    return idx, live


def compile_postmix_gemm(
    *, RT: int, NI: int, KW: int, KS: int = 1, PF: int = 0, GATES: bool = False, AUX: bool = False,
    sched_barriers: bool = True, W_CM: int | None = None,
):  # fmt: skip
    """Kernel for ``RT`` 16-row tiles per m-block, ``NI`` 16-column tiles per workgroup, ``KW`` K-waves and ``KS`` K
    splits over workgroups. Grid = ``tiles * KS`` workgroups of ``64 * KW`` threads (``tiles`` = m-blocks * launch.n_blocks,
    a multiple of 8 so that the splits of a tile share an XCD)."""
    w_cm = W_CACHE_MOD if W_CM is None else W_CM
    assert w_cm in (0, 1, 2), w_cm
    assert N % (16 * NI) == 0
    KSTEPS = K // 128
    assert KSTEPS % (KS * KW) == 0 and (KSTEPS // (KS * KW)) % 2 == 0, (KS, KW)  # KS * KW: 1, 3 or 9
    SW = KSTEPS // (KS * KW)  # steps per K-wave (even: starts on a 256-K scale block)
    BM = 16 * RT
    NB = N // (16 * NI)
    SCB = K // 256  # scale blocks (64 dwords) per 32-row / 32-column group
    RP = (RT + 1) // 2  # 32-row scale groups per workgroup
    CP = NI if NI % 2 else NI // 2
    LA_W, LA_A = _lookahead(SW, RT, NI, 4 * RP, CP, VGPR_BUDGET)
    NA = SW * RT
    n_t = RT * NI
    assert not AUX or GATES
    # Epilogue units of a wave: one tile (post-mix only), or with ``GATES`` one column tile of all RT row tiles, so that
    # the wave loads the gate-weight operands of its columns once for every row of the block; each wave then writes its
    # own partial record per row (NB * KW records).
    assert not GATES or KW <= NI, (NI, KW)
    n_u = -(-(NI if GATES else n_t) // KW)
    GP = 1 if GATES else max(1, min(n_u, EPI_GROUP))  # units whose operands are in flight at once
    N_G = -(-n_u // GP)
    assert 0 <= PF < SW
    LDS_BYTES = KW * n_t * 1024 if KW > 1 else 16
    assert LDS_BYTES + 16 <= 160 * 1024, "K-reduce scratch exceeds the LDS"

    @fx.struct
    class Shared:
        red: fx.Array[fx.Uint8, LDS_BYTES, 16]
        flag: fx.Array[fx.Uint8, 16, 16]

    name = (
        f"flydsl_mxfp8_postmix_add_k2304_rt{RT}_ni{NI}_kw{KW}_ks{KS}_pf{PF}_la{LA_W}x{LA_A}"
        + ("_gates" if GATES else "") + ("_aux" if AUX else "") + ("sb" if sched_barriers else "")
        + (f"_wcm{w_cm}" if w_cm != W_CACHE_MOD else "")
    )  # fmt: skip

    @flyc.kernel(name=name, known_block_size=[64 * KW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_as: fx.Int64,
        arg_a: fx.Int64,
        arg_res: fx.Int64,
        arg_rout: fx.Int64,
        arg_post: fx.Int64,
        arg_comb: fx.Int64,
        arg_out: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        arg_fn: fx.Int64,
        arg_pre: fx.Int64,
        arg_x1: fx.Int64,
        arg_mean: fx.Int64,
        arg_gp: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        if const_expr(KS > 1):
            # one XCD (workgroup % 8) per tile's splits: they share an L2, which publishes the partials
            tile = bx // (8 * KS) * 8 + bx % 8
            kz = bx // 8 % KS
            kwave = kz * KW + wave
        else:
            tile, kz, kwave = bx, 0, wave
        mb, nb = tile // NB, tile % NB
        lane_off = q16 * 256 + l16 * 16  # a lane's 16 B inside a 1 KB block
        k_base = kwave * (SW * 2048)  # this K-wave's first block pair (bytes)

        a_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_a, num_records_bytes=i32_m * K)
        w_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K)
        as_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_as, num_records_bytes=a_scale_bytes)
        ws_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_ws, num_records_bytes=(N // 32) * SCB * 256)
        res_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_res, num_records_bytes=i32_m * (NS * H * 2))
        rout_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_rout, num_records_bytes=i32_m * (H * 2))
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i32_m * (NS * H * 2))
        post_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_post, num_records_bytes=i32_m * (NS * 4))
        comb_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_comb, num_records_bytes=i32_m * (NS * NS * 4))
        if const_expr(GATES):
            fn_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_fn, num_records_bytes=(N // 16) * NS * 4 * 512)
            pre_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_pre, num_records_bytes=i32_m * (NS * 4))
            x1_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_x1, num_records_bytes=i32_m * (H * 2))
            gp_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_gp, num_records_bytes=i32_m * (NB * KW * PSLOT * 4)
            )
            if const_expr(AUX):
                mean_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_mean, num_records_bytes=i32_m * (H * 2))
        n16 = [nb * NI + ni for ni in range_constexpr(NI)]
        r16 = [mb * RT + rt for rt in range_constexpr(RT)]
        w_dw = [(n16[ni] * (16 * K) + k_base + lane_off) // 4 for ni in range_constexpr(NI)]
        a_dw = [((r16[rt] * 16 + l16) * K + kwave * (SW * 128) + q16 * 16) // 4 for rt in range_constexpr(RT)]
        sc_lane = q16 * 16 + l16
        sc_k = kwave * (SW // 2 * 64)  # this K-wave's first 256-K scale block (dwords)
        ws_dw = [
            ((n16[cp if NI % 2 else 2 * cp] // 2) * SCB) * 64 + sc_k + sc_lane for cp in range_constexpr(CP)
        ]
        rm_dw = [
            [((r16[t] * 16 + l16) * (K // 32) + kwave * (SW // 2) * 8) // 4 for t in (2 * rp, min(2 * rp + 1, RT - 1))]
            for rp in range_constexpr(RP)
        ]
        # a 16-column tile of an odd count sits in either half of its 32 group: move its bytes down
        w_shift = [(n16[ni] % 2) * 8 for ni in range_constexpr(NI)] if NI % 2 else None

        def load_pair(rsrc, dw, s):
            kw = dict(vec_width=4, dtype=fx.Int32, cache_modifier=w_cm)
            return _pack8(
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048, **kw),
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048 + 1024, **kw),
            )

        def load_ws(s):
            vals = [
                fx.Int32(buffer_ops.buffer_load(ws_rsrc, dw, vec_width=1, dtype=fx.Int32, soffset_bytes=(s // 2) * 256))
                for dw in ws_dw
            ]
            if const_expr(w_shift is None):
                return vals
            return [v.shrui(sh) for v, sh in zip(vals, w_shift)]

        def load_scales_rm(s):
            """Raw dwords of the row-major A scales of 256-K block s // 2, in the order (pair of row tiles, 128-K
            half, tile of the pair)."""
            return [
                fx.Int32(
                    buffer_ops.buffer_load(
                        as_rsrc, rm_dw[rp][t], vec_width=1, dtype=fx.Int32, soffset_bytes=(s // 2) * 8 + kh * 4
                    )
                )
                for rp in range_constexpr(RP)
                for kh in range_constexpr(2)
                for t in range_constexpr(2)
            ]

        def combine_rm(raw, rp):
            """The scale dword of row-tile pair rp: byte t + 2 * kh = klane q's byte of tile t of the pair, half kh."""
            sh = q16 * 8
            out = None
            for i in range_constexpr(4):
                b = (raw[4 * rp + i].shrui(sh)) & fx.Int32(0xFF)
                out = b if out is None else out | (b << (8 * i))
            return out

        def load_w_step(s):
            bb = [load_pair(w_rsrc, w_dw[ni], s) for ni in range_constexpr(NI)]
            if const_expr(s % 2 == 0):
                return bb, (load_ws(s), load_scales_rm(s))
            return bb, None

        def load_a(t):
            rt, s = t % RT, t // RT
            kw = dict(vec_width=4, dtype=fx.Int32)
            return _pack8(
                buffer_ops.buffer_load(a_rsrc, a_dw[rt], soffset_bytes=s * 128, **kw),
                buffer_ops.buffer_load(a_rsrc, a_dw[rt], soffset_bytes=s * 128 + 64, **kw),
            )

        def tile_rc(rt, ni):
            """(row, first column of the lane's 4) of the tile (``rt`` / ``ni`` trace-time ints or Int32)."""
            return mb * BM + rt * 16 + l16, (nb * NI + ni) * 16 + q16 * 4

        def load_res(rt, ni):
            row, col = tile_rc(rt, ni)
            return [
                Vec(buffer_ops.buffer_load(res_rsrc, (row * NS + j) * (H // 2) + col // 2, vec_width=2, dtype=fx.Int32))
                for j in range_constexpr(NS)
            ] + [Vec(buffer_ops.buffer_load(rout_rsrc, row * (H // 2) + col // 2, vec_width=2, dtype=fx.Int32))]  # fmt: skip

        def load_gates(rt):
            """post, comb (and with ``GATES`` the carried pre gate) of the lane's row."""
            row = mb * BM + rt * 16 + l16
            pv = Vec(buffer_ops.buffer_load(post_rsrc, row * NS, vec_width=4, dtype=fx.Float32))
            cv = [
                Vec(buffer_ops.buffer_load(comb_rsrc, row * (NS * NS) + NS * h, vec_width=4, dtype=fx.Float32))
                for h in range_constexpr(NS)
            ]
            if const_expr(GATES):
                return pv, cv, Vec(buffer_ops.buffer_load(pre_rsrc, row * NS, vec_width=4, dtype=fx.Float32))
            return pv, cv

        def load_b(ni):
            """The gate-weight operands of column tile ``ni``: [stream][hi 0-15, lo 0-15, hi 16-23, lo 16-23]."""
            cg = nb * NI + ni
            return [
                [
                    Vec(buffer_ops.buffer_load(fn_rsrc, ((cg * NS + s_) * 4 + o) * 128 + lane * 2, vec_width=2,
                                               dtype=fx.Int32))
                    for o in range_constexpr(4)
                ]
                for s_ in range_constexpr(NS)
            ]  # fmt: skip

        def post_mix(rt, ni, v, rs, pv, cv, valid):
            """New residual of the tile (stored) from its fp32 sums ``v``; returns its 4 streams as bf16 pairs."""
            row, col = tile_rc(rt, ni)
            yb = Vec(v.to(fx.BFloat16).bitcast(fx.Int32))
            ys = _unpack_bf16([yb[0], yb[1]])
            ro = _unpack_bf16([fx.Int32(rs[NS][0]), fx.Int32(rs[NS][1])])
            y = _unpack_bf16([_pk_bf16(ys[0] + ro[0], ys[1] + ro[1]), _pk_bf16(ys[2] + ro[2], ys[3] + ro[3])])
            f = [_unpack_bf16([fx.Int32(r[0]), fx.Int32(r[1])]) for r in rs[:NS]]
            packed = []
            for j in range_constexpr(NS):
                vals = []
                for e in range_constexpr(4):
                    a = pv[j] * y[e]
                    for h in range_constexpr(NS):
                        a = fx.math.fma(cv[h][j], f[h][e], a)
                    vals.append(a)
                pk = [_pk_bf16(vals[0], vals[1]), _pk_bf16(vals[2], vals[3])]
                packed.append(pk)
                buffer_ops.buffer_store(
                    Vec.from_elements(pk, fx.Int32), out_rsrc, (row * NS + j) * (H // 2) + col // 2, mask=valid
                )
            return packed

        def row_sum(x):
            """Sum over the 4 lanes (klanes) of a row."""
            for swap in (_swap32, _swap16):
                a, b = swap(x.bitcast(fx.Int32))
                x = a.bitcast(fx.Float32) + b.bitcast(fx.Float32)
            return x

        def mfma_gate(r4, fn4, acc):
            """D[mix][row] += fn[mix][k] * R'[row][k]: lane (q, r) ends up with mixes 4q .. 4q + 3 of row r."""
            return Vec(fx.rocdl.mfma_f32_16x16x16bf16_1k(Vec.make_type(4, fx.Float32), [fn4, r4, acc, 0, 0, 0]))

        def gates_tile(rt, ni, v, rs, pv, cv, qv, bops, st, valid, live):
            """Post-mix, collapse x1, square sums and the gate-projection MFMAs of one tile into the row tile's
            accumulators ``st``."""
            row, col = tile_rc(rt, ni)
            pk = post_mix(rt, ni, v, rs, pv, cv, valid)
            gf = [_unpack_bf16(p) for p in pk]
            x1v = []
            for e in range_constexpr(4):
                a = qv[0] * gf[0][e]
                for s_ in range_constexpr(1, NS):
                    a = fx.math.fma(qv[s_], gf[s_][e], a)
                x1v.append(a)
            x1p = [_pk_bf16(x1v[0], x1v[1]), _pk_bf16(x1v[2], x1v[3])]
            buffer_ops.buffer_store(Vec.from_elements(x1p, fx.Int32), x1_rsrc, row * (H // 2) + col // 2, mask=valid)
            x1f = _unpack_bf16(x1p)
            sqr, sqx = st[2], st[3]
            nsqr, nsqx = sqr, sqx
            for e in range_constexpr(4):
                nsqx = fx.math.fma(x1f[e], x1f[e], nsqx)
                for s_ in range_constexpr(NS):
                    nsqr = fx.math.fma(gf[s_][e], gf[s_][e], nsqr)
            sqr, sqx = (nsqr, nsqx) if live is None else (live.select(nsqr, sqr), live.select(nsqx, sqx))
            if const_expr(AUX):
                mv = [((gf[0][e] + gf[1][e]) + gf[2][e] + gf[3][e]) * fx.Float32(0.25) for e in range_constexpr(4)]
                buffer_ops.buffer_store(
                    Vec.from_elements([_pk_bf16(mv[0], mv[1]), _pk_bf16(mv[2], mv[3])], fx.Int32), mean_rsrc,
                    row * (H // 2) + col // 2, mask=valid,
                )  # fmt: skip
            acc12, acc3 = st[0], st[1]
            for s_ in range_constexpr(NS):
                ap = pk[s_] if live is None else [live.select(w, fx.Int32(0)) for w in pk[s_]]  # a dead unit adds 0
                a4 = Vec(Vec.from_elements(ap, fx.Int32)).bitcast(fx.Int16)
                for o in range_constexpr(4):
                    b4 = Vec(bops[s_][o]).bitcast(fx.Int16)
                    if const_expr(o < 2):
                        acc12 = mfma_gate(a4, b4, acc12)
                    else:
                        acc3 = mfma_gate(a4, b4, acc3)
            st[0], st[1], st[2], st[3] = acc12, acc3, sqr, sqx

        def write_records(rt, st):
            """The row tile's partial record of this wave, one 128 B record per row: lane (q, r) holds mixes 4q .. 4q + 3
            (and 16 + 4q .. for q < 2) of row r, lane q = 2 the square sums, lane q = 3 the zero padding."""
            acc12, acc3, sqr, sqx = st
            sqr, sqx = row_sum(sqr), row_sum(sqx)
            row = mb * BM + rt * 16 + l16
            base = (row * (NB * KW) + nb * KW + wave) * PSLOT
            ok = row < i32_m
            buffer_ops.buffer_store(acc12, gp_rsrc, base + 4 * q16, mask=ok)
            buffer_ops.buffer_store(acc3, gp_rsrc, base + 16 + 4 * q16, mask=ok & (q16 < 2))
            zero = fx.Float32(0.0)
            buffer_ops.buffer_store(Vec.from_elements([sqr, sqx, zero, zero], fx.Float32), gp_rsrc, base + 24,
                                    mask=ok & (q16 == 2))
            buffer_ops.buffer_store(Vec.filled(4, 0.0, fx.Float32), gp_rsrc, base + 28, mask=ok & (q16 == 3))

        # Units: a tile (it = wave + u * KW over the n_t tiles, the last unit of a wave past the end repeating the last
        # tile, masked) or, with GATES, the column tile ni = wave + u * KW of all row tiles.
        unit_idx, unit_live = _units(NI if GATES else n_t, KW, n_u, wave)

        def unit_tiles(u):
            """(rt, ni, linear tile index) of the tiles of unit u."""
            if const_expr(GATES):
                return [(rt, unit_idx[u], rt * NI + unit_idx[u]) for rt in range(RT)]
            return [(unit_idx[u] // NI, unit_idx[u] % NI, unit_idx[u])]

        def prefetch(u):
            """Global loads of a unit's epilogue operands."""
            if const_expr(GATES):
                return [load_res(rt, ni) for rt, ni, _ in unit_tiles(u)], load_b(unit_idx[u])
            (rt, ni, _), = unit_tiles(u)
            return load_res(rt, ni), load_gates(rt)

        pfs = [None] * n_u

        def prefetch_group(g):
            for u in range_constexpr(g * GP, min(n_u, (g + 1) * GP)):
                pfs[u] = prefetch(u)

        if const_expr(KW > 1):
            park_base = fx.Int32(fx.ptrtoint(smem.red.ptr)) + lane * 16

            def park_tile(i, v):
                llvm.StoreOp(
                    _raw(Vec(v).bitcast(fx.Int32)),
                    llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(park_base + (wave * n_t + i) * 1024))),
                    alignment=16,
                )

        # Both operands run ahead of the MFMAs: W by LA_W steps, A by LA_A (step, row tile) blocks in the order the
        # MFMAs consume them.
        acc = [[None] * NI for _ in range(RT)]
        w_q = [None] * SW
        a_q = [None] * NA
        for s in range_constexpr(LA_W):
            w_q[s] = load_w_step(s)
        for t in range_constexpr(LA_A):
            a_q[t] = load_a(t)
        if const_expr(sched_barriers):
            fx.rocdl.sched_barrier(0)
        sw = sa = sa_raw = None
        for s in range_constexpr(SW):
            if const_expr(s + LA_W < SW):
                w_q[s + LA_W] = load_w_step(s + LA_W)
            if const_expr(s == SW - 1 - PF and PF > 0):
                prefetch_group(0)
            bb, sc = w_q[s]
            if const_expr(s % 2 == 0):
                sw, sa_raw = sc
                sa = [None] * RP
            for rt in range_constexpr(RT):
                t = s * RT + rt
                if const_expr(t + LA_A < NA):
                    a_q[t + LA_A] = load_a(t + LA_A)
                if const_expr(sa[rt // 2] is None):
                    sa[rt // 2] = combine_rm(sa_raw, rt // 2)
                a_op, sa_v = a_q[t], sa[rt // 2]
                a_q[t] = None
                for ni in range_constexpr(NI):
                    acc[rt][ni] = _mfma(
                        bb[ni], a_op, acc[rt][ni], sw[ni if NI % 2 else ni // 2], sa_v, 0 if NI % 2 else ni % 2,
                        rt % 2, s % 2,
                    )  # fmt: skip
                if const_expr(sched_barriers):
                    fx.rocdl.sched_barrier(0)
                if const_expr(KW > 1 and s == SW - 1 and rt >= 1):
                    # row tile rt - 1 is final: park it behind row tile rt's MFMAs (the wait states between the XDL
                    # write and the LDS read, which the compiler cannot see through the asm)
                    if const_expr(not sched_barriers):
                        fx.rocdl.sched_barrier(0)
                    for ni in range_constexpr(NI):
                        park_tile((rt - 1) * NI + ni, acc[rt - 1][ni])
                    fx.rocdl.sched_barrier(0)
            w_q[s] = None

        if const_expr(PF == 0):
            prefetch_group(0)
        if const_expr(KS > 1):
            part_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_part, num_records_bytes=0x7FFFFFF0)
            flag_addr = fx.Int32(fx.ptrtoint(smem.flag.ptr))

            def slot(s, it):
                """float offset of the lane's 4 sums of tile ``it`` in split ``s`` of this tile's partial slab"""
                return (((tile * KS + s) * n_t + it) * 64 + lane) * 4

        if const_expr(KW > 1):
            last = _pin_accumulators([acc[RT - 1][ni] for ni in range(NI)])
            for ni in range_constexpr(NI):
                park_tile((RT - 1) * NI + ni, last[ni])
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()

            def lds_vec(addr):
                return Vec(
                    llvm.load(
                        T.vec(4, T.i32),
                        llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(addr))),
                        alignment=16,
                    )
                ).bitcast(fx.Float32)

            def group_sums(us):
                """fp32 sums of the tiles of units ``us``: all LDS reads of the group go out before its first use."""
                parts = []
                for u in us:
                    for _rt, _ni, it in unit_tiles(u):
                        parts.append([lds_vec(park_base + (w * n_t + it) * 1024) for w in range_constexpr(KW)])
                out = []
                for ps in parts:
                    v = ps[0]
                    for w in range_constexpr(1, KW):
                        v = v + ps[w]
                    out.append(v)
                return out
        else:
            flat = _pin_accumulators([acc[rt][ni] for rt in range(RT) for ni in range(NI)])

            def group_sums(us):
                return [Vec(flat[it]) for u in us for _rt, _ni, it in unit_tiles(u)]

        def new_state():
            return [[Vec.filled(4, 0.0, fx.Float32), Vec.filled(4, 0.0, fx.Float32), fx.Float32(0.0), fx.Float32(0.0)]
                    for _ in range(RT)]  # fmt: skip

        def process(u, sums, pf, st):
            """The epilogue of unit u from the fp32 sums of its tiles."""
            if const_expr(GATES):
                rs, bops = pf
                for (rt, ni, _it), v, r in zip(unit_tiles(u), sums, rs):
                    row, _c = tile_rc(rt, ni)
                    pv, cv, qv = load_gates(rt)
                    lv = unit_live[u]
                    gates_tile(rt, ni, v, r, pv, cv, qv, bops, st[rt], (row < i32_m) if lv is None else (row < i32_m) & lv, lv)
                return
            (rt, ni, _it), = unit_tiles(u)
            row, _c = tile_rc(rt, ni)
            valid = row < i32_m
            if const_expr(unit_live[u] is not None):
                valid = valid & unit_live[u]
            rs, (pv, cv) = pf
            post_mix(rt, ni, sums[0], rs, pv, cv, valid)

        def finish(st):
            if const_expr(GATES):
                for rt in range_constexpr(RT):
                    write_records(rt, st[rt])

        st0 = new_state() if (GATES and KS == 1) else None
        for g in range_constexpr(N_G):
            us = list(range(g * GP, min(n_u, (g + 1) * GP)))
            if const_expr(KS == 1 and g + 1 < N_G):
                prefetch_group(g + 1)
            sums = group_sums(us)
            k = 0
            for u in us:
                tl = unit_tiles(u)
                if const_expr(KS == 1):
                    process(u, sums[k : k + len(tl)], pfs[u], st0)
                else:
                    for (_rt, _ni, it), v in zip(tl, sums[k : k + len(tl)]):
                        buffer_ops.buffer_store(v, part_rsrc, slot(kz, it), mask=unit_live[u])
                k += len(tl)
        if const_expr(KS == 1):
            finish(st0)

        if const_expr(KS > 1):
            # the tile's last split to arrive sums the partials in split order and runs the epilogue; the counter
            # re-arms itself (it is not an output)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            counter = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), arg_cnt + fx.Int64(tile) * 4)
            if tx == 0:
                arrived = fx.llvm.atomic_add(
                    counter, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.AcqRel
                )
                llvm.StoreOp(_raw(arrived), llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(flag_addr))),
                             alignment=4)  # fmt: skip
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            seen = fx.Int32(llvm.load(T.i32, llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(flag_addr))),
                                      alignment=4))  # fmt: skip
            if seen == KS - 1:
                st1 = new_state() if GATES else None
                for g in range_constexpr(N_G):
                    if const_expr(g + 1 < N_G):
                        prefetch_group(g + 1)
                    for u in range_constexpr(g * GP, min(n_u, (g + 1) * GP)):
                        totals = []
                        for _rt, _ni, it in unit_tiles(u):
                            total = None
                            for s_ in range_constexpr(KS):
                                # sc0 sc1: read past this CU's L1, which never held the others' partials
                                p = Vec(buffer_ops.buffer_load(part_rsrc, slot(s_, it), vec_width=4, dtype=fx.Float32,
                                                               cache_modifier=17))  # fmt: skip
                                total = p if total is None else total + p
                            totals.append(total)
                        process(u, totals, pfs[u], st1)
                finish(st1)
                if tx == 0:
                    fx.ptr_store(fx.Int32(0), counter)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        W: fx.Tensor,
        R: fx.Tensor,
        ROUT: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W_scale: fx.Tensor,
        POST: fx.Tensor,
        COMB: fx.Tensor,
        PART: fx.Tensor,
        CNT: fx.Tensor,
        FN: fx.Tensor,
        PRE: fx.Tensor,
        X1: fx.Tensor,
        MEAN: fx.Tensor,
        GP: fx.Tensor,
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, a_scale_bytes, _addr(W), _addr(W_scale), _addr(A_scale), _addr(A), _addr(R), _addr(ROUT), _addr(POST),
            _addr(COMB), _addr(OUT), _addr(PART), _addr(CNT), _addr(FN), _addr(PRE), _addr(X1), _addr(MEAN),
            _addr(GP),
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(64 * KW, 1, 1), stream=stream)  # fmt: skip

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    launch.n_blocks = NB
    launch.block_m = BM
    launch.k_split = KS
    return launch
