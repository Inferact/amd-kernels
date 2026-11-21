# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DCP8 combine in one launch: W_UV up-projection of all 96 heads, push to the owner rank, LSE-weighted combine.

Hand-off protocol (control words in uncached memory, one block per rank; the data pools are uncached too, so the
owner's reads never hit a stale line of a peer's write, and mapped on the peers by IPC): a call takes
``seq = counter + 1`` and the slot ``seq & 1`` of every pool.

Compute workgroups (192: XCD x serves the 12 heads of owner rank x, two 64-column halves per head): four waves split
the head's K = 512, read W_UV^T once and the partial output one 16-token tile at a time (the next two tiles
prefetched);
the waves' fp32 partial sums meet in LDS and are added in wave order, rounded to bf16 and stored with system-scope
stores to the owner's slot, contiguous per (source, head, half). The workgroups of the first column half also store
the masked LSE of their rows. After its last tile a workgroup waits for its stores, then stores ``seq`` into its own
word of the owner's flag array (``flags[src][workgroup]``): no atomics.

Combine workgroups (after the compute workgroups in the grid, so they never starve them): wait for the 8 x 24 flags,
read the 8 sources' rows (local memory, system-coherent loads), weigh by ``exp(lse_s - max) / sum`` and sum in source
order in fp32. Every workgroup stores ``seq`` into its word of the start array after reading the counter; one extra
workgroup (dispatched last, so every other one has started) waits for all start words and then bumps the counter, so
a CUDA graph replays with no host state and the grid may change between calls. A slot is rewritten two calls later,
after every peer's flags of the call in between, which it sets after its own previous call finished.

Control block (int32 words): flags [0, 192), counter 700, start words 1024 + workgroup.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .reference import H, HPR, L, V, WORLD

NT = 256
UNITS = 2 * H
PER_SRC = UNITS // WORLD
PADN = 68
SYS = 0x11
CTL_BYTES = 8192
FLAG_WORD, SEQ_WORD, START_WORD = 0, 700, 1024
ROWS_PER_WG = 16
PREFETCH = 2
LOG2E = 1.4426950408889634


def _ptr(elem, addr_i64, align):
    ty = fx.PointerType.get(elem, address_space=fx.AddressSpace.Global, alignment=align)
    return fx.inttoptr(ty, fx.Int64(addr_i64))


def _volatile_load_v4(addr_i64):
    """A system-coherent 16-byte load the compiler neither hoists out of a spin loop nor drops."""
    ty = ir.VectorType.get([4], T.i32)
    return Vec(_llvm.LoadOp(ty, _ptr(T.i32, addr_i64, 16).llvm_ptr, alignment=16, volatile_=True).result)


def _pk_bf16(a, b):
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def grid_of(m_tiles):
    return UNITS + -(-m_tiles * 16 * HPR // ROWS_PER_WG) + 1


@launcher
def compile_combine(m_tiles: int, base_e: bool, gated: bool = False):
    """``m_tiles``: 16-token tiles of the padded token count. ``gated``: the output is ``bf16(bf16(combine) *
    sigmoid(gate))`` with a bf16 ``gate`` [T, 12 * 128]."""
    assert 1 <= m_tiles <= 12
    base = grid_of(m_tiles) - 1
    name = f"dcp8_lse_combine_vup_mla_h12_kv512_m{m_tiles}" + ("_e" if base_e else "_2") + ("_gate" if gated else "")

    @fx.struct
    class Shared:
        red: fx.Array[fx.Int32, 4 * 16 * PADN, 16]
        valid: fx.Array[fx.Int32, NT]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # [T, >=96, 512] bf16 partial output
        i32_sxt: fx.Int32,  # token stride (dwords)
        i32_sxh: fx.Int32,  # head stride (dwords)
        arg_lse: fx.Int64,  # [T, >=96] fp32
        i32_slt: fx.Int32,
        i32_slh: fx.Int32,
        arg_wt: fx.Int64,  # [96, 128, 512] bf16 (W_UV transposed)
        arg_seq: fx.Int64,  # [num_seqs] int32 local seq lens
        arg_qsl: fx.Int64,  # [num_seqs + 1] int32
        i32_nseq: fx.Int32,
        arg_out: fx.Int64,  # [T, 12, 128] bf16
        arg_gate: fx.Int64,  # gated: [T, 1536] bf16
        arg_tab: fx.Int64,  # int64 [16]: pool and control block of each rank
        i32_m: fx.Int32,
        i32_rank: fx.Int32,
        i32_mmax: fx.Int32,
        i64_slot_bytes: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        tab = _ptr(T.i64, arg_tab, 8)
        xcd = bid % 8
        pool_me = fx.Int64(tab[i32_rank])
        me_ctl = fx.Int64(tab[8 + i32_rank])
        pool_dst = fx.Int64(tab[xcd])
        ctl_dst = fx.Int64(tab[8 + xcd])
        ctl = buffer_ops.create_buffer_resource_from_addr(me_ctl, num_records_bytes=CTL_BYTES)
        neg_inf = fx.Float32(float("-inf"))
        rows = i32_mmax * HPR
        lse_off = fx.Int64(WORLD) * fx.Int64(rows) * 256

        def read_seq():
            return fx.Int32(
                fx.rocdl.readfirstlane(
                    T.i32,
                    fx.Int32(buffer_ops.buffer_load(ctl, SEQ_WORD, vec_width=1, dtype=fx.Int32, cache_modifier=SYS)) + 1,
                )
            )

        if bid < UNITS:
            idx = bid // 8
            j = idx // 2
            ns = idx % 2
            h = xcd * HPR + j
            lm = lane % 16
            lg = lane // 16

            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_wt + fx.Int64(h * (V * L * 2) + ns * (64 * L * 2)), num_records_bytes=64 * L * 2
            )
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x + fx.Int64(h * i32_sxh) * 4, num_records_bytes=((i32_m - 1) * i32_sxt + L // 2) * 4
            )

            def load_x(p):
                return [
                    Vec(
                        buffer_ops.buffer_load(
                            xr, (lm + p * 16) * i32_sxt + w * 64 + ks * 16 + lg * 4, vec_width=4, dtype=fx.Int32
                        )
                    )
                    for ks in range_constexpr(4)
                ]

            wv = [
                [
                    Vec(
                        buffer_ops.buffer_load(
                            wr, (nt * 16 + lm) * (L // 2) + w * 64 + ks * 16 + lg * 4, vec_width=4, dtype=fx.Int32
                        )
                    )
                    for nt in range_constexpr(4)
                ]
                for ks in range_constexpr(4)
            ]
            xvs = [load_x(q) for q in range_constexpr(min(PREFETCH, m_tiles))]

            seq = read_seq()
            dst = pool_dst + fx.Int64(seq & 1) * i64_slot_bytes
            if tx == 0:
                buffer_ops.buffer_store(seq, ctl, START_WORD + bid, cache_modifier=SYS)

            sq = _ptr(T.i32, arg_qsl, 4)
            sl = _ptr(T.i32, arg_seq, 4)
            live = tx < i32_nseq
            at = live.select(tx, fx.Int32(0))
            s0 = fx.Int32(sq[at])
            e0 = fx.Int32(sq[at + 1])
            l0 = fx.Int32(sl[at])
            er = fx.Int32(sq[i32_nseq])
            lxs = [
                fx.Float32(
                    _ptr(T.f32, arg_lse, 4)[
                        ((p * 16 + tx < i32_m) & (tx < 16)).select(p * 16 + tx, fx.Int32(0)) * i32_slt + h * i32_slh
                    ]
                )
                for p in range_constexpr(m_tiles)
            ]

            dr = buffer_ops.create_buffer_resource_from_addr(dst, num_records_bytes=rows * (WORLD * 256))
            lr = buffer_ops.create_buffer_resource_from_addr(dst + lse_off, num_records_bytes=rows * (WORLD * 4))
            red = fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, smem.red.ptr), fx.make_layout(4 * 16 * PADN, 1)),
                fx.make_layout(4, 1),
            )
            atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
            for p in range_constexpr(m_tiles):
                xn = load_x(p + PREFETCH) if p + PREFETCH < m_tiles else None
                xv = xvs.pop(0)
                if xn is not None:
                    xvs.append(xn)
                acc = [Vec.filled(4, 0.0, fx.Float32) for nt in range_constexpr(4)]
                for ks in range_constexpr(4):
                    for nt in range_constexpr(4):
                        acc[nt] = Vec(
                            fx.rocdl.mfma_f32_16x16x32_bf16(
                                Vec.make_type(4, fx.Float32),
                                [wv[ks][nt].bitcast(fx.BFloat16), xv[ks].bitcast(fx.BFloat16), acc[nt], 0, 0, 0],
                            )
                        )
                for nt in range_constexpr(4):
                    rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    rm.store(acc[nt].bitcast(fx.Int32))
                    fx.copy(atom, rm, fx.slice(red, (None, (w * 16 + lm) * (PADN // 4) + nt * 4 + lg)))
                fx.gpu.barrier()
                row = tx // 8
                c8 = tx % 8
                t = p * 16 + row
                if (row < 16) & (t < i32_m):
                    flat = []
                    for q in range_constexpr(2):
                        s4 = None
                        for ww in range_constexpr(4):
                            rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            fx.copy(atom, fx.slice(red, (None, (ww * 16 + row) * (PADN // 4) + c8 * 2 + q)), rm)
                            v4 = Vec(rm.load()).bitcast(fx.Float32)
                            s4 = v4 if s4 is None else s4 + v4
                        flat += [s4[0], s4[1], s4[2], s4[3]]
                    pk = [_pk_bf16(flat[2 * e], flat[2 * e + 1]) for e in range_constexpr(4)]
                    buffer_ops.buffer_store(
                        Vec.from_elements(pk, fx.Int32),
                        dr,
                        ((((i32_rank * HPR + j) * 2 + ns) * i32_mmax + t) * 32 + c8 * 4),
                        cache_modifier=SYS,
                    )
                if const_expr(p == 0):
                    if ns == 0:
                        end_row = (i32_nseq > 0).select(er, fx.Int32(0x7FFFFFFF))
                        smem.valid.ptr[tx] = (tx < end_row).select(fx.Int32(1), fx.Int32(0))
                        fx.gpu.barrier()
                        if live & (l0 == 0):
                            rr = s0
                            while rr < e0:
                                if rr < NT:
                                    smem.valid.ptr[rr] = fx.Int32(0)
                                rr = rr + 1
                        fx.gpu.barrier()
                if ns == 0:
                    lt = p * 16 + tx
                    if (tx < 16) & (lt < i32_m):
                        bad = (lxs[p] != lxs[p]) | (lxs[p] == fx.Float32(float("inf"))) | (fx.Int32(smem.valid.ptr[lt]) == 0)
                        buffer_ops.buffer_store(
                            bad.select(neg_inf, lxs[p]), lr, ((lt * HPR + j) * WORLD + i32_rank), cache_modifier=SYS
                        )
                if const_expr(p + 1 < m_tiles):
                    fx.gpu.barrier()
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx == 0:
                fr = buffer_ops.create_buffer_resource_from_addr(ctl_dst, num_records_bytes=CTL_BYTES)
                buffer_ops.buffer_store(seq, fr, FLAG_WORD + i32_rank * PER_SRC + idx, cache_modifier=SYS)

        if bid == base:
            seq = read_seq()
            for k in range_constexpr(-(-base // (NT * 4))):
                wid = (tx + k * NT) * 4
                lane_ok = wid < base
                addr = me_ctl + (START_WORD + lane_ok.select(wid, fx.Int32(0))) * 4
                fv = _volatile_load_v4(addr)
                pending = lane_ok & (
                    ((fv[0] - seq) < 0)
                    | ((wid + 1 < base) & ((fv[1] - seq) < 0))
                    | ((wid + 2 < base) & ((fv[2] - seq) < 0))
                    | ((wid + 3 < base) & ((fv[3] - seq) < 0))
                )
                while pending:
                    fv = _volatile_load_v4(addr)
                    pending = lane_ok & (
                        ((fv[0] - seq) < 0)
                        | ((wid + 1 < base) & ((fv[1] - seq) < 0))
                        | ((wid + 2 < base) & ((fv[2] - seq) < 0))
                        | ((wid + 3 < base) & ((fv[3] - seq) < 0))
                    )
            fx.gpu.barrier()
            if tx == 0:
                buffer_ops.buffer_store(seq, ctl, SEQ_WORD, cache_modifier=SYS)

        if (bid >= UNITS) & (bid < base):
            r0 = (bid - UNITS) * ROWS_PER_WG
            seq = read_seq()
            mine = pool_me + fx.Int64(seq & 1) * i64_slot_bytes
            if tx == 0:
                buffer_ops.buffer_store(seq, ctl, START_WORD + bid, cache_modifier=SYS)
            dr = buffer_ops.create_buffer_resource_from_addr(mine, num_records_bytes=rows * (WORLD * 256))
            lr = buffer_ops.create_buffer_resource_from_addr(mine + lse_off, num_records_bytes=rows * (WORLD * 4))
            outr = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i32_m * (HPR * V * 2))
            sub = tx % 16
            total = i32_m * HPR
            gvs = []
            if const_expr(gated):
                gr = buffer_ops.create_buffer_resource_from_addr(arg_gate, num_records_bytes=i32_m * (HPR * V * 2))
                for it in range_constexpr(ROWS_PER_WG // 16):
                    row = r0 + it * 16 + tx // 16
                    gvs.append(Vec(buffer_ops.buffer_load(gr, row * 64 + sub * 4, vec_width=4, dtype=fx.Int32, mask=row < total)))
            if tx < 64:
                lane_ok = tx < WORLD * (PER_SRC // 4)
                word = FLAG_WORD + (tx // (PER_SRC // 4)) * PER_SRC + (tx % (PER_SRC // 4)) * 4
                word = lane_ok.select(word, fx.Int32(0))
                fv = _volatile_load_v4(me_ctl + word * 4)
                pending = lane_ok & (((fv[0] - seq) < 0) | ((fv[1] - seq) < 0) | ((fv[2] - seq) < 0) | ((fv[3] - seq) < 0))
                while pending:
                    fv = _volatile_load_v4(me_ctl + word * 4)
                    pending = lane_ok & (((fv[0] - seq) < 0) | ((fv[1] - seq) < 0) | ((fv[2] - seq) < 0) | ((fv[3] - seq) < 0))
            fx.gpu.barrier()
            loads = []
            for it in range_constexpr(ROWS_PER_WG // 16):
                row = r0 + it * 16 + tx // 16
                ok = row < total
                rt = row // HPR
                rj = row % HPR
                vs = [
                    Vec(
                        buffer_ops.buffer_load(
                            dr,
                            ((((s_ * HPR + rj) * 2 + sub // 8) * i32_mmax + rt) * 32 + (sub % 8) * 4),
                            vec_width=4,
                            dtype=fx.Int32,
                            mask=ok,
                            cache_modifier=SYS,
                        )
                    )
                    for s_ in range_constexpr(WORLD)
                ]
                l4 = [
                    Vec(buffer_ops.buffer_load(lr, row * WORLD + 4 * q, vec_width=4, dtype=fx.Float32, mask=ok, cache_modifier=SYS))
                    for q in range_constexpr(2)
                ]
                ls = [fx.Float32(l4[s_ // 4][s_ % 4]) for s_ in range_constexpr(WORLD)]
                loads.append((row, ok, vs, ls, gvs[it] if const_expr(gated) else None))
            for row, ok, vs, ls, gv in loads:
                m = ls[0]
                for s_ in range_constexpr(1, WORLD):
                    m = (ls[s_] > m).select(ls[s_], m)
                m = (m == neg_inf).select(fx.Float32(0.0), m)
                scale = fx.Float32(LOG2E if base_e else 1.0)
                es = [fx.Float32(fx.math.exp2((ls[s_] - m) * scale)) for s_ in range_constexpr(WORLD)]
                den = es[0]
                for s_ in range_constexpr(1, WORLD):
                    den = den + es[s_]
                inv = fx.Float32(fx.rocdl.rcp(T.f32, fx.as_ir_value(den)))
                acc = [fx.Float32(0.0)] * 8
                for s_ in range_constexpr(WORLD):
                    ws = es[s_] * inv
                    ws = (ws != ws).select(fx.Float32(0.0), ws)
                    for k in range_constexpr(4):
                        d = fx.Int32(vs[s_][k])
                        lo = (d << 16).bitcast(fx.Float32)
                        hi = (d & fx.Int32(-65536)).bitcast(fx.Float32)
                        acc[2 * k] = acc[2 * k] + (ws == 0.0).select(fx.Float32(0.0), lo * ws)
                        acc[2 * k + 1] = acc[2 * k + 1] + (ws == 0.0).select(fx.Float32(0.0), hi * ws)
                pk = [_pk_bf16(acc[2 * k], acc[2 * k + 1]) for k in range_constexpr(4)]
                if const_expr(gated):
                    gated_pk = []
                    for k in range_constexpr(4):
                        a, g = fx.Int32(pk[k]), fx.Int32(gv[k])
                        outs = []
                        for x, y in (((a << 16), (g << 16)), ((a & fx.Int32(-65536)), (g & fx.Int32(-65536)))):
                            xf, yf = x.bitcast(fx.Float32), y.bitcast(fx.Float32)
                            outs.append(xf * (fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(-yf))))
                        gated_pk.append(_pk_bf16(outs[0], outs[1]))
                    pk = gated_pk
                buffer_ops.buffer_store(Vec.from_elements(pk, fx.Int32), outr, row * 64 + sub * 4, mask=ok)

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        i32_sxt: fx.Int32,
        i32_sxh: fx.Int32,
        arg_lse: fx.Int64,
        i32_slt: fx.Int32,
        i32_slh: fx.Int32,
        arg_wt: fx.Int64,
        arg_seq: fx.Int64,
        arg_qsl: fx.Int64,
        i32_nseq: fx.Int32,
        arg_out: fx.Int64,
        arg_gate: fx.Int64,
        arg_tab: fx.Int64,
        i32_m: fx.Int32,
        i32_rank: fx.Int32,
        i32_mmax: fx.Int32,
        i64_slot_bytes: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            arg_x, i32_sxt, i32_sxh, arg_lse, i32_slt, i32_slh, arg_wt, arg_seq, arg_qsl, i32_nseq, arg_out,
            arg_gate, arg_tab, i32_m, i32_rank, i32_mmax, i64_slot_bytes,
        ).launch(grid=(base + 1, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
