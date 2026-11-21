# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Split-K bf16 GEMM of the drafter's dense MLP (gfx950): ``gate_up`` with the SwiGLU epilogue and ``down``.

Operands are in the load-time MFMA layout (``pack.py``), so every wave load of weights or activations is one coalesced
1 KB. The weights are the MFMA's first operand, the activation rows the second. A workgroup owns ``mt`` row tiles of
16, ``g`` weight groups of 16 rows and 1 / ``split`` of K; its ``wr * wk`` waves split the row tiles and K, so every
activation and weight byte is loaded once per workgroup (the weights bypass the L2 so that the activation tiles stay
resident). The K loop is fully unrolled with ``depth`` steps of loads in flight, which a loop-carried software
pipeline cannot keep (the compiler drains its loads at the back edge). The waves of a K group are summed in wave order
through LDS.

``split`` > 1: the ``split`` workgroups of a tile run on one XCD (workgroup id mod 8), store their fp32 partial tiles
to the workspace, take a ticket, and the last one to arrive sums the partials in split order (its own from
registers) and runs the epilogue; the ticket counter resets itself. No output atomics, deterministic bits.

``mode`` 1: bf16 row-major output [rows, 16 * groups]. ``mode`` 2 (``g`` = 2 ``gp``): the first ``gp`` groups of a
block are gate groups and the last ``gp`` the up groups of the same columns; the epilogue is bf16(silu(gate) * up)
in fp32, stored in the packed activation layout of the next GEMM [tiles, 16 * groups / 2 / 32, 64, 8].
"""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

LDS_MAX = 160 * 1024
NT = 0x2
CNT_STRIDE = 32  # ints between ticket counters
LOG2E = 1.4426950408889634


def _lds_ptr(addr):
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int64(addr)))


def _atomic_add(addr, v):
    """Returning scalar global add by one lane (the ticket)."""
    low = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int64(addr).to(fx.Int32)))
    high = fx.Int32(fx.rocdl.readfirstlane(T.i32, (fx.Int64(addr) >> 32).to(fx.Int32)))
    pointer = (low.to(fx.Uint32).to(fx.Uint64) | (high.to(fx.Uint32).to(fx.Uint64) << 32)).bitcast(fx.Int64)
    inc = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))
    return fx.Int32(_llvm.inline_asm(
        T.i32, [fx.as_ir_value(inc), fx.as_ir_value(pointer)], "s_atomic_add $0, $2, 0 glc\n\ts_waitcnt lgkmcnt(0)",
        "=s,0,s,~{memory}", has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT))


def _pk(a, b):
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def _silu_mul(gate, up):
    e = fx.Float32(_rocdl.exp2(T.f32, fx.as_ir_value(gate * fx.Float32(-LOG2E))))
    return gate * fx.Float32(_rocdl.rcp(T.f32, fx.as_ir_value(fx.Float32(1.0) + e))) * up


@functools.cache
@launcher
def compile_gemm(k: int, groups: int, mt: int, g: int, split: int, depth: int, wr: int, wk: int, mode: int):
    ks = k // 32
    assert groups % g == 0 and mt % wr == 0 and mode in (1, 2)
    gp = g // 2
    assert mode == 1 or g % 2 == 0
    nblocks = groups // g
    nw = wr * wk
    mtw = mt // wr
    steps = -(-ks // (split * wk))
    tw = mtw * g
    lds = max(nw * tw * 1024, 4096)
    assert lds <= LDS_MAX, lds
    ntot = groups * 16
    nv = 2 if mode == 2 else 1
    ipt = mtw * (gp if mode == 2 else g)
    items = wr * ipt * 64
    kso = ntot // 2 // 32
    name = f"bf16_mlp_gemm_k{k}_n{ntot}_mt{mt}_g{g}_sk{split}_d{depth}_w{wr}x{wk}_m{mode}"

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, lds, 16]
        flag: fx.Array[fx.Int32, 4]

    def mfma(a, b, c):
        return Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
            Vec.make_type(4, fx.Float32), [a.bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), c, 0, 0, 0]))

    @flyc.kernel(name=name, known_block_size=[64 * nw, 1, 1])
    def kernel(x: fx.Int64, weight: fx.Int64, out: fx.Int64, pslab: fx.Int64, cnt: fx.Int64, rows: fx.Int32,
               mblocks: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wi = wave % wr
        wj = wave // wr
        sp = (bid // 8) % split
        tile = (bid // (8 * split)) * 8 + bid % 8
        valid = tile < mblocks * nblocks
        tile = valid.select(tile, fx.Int32(0))
        mb = tile // nblocks
        nb = tile % nblocks
        wres = buffer_ops.create_buffer_resource_from_addr(weight, num_records_bytes=groups * 16 * k * 2)
        xres = buffer_ops.create_buffer_resource_from_addr(x, num_records_bytes=fx.Int64(0x7FFFFFF0))
        k0 = (sp * wk + wj) * steps

        def issue(step):
            gs = k0 + step
            ok = (gs < ks) & valid
            av = [Vec(buffer_ops.buffer_load(wres, ((nb * g + j) * ks + gs) * 256 + lane * 4, vec_width=4,
                                             dtype=fx.Int32, mask=ok, cache_modifier=NT)) for j in range_constexpr(g)]
            bv = [Vec(buffer_ops.buffer_load(xres, (((mb * mt + wi * mtw + i) * ks + gs) * 64 + lane) * 4,
                                             vec_width=4, dtype=fx.Int32, mask=ok)) for i in range_constexpr(mtw)]
            return av, bv

        acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(tw)]
        pend = {}
        for st in range_constexpr(min(depth, steps)):
            pend[st] = issue(st)
        for st in range_constexpr(steps):
            av, bv = pend.pop(st)
            if st + depth < steps:
                pend[st + depth] = issue(st + depth)
            fx.rocdl.sched_barrier(0)
            for i in range_constexpr(mtw):
                for j in range_constexpr(g):
                    acc[i * g + j] = mfma(av[j], bv[i], acc[i * g + j])
            fx.rocdl.sched_barrier(0)
        for t in range_constexpr(tw):
            _llvm.store(fx.as_ir_value(acc[t]), _lds_ptr(lds_base + ((wave * tw + t) * 64 + lane) * 16), alignment=16)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()

        nth = 64 * nw
        mine = []
        for u in range_constexpr((items + nth - 1) // nth):
            it = tx + u * nth
            ok = it < items
            itc = ok.select(it, fx.Int32(0))
            r = itc // (ipt * 64)
            q = (itc // 64) % ipt
            ln = itc % 64
            if const_expr(mode == 2):
                i, jj = q // gp, q % gp
                tiles_ = (i * g + jj, i * g + gp + jj)
            else:
                i, jj = q // g, q % g
                tiles_ = (q,)
            vals = []
            for tt in tiles_:
                tot = Vec.filled(4, 0.0, fx.Float32)
                for w in range_constexpr(wk):
                    tot = tot + Vec(_llvm.load(T.vec(4, T.f32), _lds_ptr(
                        lds_base + (((w * wr + r) * tw + tt) * 64 + ln) * 16), alignment=16))
                vals.append(tot)
            mine.append((vals, r, i, jj, ln, ok & valid))

        def finish(vals, r, i, jj, ln, ok):
            row = (mb * mt + r * mtw + i) * 16 + ln % 16
            if const_expr(mode == 1):
                col = (nb * g + jj) * 16 + (ln // 16) * 4
                ores = buffer_ops.create_buffer_resource_from_addr(out, num_records_bytes=fx.Int64(rows) * (ntot * 2))
                pk = Vec.from_elements([_pk(vals[0][0], vals[0][1]), _pk(vals[0][2], vals[0][3])], fx.Int32)
                buffer_ops.buffer_store(pk, ores, (row * ntot + col) // 2, mask=ok & (row < rows))
            else:
                res = [_silu_mul(fx.Float32(vals[0][e]), fx.Float32(vals[1][e])) for e in range_constexpr(4)]
                pk = Vec.from_elements([_pk(res[0], res[1]), _pk(res[2], res[3])], fx.Int32)
                oc = (nb * gp + jj) * 16 + (ln // 16) * 4
                tout = mb * mt + r * mtw + i
                dw = ((((tout * kso + oc // 32) * 64 + ((oc % 32) // 8) * 16 + ln % 16) * 16) + ((oc // 4) % 2) * 8) // 4
                ores = buffer_ops.create_buffer_resource_from_addr(out, num_records_bytes=fx.Int64(0x7FFFFFF0))
                buffer_ops.buffer_store(pk, ores, dw, mask=ok)

        if const_expr(split == 1):
            for vals, r, i, jj, ln, ok in mine:
                finish(vals, r, i, jj, ln, ok)
        else:
            sres = buffer_ops.create_buffer_resource_from_addr(pslab, num_records_bytes=fx.Int64(0x7FFFFFF0))

            def slot(qs, it):
                return (tile * split + qs) * items + it

            for u, (vals, r, i, jj, ln, ok) in enumerate(mine):
                for v in range_constexpr(nv):
                    buffer_ops.buffer_store(vals[v], sres, slot(sp, tx + u * nth) * (nv * 4) + v * 4, mask=ok)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx == 0:
                caddr = cnt + fx.Int64(tile) * (CNT_STRIDE * 4)
                old = _atomic_add(caddr, valid.select(fx.Int32(1), fx.Int32(0)))
                last = (old == (split - 1)) & valid
                if last:
                    _atomic_add(caddr, -split)
                smem.flag.ptr[0] = last.select(fx.Int32(1), fx.Int32(0))
            fx.gpu.barrier()
            if fx.Int32(smem.flag.ptr[0]) == 1:
                for u, (vals, r, i, jj, ln, ok) in enumerate(mine):
                    parts = [[Vec(buffer_ops.buffer_load(sres, slot(qs, tx + u * nth) * (nv * 4) + v * 4, vec_width=4,
                                                         dtype=fx.Float32, mask=ok & (sp != qs)))
                              for v in range_constexpr(nv)] for qs in range_constexpr(split)]
                    sums = []
                    for v in range_constexpr(nv):
                        tot = None
                        for qs in range_constexpr(split):
                            val = Vec.from_elements([(sp == qs).select(fx.Float32(vals[v][e]), fx.Float32(parts[qs][v][e]))
                                                     for e in range_constexpr(4)], fx.Float32)
                            tot = val if tot is None else tot + val
                        sums.append(tot)
                    finish(sums, r, i, jj, ln, ok)

    @flyc.jit
    def launch(x: fx.Int64, weight: fx.Int64, out: fx.Int64, pslab: fx.Int64, cnt: fx.Int64, rows: fx.Int32,
               mblocks: fx.Int32, ncta: fx.Int32, stream: fx.Stream):
        kernel(x, weight, out, pslab, cnt, rows, mblocks).launch(grid=(fx.Int64(ncta), 1, 1), block=(64 * nw, 1, 1),
                                                                 stream=stream)

    launch.kernel_name = name
    return launch
