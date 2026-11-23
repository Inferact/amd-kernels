# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 [M, 5120] x bf16 [NA + NB, 5120]^T for M up to 1024: the first NA output columns fp32 (the compressor's
kv_score), the last NB columns bf16 (the indexer's weights_proj), FlyDSL (gfx950).

A workgroup of NW waves takes NT 16-column tiles x MT 16-row tiles of the output. Wave w sums the K part
[5120 w / NW, 5120 (w + 1) / NW) of every output with v_mfma_f32_16x16x32_bf16 (K ascending, the weight as the A operand), and
the NW fp32 partial sums are added in the order (((w0 + w1) + w2) + ...) through LDS, so an output is one fixed fp32
sum. Weight fragments come from HBM / L2 straight into registers, 1 KB contiguous per wave load in the load-time
layout [column tile, K step, lane, 8] (host.shuffle_weight); activation fragments are 16 rows x 64 bytes.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

K = 5120
AGENT_SCOPE = 0x10
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def _g_ptr(addr):
    return _llvm.inttoptr(_ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr)))


def _barrier():
    _llvm.inline_asm(None, [], "s_waitcnt lgkmcnt(0)\ns_barrier", "~{memory}", has_side_effects=True)


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def compile_gemm(na: int, nb: int, nt: int, mt: int, ch: int, nw: int, ksp: int = 1):
    """``na``: fp32 columns, ``nb``: bf16 columns (multiples of 16); ``nt`` / ``mt``: 16-column / 16-row tiles of a
    workgroup; ``ch``: K steps whose fragments are in flight (a chunk loads while the previous one multiplies); ``nw``:
    waves (K parts) of a workgroup; ``ksp``: workgroups a tile's K is split over (a wave sums 160 / (nw ksp) K steps; the
    workgroups' sums go through a partials buffer and a ticket, and the last workgroup of a tile adds them in K order)."""
    NW = nw
    S = ksp
    KSTEPS = K // 32 // (NW * S)  # K steps of a wave
    assert KSTEPS * NW * S * 32 == K
    NTILES = (na + nb) // 16
    NA16 = na // 16
    name = f"bf16_gemm_k{K}_a{na}_b{nb}_nt{nt}_mt{mt}_c{ch}_w{NW}" + (f"_s{S}" if S > 1 else "")
    LDS_B = NW * nt * mt * 64 * 16

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS_B, 16]
        tk: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        arg_x: fx.Int64,  # [m, 5120] bf16
        arg_w: fx.Int64,  # weight, fragment order
        arg_ya: fx.Int64,  # [m, na] fp32
        arg_yb: fx.Int64,  # [m, nb] bf16
        arg_part: fx.Int64,  # ksp > 1: partial sums [tile group, K split, nt mt, lane, 4] fp32
        arg_cnt: fx.Int64,  # ksp > 1: [tile group] int32 tickets, zero
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        g = lane // 16
        l16 = lane % 16
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lds = fx.Int32(fx.ptrtoint(smem.buf.ptr))
        tk = fx.Int32(fx.ptrtoint(smem.tk.ptr))
        n_groups = (NTILES + nt - 1) // nt
        m_groups = (i32_m + (16 * mt - 1)) // (16 * mt)
        tot = n_groups * m_groups
        # workgroups go round-robin over the 8 XCDs (one L2 each): an XCD takes a contiguous run of the flat order
        # [column group][row group][K split], so the weights of its column groups and the activations are read once
        # per XCD from the MALL instead of once per workgroup
        nwg = tot * S
        qn, rn = nwg // 8, nwg % 8
        xcd, slot = bid % 8, bid // 8
        flat = (xcd < rn).select(xcd * (qn + 1) + slot, rn * (qn + 1) + (xcd - rn) * qn + slot)
        ksi = flat % S  # K split of this workgroup
        mg = (flat // S) % m_groups
        ng = flat // (S * m_groups)
        gb = ng * m_groups + mg  # tile group
        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=fx.Int64(i32_m) * (K * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=fx.Int64(NTILES * 16) * (K * 2))
        rows = []
        for m in range_constexpr(mt):
            r = mg * (16 * mt) + m * 16 + l16
            rows.append((r < i32_m).select(r, i32_m - 1))
        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(mt)] for _ in range_constexpr(nt)]
        ntile = [(ng * nt + j < NTILES).select(ng * nt + j, fx.Int32(NTILES - 1)) for j in range(nt)]
        def frags(ks):
            k = (ksi * NW + wave) * KSTEPS + ks
            wf = [Vec(buffer_ops.buffer_load(wr, ((ntile[j] * (K // 32) + k) * 64 + lane) * 4, vec_width=4,
                                             dtype=fx.Int32)).bitcast(fx.BFloat16) for j in range_constexpr(nt)]
            xf = [Vec(buffer_ops.buffer_load(xr, rows[m] * (K // 2) + k * 16 + g * 4, vec_width=4,
                                             dtype=fx.Int32)).bitcast(fx.BFloat16) for m in range_constexpr(mt)]
            return wf, xf

        chunks = [list(range(c, min(c + ch, KSTEPS))) for c in range(0, KSTEPS, ch)]
        nxt = [frags(ks) for ks in chunks[0]]
        for ci in range_constexpr(len(chunks)):
            cur = nxt
            if ci + 1 < len(chunks):
                nxt = [frags(ks) for ks in chunks[ci + 1]]
            for wf, xf in cur:
                for j in range_constexpr(nt):
                    for m in range_constexpr(mt):
                        acc[j][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                            Vec.make_type(4, fx.Float32), [wf[j], xf[m], acc[j][m], 0, 0, 0]))
        for j in range_constexpr(nt):
            for m in range_constexpr(mt):
                _llvm.store(
                    _raw(Vec.from_elements([fx.Float32(acc[j][m][i]) for i in range(4)], fx.Float32)),
                    _llvm.inttoptr(_llvm_ptr3(), _raw(fx.Int64(lds + (((wave * nt + j) * mt + m) * 64 + lane) * 16))),
                    alignment=16,
                )
        fx.gpu.barrier()
        def lds_sum(j, m):
            parts = [Vec(_llvm.load(T.vec(4, T.f32), _llvm.inttoptr(
                _llvm_ptr3(), _raw(fx.Int64(lds + (((w * nt + j) * mt + m) * 64 + lane) * 16))), alignment=16))
                for w in range(NW)]
            s = [parts[0][i] for i in range(4)]
            for w in range_constexpr(1, NW):
                s = [s[i] + parts[w][i] for i in range(4)]
            return s

        def emit(s, j, m):
            row = mg * (16 * mt) + m * 16 + l16
            nti = ng * nt + j
            ok = (row < i32_m) & (nti < NTILES)
            col = nti * 16 + g * 4
            if nti < NA16:
                ya = buffer_ops.create_buffer_resource_from_addr(
                    arg_ya, num_records_bytes=fx.Int64(i32_m) * (na * 4))
                buffer_ops.buffer_store(Vec.from_elements(s, fx.Float32), ya, row * na + col, mask=ok)
            else:
                yb = buffer_ops.create_buffer_resource_from_addr(
                    arg_yb, num_records_bytes=fx.Int64(i32_m) * (nb * 2))
                pk = [fx.Float32(s[i]).to(fx.BFloat16) for i in range(4)]
                packed = Vec.from_elements(pk, fx.BFloat16).bitcast(fx.Int32)
                buffer_ops.buffer_store(packed, yb, row * (nb // 2) + (col - na) // 2, mask=ok)

        fx.gpu.barrier()
        if S == 1:
            # wave w finishes the tiles idx = w, w + NW, ...
            for q in range_constexpr((nt * mt + NW - 1) // NW):
                idx = wave + NW * q
                if idx < nt * mt:
                    emit(lds_sum(idx // mt, idx % mt), idx // mt, idx % mt)
        else:
            pr = buffer_ops.create_buffer_resource_from_addr(arg_part, num_records_bytes=fx.Int64(0x7FFFFFF0))

            def poff(sp, idx):  # in dwords
                return (((gb * S + sp) * (nt * mt) + idx) * 64 + lane) * 4

            for q in range_constexpr((nt * mt + NW - 1) // NW):
                idx = wave + NW * q
                if idx < nt * mt:
                    s = lds_sum(idx // mt, idx % mt)
                    buffer_ops.buffer_store(Vec.from_elements([fx.Float32(v).bitcast(fx.Int32) for v in s], fx.Int32),
                                            pr, poff(ksi, idx), cache_modifier=AGENT_SCOPE)
            _llvm.inline_asm(None, [], "s_waitcnt vmcnt(0)", "~{memory}", has_side_effects=True)
            _barrier()
            if tx == 0:
                old = fx.Int32(_llvm.AtomicRMWOp(
                    _llvm.AtomicBinOp.add, _g_ptr(arg_cnt + fx.Int64(gb) * 4), fx.Int32(1).ir_value(),
                    _llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4).result)
                _llvm.store(_raw(old), _llvm.inttoptr(_llvm_ptr3(), _raw(fx.Int64(tk))), alignment=4)
            _barrier()
            last = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(
                _llvm.load(T.i32, _llvm.inttoptr(_llvm_ptr3(), _raw(fx.Int64(tk))), alignment=4))))
            if last == S - 1:
                for q in range_constexpr((nt * mt + NW - 1) // NW):
                    idx = wave + NW * q
                    if idx < nt * mt:
                        ps = [Vec(buffer_ops.buffer_load(pr, poff(sp, idx), vec_width=4, dtype=fx.Int32,
                                                         cache_modifier=AGENT_SCOPE)).bitcast(fx.Float32)
                              for sp in range_constexpr(S)]
                        sm = [ps[0][i] for i in range(4)]
                        for sp in range_constexpr(1, S):
                            sm = [sm[i] + ps[sp][i] for i in range(4)]
                        emit(sm, idx // mt, idx % mt)
                if tx == 0:
                    _llvm.store(_raw(fx.Int32(0)), _g_ptr(arg_cnt + fx.Int64(gb) * 4), alignment=4)

    def _llvm_ptr3():
        from flydsl._mlir import ir as _ir

        return _ir.Type.parse("!llvm.ptr<3>")

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_ya: fx.Int64, arg_yb: fx.Int64, arg_part: fx.Int64,
               arg_cnt: fx.Int64, i32_m: fx.Int32, i32_groups: fx.Int32, stream: fx.Stream):
        kernel(i32_m, arg_x, arg_w, arg_ya, arg_yb, arg_part, arg_cnt).launch(
            grid=(i32_groups, 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
