# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""gfx950 MXFP8 decode GEMM with K split over workgroups and an in-kernel fixed-order reduce:

    out[m, n] = sum_k A[m, k] * W[n, k]      (E4M3 codes, per-32 E8M0 scales, bf16 out)

Inputs are the decode kernel's (``mxfp8_gemm_decode``): A codes in the (16, 16) layout, rows padded to
``n_mb * 16 * RT``, A scales in AITER's shuffled layout (rows padded to 32), the preshuffled W and W scales.

Workgroup = one (m-block, column block, K split): BM = 16 * RT rows, BN = 16 * NIW * NWM columns, K steps
[kz * SWG, (kz + 1) * SWG) of 128.
  - NWM MFMA waves: wave w owns the NIW 16-column tiles [w * NIW, (w + 1) * NIW) x all RT row tiles (accumulators in
    AGPRs); it streams its own W into VGPRs (non-temporal, LW steps ahead), every W scale dword of the split first:
    a scale issued behind later codes would make the in-order vmcnt wait drain the W lookahead.
  - NL loader waves stream the workgroup's A slice into an LDS ring of R step slots (``buffer_load ... lds``, 1 KB
    code blocks and 256 B scale blocks, exact vmcnt waits); the MFMA waves read A with ds_read_b128.
  - One ``s_barrier`` per K step: loaders pass it once step s has landed, MFMA waves once their reads of step s - 1
    are issued; after it the loaders refill step s - 1's slot with step s + R - 1.
An m-block starting on an odd 16-row tile re-pairs its A scale bytes (the decode kernel's repair; RT = 1 shifts).

Hand-off (README rule 7: no atomic accumulation): wave w of split kz cuts its RT * NIW tiles into KS groups, stores
every group but group kz as fp32 partials (1 KB per 16x16 tile, the MFMA lane layout) into the slab, drains them
(vmcnt(0)) and arrives on the (tile, wave) counter. Once the epoch is complete it sums group kz over the splits in
split order ((P0 + P1) + P2) + ... (its own partial from AGPRs), rounds once to bf16 and stores. Every output is
the same KS sequential MFMA chains over equal K ranges summed in split order, whichever split reduces it: the decode
kernel's bytes where its K-waves x splits partition K the same way (KS 3 = the kw3 rows).

Epoch counters: each call adds ``EPOCH`` to every counter it uses (arrival weights EPOCH // KS, plus one for the
first EPOCH % KS splits), so they stay a multiple of EPOCH between calls and are never reset; calls sharing a
counter set must not overlap.

Placement: workgroup b = (g * KS + kz) * 8 + x with g = (nb // 8) * n_mb + mb, x = nb % 8, so the KS splits of a
tile (and the m-blocks of a column block) share b % 8. The partials use the default cache policy and are read from
the writer's L2: this needs the workgroups of one b % 8 on one XCD, which the gfx950 round-robin dispatch gives
(tests/gemm/mxfp8_common/test_decode_splitk.py checks it next to a concurrent kernel). Coherent (sc1) partials
cost 0.1-0.6 us per call. The wait needs the KS splits of a tile co-resident: grid <= 256 workgroups.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .mxfp8_gemm_4wave import _asm_void, _uniform_i32
from .mxfp8_gemm_decode import PRELOAD_HINTS, W_CACHE_MOD, _addr, _mfma, _pack8, _pin_accumulators, _raw

LDS_MAX = 160 * 1024
CNT_STRIDE = 32  # dwords: one counter per 128-byte line
EPOCH = 8  # what one call adds to each counter it uses: a power of 2 >= KS
GRID_MAX = 256  # every workgroup resident at once (the epoch wait)
VMCNT_MAX = 63
# 8 waves put two on each SIMD: 256 VGPR + AGPR per wave. Past the budget the allocator parks MFMA operands in AGPRs
# and the copy back races the inline-asm MFMA (wrong sums; ``mxfp8_gemm_decode.VGPR_BUDGET``).
REG_BUDGET = 256


def geometry(N, M, RT, NIW, NWM, KS):
    """(grid, m-blocks, counter dwords, slab bytes) of one call."""
    nb = N // (16 * NIW * NWM)
    n_mb = -(-M // (16 * RT))
    grid = -(-nb // 8) * 8 * n_mb * KS
    return grid, n_mb, (grid // KS) * NWM * CNT_STRIDE, grid * NWM * RT * NIW * 1024


@launcher
def compile_mxfp8_gemm_decode_splitk(*, N, K, RT, NIW, NWM, NL, KS, LW, R=0):
    """Launcher for one (N, K) and tiling: ``R`` ring slots (0: as many as the LDS holds)."""
    KSTEPS = K // 128
    assert K % 256 == 0 and 2 <= KS <= EPOCH and KSTEPS % KS == 0 and (KSTEPS // KS) % 2 == 0, (K, KS)
    SWG = KSTEPS // KS
    BN = 16 * NIW * NWM
    assert N % BN == 0, (N, BN)
    NB = N // BN
    BM = 16 * RT
    SCB = K // 256
    RP = (RT + 1) // 2  # 32-row scale groups
    CP = NIW if NIW % 2 else NIW // 2
    NT = RT * NIW  # 16x16 tiles per MFMA wave
    SLOT = RT * 2048 + RP * 256
    R = min(R or (LDS_MAX // SLOT), SWG)
    LW = min(LW, SWG)
    NWAVE = NWM + NL
    assert 2 <= R and R * SLOT <= LDS_MAX and NWAVE <= 8, (R, SLOT, NWAVE)
    assert NWAVE <= 4 or NT * 4 + 8 * NIW * (LW + 1) + 40 <= REG_BUDGET, ("registers", NT, NIW, LW)

    def step_blocks(s):
        bl = [("a", rt, h) for rt in range(RT) for h in range(2)]
        if s % 2 == 0:
            bl += [("s", rp, 0) for rp in range(RP)]
        return bl

    def loader_blocks(l, s):
        return [b for j, b in enumerate(step_blocks(s)) if j % NL == l]

    for l in range(NL):
        assert sum(len(loader_blocks(l, s)) for s in range(R - 1)) <= VMCNT_MAX, ("vmcnt", l)
    groups = [list(range(g * NT // KS, (g + 1) * NT // KS)) for g in range(KS)]

    @fx.struct
    class Shared:
        ring: fx.Array[fx.Uint8, R * SLOT, 16]

    name = f"mxfp8_gemm_decode_splitk_n{N}_k{K}_rt{RT}_niw{NIW}_nwm{NWM}_nl{NL}_ks{KS}_lw{LW}_r{R}"

    @flyc.kernel(name=name, known_block_size=[64 * NWAVE, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        n_mb: fx.Int32,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_as: fx.Int64,
        arg_a: fx.Int64,
        arg_out: fx.Int64,
        arg_slab: fx.Int64,
        arg_cnt: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        x = bx % 8
        kz = (bx // 8) % KS
        g = bx // (8 * KS)
        tile = g * 8 + x
        mb = g % n_mb
        nb = (g // n_mb) * 8 + x
        lds_base = fx.Int32(fx.ptrtoint(smem.ring.ptr))
        kstep0 = kz * SWG

        if nb < NB:
            if wave >= NWM:
                a_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_a, num_records_bytes=n_mb * (BM * K))
                as_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_as, num_records_bytes=a_scale_bytes)
                a_voff = [(mb * RT + rt) * (16 * K) + lane * 16 for rt in range_constexpr(RT)]
                s_voff = [(((mb * RT) // 2 + rp) * SCB * 64 + lane) * 4 for rp in range_constexpr(RP)]
                ring_s = _uniform_i32(lds_base)

                def issue(s, blk):
                    kind, i, h = blk
                    slot = s % R
                    if const_expr(kind == "a"):
                        m0 = ring_s + slot * SLOT + i * 2048 + h * 1024
                        soff = _uniform_i32((kstep0 + s) * 2048 + h * 1024)
                        _asm_void([fx.as_ir_value(fx.Int32(m0)), fx.as_ir_value(fx.Int32(a_voff[i])),
                                   fx.as_ir_value(a_rsrc), fx.as_ir_value(fx.Int32(soff))],
                                  "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds", "s,v,s,s", "~{m0}")
                    else:
                        m0 = ring_s + slot * SLOT + RT * 2048 + i * 256
                        soff = _uniform_i32(((kstep0 + s) // 2) * 256)
                        _asm_void([fx.as_ir_value(fx.Int32(m0)), fx.as_ir_value(fx.Int32(s_voff[i])),
                                   fx.as_ir_value(as_rsrc), fx.as_ir_value(fx.Int32(soff))],
                                  "s_mov_b32 m0, $0\nbuffer_load_dword $1, $2, $3 offen lds", "s,v,s,s", "~{m0}")

                for l in range_constexpr(NL):
                    if wave == NWM + l:
                        for s in range_constexpr(R - 1):
                            for blk in loader_blocks(l, s):
                                issue(s, blk)
                        for s in range_constexpr(SWG):
                            younger = sum(len(loader_blocks(l, t)) for t in range(s + 1, min(s + R - 1, SWG)))
                            _asm_void([], f"s_waitcnt vmcnt({younger})", "")
                            _asm_void([], "s_barrier", "")
                            if const_expr(s + R - 1 < SWG):
                                for blk in loader_blocks(l, s + R - 1):
                                    issue(s + R - 1, blk)
            else:
                w_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K)
                ws_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_ws, num_records_bytes=(N // 32) * SCB * 256)
                n16 = [nb * (NIW * NWM) + wave * NIW + ni for ni in range_constexpr(NIW)]
                w_dw = [(n16[ni] * (16 * K) + kstep0 * 2048 + q16 * 256 + l16 * 16) // 4 for ni in range_constexpr(NIW)]
                sc_k = (kstep0 // 2) * 64
                ws_dw = [((n16[cp if NIW % 2 else 2 * cp] // 2) * SCB) * 64 + sc_k + lane for cp in range_constexpr(CP)]
                w_shift = [(n16[ni] % 2) * 8 for ni in range_constexpr(NIW)] if NIW % 2 else None

                def load_pair(dw, s):
                    kw = dict(vec_width=4, dtype=fx.Int32, cache_modifier=W_CACHE_MOD)
                    return _pack8(buffer_ops.buffer_load(w_rsrc, dw, soffset_bytes=s * 2048, **kw),
                                  buffer_ops.buffer_load(w_rsrc, dw, soffset_bytes=s * 2048 + 1024, **kw))

                def load_w_scales(p):
                    sc = [fx.Int32(buffer_ops.buffer_load(ws_rsrc, d, vec_width=1, dtype=fx.Int32,
                                                          soffset_bytes=p * 256)) for d in ws_dw]
                    if const_expr(w_shift is not None):
                        sc = [v.shrui(sh) for v, sh in zip(sc, w_shift)]
                    return sc

                ring_v = lds_base + lane * 16
                sring_v = lds_base + lane * 4

                def lds_ptr(addr):
                    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int32(addr)))

                def read_a(s, rt):
                    base = ring_v + (s % R) * SLOT + rt * 2048
                    return _pack8(Vec(llvm.load(T.vec(4, T.i32), lds_ptr(base), alignment=16)),
                                  Vec(llvm.load(T.vec(4, T.i32), lds_ptr(base + 1024), alignment=16)))

                def read_a_scale(s, rp):
                    return fx.Int32(llvm.load(T.i32, lds_ptr(sring_v + (s % R) * SLOT + RT * 2048 + rp * 256),
                                              alignment=4))

                acc = [[None] * NIW for _ in range(RT)]
                wq = [None] * SWG
                wsc = [load_w_scales(p) for p in range_constexpr(SWG // 2)]
                fx.rocdl.sched_barrier(0)
                for s in range_constexpr(LW):
                    wq[s] = [load_pair(w_dw[ni], s) for ni in range_constexpr(NIW)]
                fx.rocdl.sched_barrier(0)
                sw = sa = None
                for s in range_constexpr(SWG):
                    if const_expr(s + LW < SWG):
                        wq[s + LW] = [load_pair(w_dw[ni], s + LW) for ni in range_constexpr(NIW)]
                    fx.rocdl.sched_barrier(0)
                    _asm_void([], "s_barrier", "")
                    fx.rocdl.sched_barrier(0)
                    bb = wq[s]
                    if const_expr(s % 2 == 0):
                        sw = wsc[s // 2]
                        sa = [read_a_scale(s, rp) for rp in range_constexpr(RP)]
                        if const_expr(RT == 1):
                            sa = [sa[0].shrui(((mb % 2) * 8))]
                        if const_expr(RT > 1 and RT % 2 == 1):
                            # an m-block starting on an odd 16-row tile: re-pair the scale bytes to the even layout
                            # the MFMA immediates assume (the decode kernel's repair)
                            odd = (mb * RT) % 2 == 1
                            raw = sa
                            sa = [odd.select((raw[p].shrui(fx.Int32(8)) & fx.Int32(0x00FF00FF))
                                             | ((raw[min(p + 1, RP - 1)] & fx.Int32(0x00FF00FF)) << 8), raw[p])
                                  for p in range_constexpr(RP)]
                    aq = [None] * RT
                    for rt in range_constexpr(min(2, RT)):
                        aq[rt] = read_a(s, rt)
                    for rt in range_constexpr(RT):
                        if const_expr(rt + 2 < RT):
                            aq[rt + 2] = read_a(s, rt + 2)
                        for ni in range_constexpr(NIW):
                            acc[rt][ni] = _mfma(bb[ni], aq[rt], acc[rt][ni], sw[ni if NIW % 2 else ni // 2],
                                                sa[rt // 2], 0 if NIW % 2 else ni % 2, rt % 2, s % 2)
                        aq[rt] = None
                        fx.rocdl.sched_barrier(0)
                    wq[s] = None

                flat = _pin_accumulators([acc[rt][ni] for rt in range(RT) for ni in range(NIW)])
                slab_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_slab, num_records_bytes=0x7FFFFFFF)
                out_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i32_m * (N * 2))

                def slab_off(k):
                    return ((tile * KS + k) * NWM + wave) * (NT * 1024) + lane * 16

                def store_out(i, v):
                    row = mb * BM + (i // NIW) * 16 + l16
                    col = n16[i % NIW] * 16 + q16 * 4
                    buffer_ops.buffer_store(v.to(fx.BFloat16).bitcast(fx.Int32), out_rsrc, (row * N + col) * 2,
                                            offset_is_bytes=True, mask=row < i32_m)

                cnt_addr = arg_cnt + fx.Int64((tile * NWM + wave) * (CNT_STRIDE * 4))
                inc = (kz < EPOCH % KS).select(fx.Int32(EPOCH // KS + 1), fx.Int32(EPOCH // KS))

                def atomic_add(value):
                    ptr = fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
                                      cnt_addr)
                    return fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, ptr.llvm_ptr, fx.Int32(value).ir_value(),
                                                     llvm.AtomicOrdering.monotonic, syncscope="agent",
                                                     alignment=4).result)

                for r in range_constexpr(KS):
                    if kz == r:
                        mine = groups[r]
                        for i in [i for gi in range(KS) if gi != r for i in groups[gi]]:
                            buffer_ops.buffer_store(Vec(flat[i]).bitcast(fx.Int32), slab_rsrc, slab_off(r) + i * 1024,
                                                    offset_is_bytes=True)
                        fx.rocdl.s_waitcnt(vmcnt=0)
                        if lane == 0:
                            old = atomic_add(inc)
                            base = old - (old & (EPOCH - 1))  # i32 wrap included
                            arrived = old + inc - base
                            while arrived < EPOCH:
                                arrived = atomic_add(0) - base
                        parts = [[None if k == r else Vec(buffer_ops.buffer_load(
                            slab_rsrc, (slab_off(k) + i * 1024) // 4, vec_width=4, dtype=fx.Int32)).bitcast(fx.Float32)
                            for k in range_constexpr(KS)] for i in mine]
                        for j, i in enumerate(mine):
                            v = None
                            for k in range_constexpr(KS):
                                p = Vec(flat[i]) if k == r else parts[j][k]
                                v = p if v is None else v + p
                            store_out(i, v)

    @flyc.jit
    def launch(A: fx.Tensor, W: fx.Tensor, OUT: fx.Tensor, A_scale: fx.Tensor, W_scale: fx.Tensor,
               SLAB: fx.Tensor, CNT: fx.Tensor, i32_m: fx.Int32, a_scale_bytes: fx.Int32, n_mb: fx.Int32,
               grid: fx.Int32, stream: fx.Stream):
        kernel(i32_m, a_scale_bytes, n_mb, _addr(W), _addr(W_scale), _addr(A_scale), _addr(A), _addr(OUT),
               _addr(SLAB), _addr(CNT)).launch(grid=(fx.Int64(grid), 1, 1), block=(64 * NWAVE, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
