# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Weight-streaming bf16 GEMM y [M, N] = x [M, K] @ W^T for skinny M (decode), W in the load-time packed layout.

Tile: a workgroup owns WN * NG 16-row weight groups (each of its WN n-waves NG of them) and one M tile of 16 MT rows,
and runs K / S of the reduction (S workgroups split K); its WK k-waves split that K range into 128-wide blocks in
turn. Weights stream from HBM straight into MFMA operands (v_mfma_f32_16x16x32_bf16, A = weight group, B = activation
rows): the packed layout makes every 16-byte lane load one 1 KB piece of a group's K stream, DEPTH K steps ahead.
Activations go through LDS, shared by the waves: a ring of RING blocks (128 K x 16 MT rows per k-wave) is filled by
asynchronous 16-byte DMAs (row pieces XOR-swizzled so the fragment reads are bank-conflict free), one barrier per
block; a wave waits on a DMA by counting the vector-memory operations issued after it (``log``), so the weight
prefetch DEPTH stays below 4 (RING - 1) steps. The k-waves' fp32 tiles are summed in wave order through LDS.
With S > 1 each workgroup writes its fp32 tile to the workspace and takes a ticket; the last workgroup of the output
tile sums the S tiles in split order and writes bf16 (no output atomics; the counter resets itself). The S workgroups
of a tile run on one XCD (workgroup b runs on XCD b % 8), so the tiles are exchanged through that XCD's L2 with plain
stores and loads: a workgroup id -> tile map that keeps ``bid % 8`` of a tile's S workgroups equal is what makes the
exchange coherent.
Every output element is one fp32 sum (K ascending within a wave, waves then splits in order), rounded once to bf16.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}
NT = 0x2
CNT_STRIDE = 32
KB = 128
LDS_MAX = 160 * 1024
EXCHANGE_CHUNK = 32


def _raw(v):
    return v.ir_value() if not isinstance(v, ir.Value) and hasattr(v, "ir_value") else v


def _lds_ptr(byte_addr):
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def _atomic_add(addr, v):
    """Returning scalar global RMW by one active lane (the ticket)."""
    low = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int64(addr).to(fx.Int32)))
    high = fx.Int32(fx.rocdl.readfirstlane(T.i32, (fx.Int64(addr) >> 32).to(fx.Int32)))
    pointer = (low.to(fx.Uint32).to(fx.Uint64) | (high.to(fx.Uint32).to(fx.Uint64) << 32)).bitcast(fx.Int64)
    inc = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))
    return fx.Int32(_llvm.inline_asm(
        T.i32, [fx.as_ir_value(inc), fx.as_ir_value(pointer)], "s_atomic_add $0, $2, 0 glc\n\ts_waitcnt lgkmcnt(0)",
        "=s,0,s,~{memory}", has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT))


def _barrier():
    _llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)


def slab_floats(n: int, mt: int, wn: int, wk: int, ng: int, s: int, mtiles: int) -> int:
    """fp32 words of the split-K workspace of one launch."""
    tiles = -(-(n // 16) // (wn * ng))
    nw = wn * wk
    per_thread = ng * mt if wk == 1 else -(-wn * ng * mt * 64 // (64 * nw))
    return mtiles * tiles * s * per_thread * 64 * nw * 4


@launcher
def compile_bf16_skinny(n: int, k: int, mt: int, nwn: int, nwk: int, ng: int, s: int, depth: int, ring: int = 2):
    assert n % 16 == 0 and k % (KB * s * nwk) == 0, (n, k, s, nwk)
    nw = nwn * nwk
    NB = k // (KB * s)
    NBK = NB // nwk
    NSTEP = NBK * 4
    KS = k // s
    NGT = n // 16
    NTILES = (NGT + nwn * ng - 1) // (nwn * ng)
    PER = ((NTILES + 7) // 8) * 8 * s
    R = max(2, min(ring, NBK + 1))
    XD = 4 * mt
    XDT = nwk * XD
    XPW = (XDT + nw - 1) // nw
    XSLOT = 4096 * mt
    SLOT = nwk * XSLOT
    UN = nwn * ng * mt
    UPT = (UN * 64 + 64 * nw - 1) // (64 * nw)
    LDS = max(R * SLOT, nwk * UN * 1024 if nwk > 1 else 0)
    D = min(depth, NSTEP)
    assert LDS + 16 <= LDS_MAX and 64 * nw <= 1024 and (XPW * (R - 1) + D * ng) <= 63, (LDS, XPW, D, ng)
    name = f"bf16_skinny_n{n}_k{k}_mt{mt}_wn{nwn}_wk{nwk}_ng{ng}_s{s}_d{D}_r{R}"

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]
        flag: fx.Array[fx.Int32, 4]

    @flyc.kernel(name=name, known_block_size=[64 * nw, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, arg_slab: fx.Int64, arg_cnt: fx.Int64,
               i32_m: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wn = wave % nwn
        wk = wave // nwn
        l16, g4 = lane % 16, lane // 16
        mtile = bid // PER
        b8 = bid % PER
        ntile_raw = ((b8 // 8) // s) * 8 + b8 % 8
        sp = (b8 // 8) % s
        valid = ntile_raw < NTILES
        ntile = valid.select(ntile_raw, fx.Int32(NTILES - 1))
        m0 = mtile * (16 * mt)

        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_m * (k * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=n * k * 2)
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))

        gg = [(ntile * nwn + wn) * ng + gi for gi in range_constexpr(ng)]
        w_off = []
        for gi in range_constexpr(ng):
            off = (gg[gi] * (k // 32) + sp * (KS // 32) + wk * 4) * 256 + lane * 4
            w_off.append(((gg[gi] < NGT) & valid).select(off, fx.Int32(0x1FFFFFF0)))

        x_src, x_dst, x_blk = [], [], []
        for q in range_constexpr(XPW):
            j = wave + q * nw
            j = (j < XDT).select(j, fx.Int32(XDT - 1))
            kk = j // XD
            jj = j % XD
            rl = jj * 4 + lane // 16
            row = m0 + rl
            row = (row < i32_m).select(row, i32_m - 1)
            piece = (lane % 16) ^ (rl % 16)
            x_src.append(row * (k * 2) + sp * (KS * 2) + piece * 16)
            x_dst.append(kk * XSLOT + jj * 1024)
            x_blk.append(fx.Int32(fx.rocdl.readfirstlane(T.i32, kk * 256)))

        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
        log = []

        def _sb():
            fx.rocdl.sched_barrier(0)

        def issue_x(r):
            for q in range_constexpr(XPW):
                d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + (r % R) * SLOT + x_dst[q]))
                lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
                _rocdl_ods.raw_ptr_buffer_load_async_lds(
                    xr, lp, fx.Int32(16).ir_value(), fx.Int32(x_src[q]).ir_value(),
                    fx.Int32(x_blk[q] + r * nwk * 256).ir_value(), fx.Int32(0).ir_value(), aux=aux_x)
                log.append(("x", r))
            _sb()

        wkw = dict(vec_width=4, dtype=fx.Int32, cache_modifier=NT)
        wregs = {}

        def issue_w(t):
            r, j = t // 4, t % 4
            wregs[t] = [buffer_ops.buffer_load(wr, w_off[gi], soffset_bytes=(r * nwk * 4 + j) * 1024, **wkw)
                        for gi in range_constexpr(ng)]
            for _ in range_constexpr(ng):
                log.append(("w", t))
            _sb()

        rd_off = [[(mi * 16 + l16) * 256 + ((j * 4 + g4) ^ l16) * 16 for mi in range_constexpr(mt)]
                  for j in range_constexpr(4)]
        rd_base = wk * XSLOT

        def read_x(r, j):
            return [Vec(_llvm.load(T.vec(4, T.i32), _lds_ptr(lds_base + (r % R) * SLOT + rd_base + rd_off[j][mi]),
                                   alignment=16)) for mi in range_constexpr(mt)]

        def wait_x(r):
            last = max(i for i, e in enumerate(log) if e == ("x", r))
            fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=min(63, len(log) - 1 - last))

        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(mt)] for _ in range_constexpr(ng)]

        for r in range_constexpr(min(R - 1, NBK)):
            issue_x(r)
        for t in range_constexpr(D):
            issue_w(t)
        for r in range_constexpr(NBK):
            wait_x(r)
            _barrier()
            _sb()
            if r + R - 1 < NBK:
                issue_x(r + R - 1)
            xv = read_x(r, 0)
            _sb()
            for j in range_constexpr(4):
                t = r * 4 + j
                if t + D < NSTEP:
                    issue_w(t + D)
                xn = read_x(r, j + 1) if j < 3 else None
                _sb()
                wv = wregs.pop(t)
                for gi in range_constexpr(ng):
                    for mi in range_constexpr(mt):
                        acc[gi][mi] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                            Vec.make_type(4, fx.Float32),
                            [Vec(wv[gi]).bitcast(fx.BFloat16), xv[mi].bitcast(fx.BFloat16), acc[gi][mi], 0, 0, 0]))
                _sb()
                xv = xn

        yr = buffer_ops.create_buffer_resource_from_addr(arg_y, num_records_bytes=i32_m * (n * 2))
        tiles = []
        if const_expr(nwk == 1):
            for gi in range_constexpr(ng):
                nn = gg[gi] * 16 + g4 * 4
                for mi in range_constexpr(mt):
                    mm = m0 + mi * 16 + l16
                    tiles.append((acc[gi][mi], nn, mm, valid & (gg[gi] < NGT) & (mm < i32_m)))
        else:
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _barrier()
            for gi in range_constexpr(ng):
                for mi in range_constexpr(mt):
                    p = _lds_ptr(lds_base + ((wk * UN + (wn * ng + gi) * mt + mi) * 64 + lane) * 16)
                    _llvm.store(acc[gi][mi], p, alignment=16)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _barrier()
            for u in range_constexpr(UPT):
                idx = tx + u * (64 * nw)
                ui = idx // 64
                ln = idx % 64
                ok = ui < UN
                uis = ok.select(ui, fx.Int32(0))
                tot = Vec.filled(4, 0.0, fx.Float32)
                for w in range_constexpr(nwk):
                    p = _lds_ptr(lds_base + ((w * UN + uis) * 64 + ln) * 16)
                    tot = tot + Vec(_llvm.load(T.vec(4, T.f32), p, alignment=16))
                g_u = ntile * nwn * ng + uis // mt
                nn = g_u * 16 + (ln // 16) * 4
                mm = m0 + (uis % mt) * 16 + ln % 16
                tiles.append((tot, nn, mm, valid & ok & (g_u < NGT) & (mm < i32_m)))

        def store_bf16(tot, nn, mm, live):
            pk = Vec(tot).to(fx.BFloat16).bitcast(fx.Int32)
            buffer_ops.buffer_store(pk, yr, mm * (n // 2) + nn // 2, mask=live)

        if const_expr(s == 1):
            for tot, nn, mm, live in tiles:
                store_bf16(tot, nn, mm, live)
        else:
            t_per = len(tiles)
            sr = buffer_ops.create_buffer_resource_from_addr(arg_slab, num_records_bytes=0x7FFFFFF0)
            wg_base = ((mtile * NTILES + ntile) * s) * (t_per * 64 * nw)

            def slot(q, t):
                return (wg_base + (q * t_per + t) * 64 * nw + tx) * 4

            for t, (tot, nn, mm, live) in enumerate(tiles):
                buffer_ops.buffer_store(tot, sr, slot(sp, t), mask=valid)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx == 0:
                cnt = arg_cnt + fx.Int64(mtile * NTILES + ntile) * (CNT_STRIDE * 4)
                old = _atomic_add(cnt, valid.select(fx.Int32(1), fx.Int32(0)))
                last = (old == (s - 1)) & valid
                if last:
                    _atomic_add(cnt, -s)
                smem.flag.ptr[0] = last.select(fx.Int32(1), fx.Int32(0))
            fx.gpu.barrier()
            if fx.Int32(smem.flag.ptr[0]) == 1:
                chunk = max(1, EXCHANGE_CHUNK // s)
                for c0 in range_constexpr(0, t_per, chunk):
                    ts = list(range(c0, min(t_per, c0 + chunk)))
                    parts = [[Vec(buffer_ops.buffer_load(sr, slot(q, t), vec_width=4, dtype=fx.Float32,
                                                         mask=(sp != q))) for q in range_constexpr(s)] for t in ts]
                    _sb()
                    for t, pv in zip(ts, parts):
                        tot, nn, mm, live = tiles[t]
                        sums = None
                        for q in range_constexpr(s):
                            own = sp == q
                            v = Vec.from_elements([own.select(tot[e], pv[q][e]) for e in range_constexpr(4)],
                                                  fx.Float32)
                            sums = v if sums is None else sums + v
                        store_bf16(sums, nn, mm, live)

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, arg_slab: fx.Int64, arg_cnt: fx.Int64,
               i32_m: fx.Int32, i32_mtiles: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_w, arg_y, arg_slab, arg_cnt, i32_m).launch(
            grid=(PER * i32_mtiles, 1, 1), block=(64 * nw, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
