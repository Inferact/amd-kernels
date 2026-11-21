# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 [M, 7168] x bf16 [3584, 7168]^T -> MXFP4 codes [M, 1792] + e8m0 scales [M, 112] (the latent down projection
with the a4w4 activation quant in its epilogue), 1-192 rows.

Tiles: 56 column tiles of 64 weight rows (4 groups of 16) x KS K slices of 112 / KS blocks of 64: grid 56 KS, KS 4 is
224 workgroups, one per CU (KS 7, 8 measured slower). A workgroup covers all M rows. Both operands go HBM / L2 -> LDS
by asynchronous `buffer_load ... lds` into a ring of R slots (one slot = the 4 weight fragments + MT activation
fragments, 2 KB each, as gemm/bf16_n50016_k6144: XOR-swizzled fragment reads, v_mfma_f32_16x16x32_bf16, K ascending in
a slice). Weight loads are non-temporal: the weight streams through. Wave (pair, half) multiplies the K half `half`
of every block for the weight groups 2 pair and 2 pair + 1, so an activation fragment read from LDS feeds two MFMAs;
the halves add in a fixed order (half 0 + half 1) through LDS.

Epilogue: wave half 0 stores the workgroup's fp32 accumulators (one contiguous 1 KB per fragment) to its slice of the
tile's slab, the workgroup drains and increments the tile's counter (a scalar atomic that only ever counts up; the tile's
KS = 4 workgroups meet at multiples of 4, so nothing is reset), then waits until all four have arrived. Workgroup kz
reduces a quarter of the tile: the 32-column block kz // 2 of the fragments 2 i + kz % 2 (a workgroup with no fragment
leaves at once): its four slices summed in slice order (the same bits whichever arrives last), rounded to bf16 into an LDS
tile, then one thread per (row, block) quantizes: |max| by v_max3, the e8m0 exponent ceil(log2(amax / 6)) floored at 92,
e2m1 by v_cvt_scalef32_pk_fp4_f32 (RNE, saturating), one 16-byte code store and one scale byte; the bf16 tile is
optionally stored too. Counters are tickets, never an output sum. The wait needs the tile's four workgroups to be
resident together: they are consecutive (or, with ``xcd``, one XCC's) workgroups of a grid that fits the GPU.

``xcd`` placement: the dispatcher sends workgroup b to XCC b % 8, and each XCC has its own L2, so the slab exchange
costs a write-through to memory (sc1) when the four slices of a tile sit on four XCCs. With ``xcd`` the tile's slices are
workgroups x + 8 (4 j .. 4 j + 3), one XCC, and the slab stays in that L2 (sc0 loads and stores); the host checks the
placement with a probe launch (``xcc_probe``) once per device and uses the other placement if it does not hold.

Weight layouts: ROW (the checkpoint's [3584, 7168]) or SHUF ([224 groups, 112 K blocks, 16 rows, 64 K]: one DMA reads
1 KB in one piece, a bit-exact permutation made once at load).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

from .reference import K, N

TN = 64
TILES = N // TN  # 56
NB = K // 64  # 112 K blocks
NWV = 4
W_CM = 2  # nt
SC0 = 0x1  # workgroup-coherent through the L2 the K slices of a tile share (placement tile_peers_share_l2)
SC1 = 0x10  # agent-scope (cross-XCD) partial stores / loads
XCDS = 8
MT_MAX = 12
RING_BYTES = 160 * 1024
CNT_STRIDE = 128  # bytes: one counter per tile, each on its own line
TS = 144  # LDS bf16 tile row stride, bytes
SCALE_FLOOR = 92
QCOLS = N // 2
QGROUPS = N // 32


def ring_depth(mt: int) -> int:
    return 4 if (4 + mt) * 2048 * 4 <= RING_BYTES else 3


def partial_bytes(mt: int, ks: int) -> int:
    return TILES * ks * 16 * mt * TN * 4


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def _atomic_add(addr, v):
    """Returning scalar global RMW by one active lane; the caller drained its sc1 stores first."""
    low = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int64(addr).to(fx.Int32)))
    high = fx.Int32(fx.rocdl.readfirstlane(T.i32, (fx.Int64(addr) >> 32).to(fx.Int32)))
    pointer = (low.to(fx.Uint32).to(fx.Uint64) | (high.to(fx.Uint32).to(fx.Uint64) << 32)).bitcast(fx.Int64)
    inc = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))
    return fx.Int32(llvm.inline_asm(T.i32, [fx.as_ir_value(inc), fx.as_ir_value(pointer)],
                                    "s_atomic_add $0, $2, 0 glc\n\ts_waitcnt lgkmcnt(0)", "=s,0,s,~{memory}",
                                    has_side_effects=True, is_align_stack=False,
                                    asm_dialect=llvm.AsmDialect.AD_ATT))


def _fmax3_abs(a, b, c):
    return fx.Float32(llvm.inline_asm(T.f32, [fx.as_ir_value(fx.Float32(v)) for v in (a, b, c)],
                                      "v_max3_f32 $0, |$1|, |$2|, |$3|", "=v,v,v,v", has_side_effects=False))


def _cvt_pk_fp4(old, a, b, scale_f32, sel):
    """v_cvt_scalef32_pk_fp4_f32: two f32 / scale -> two e2m1 into byte ``sel`` of ``old``."""
    return fx.Int32(llvm.call_intrinsic(T.i32, "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                                        [fx.as_ir_value(old), fx.as_ir_value(a), fx.as_ir_value(b),
                                         fx.as_ir_value(scale_f32), fx.as_ir_value(fx.Int32(sel))], [], []))


def _f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


@launcher
def compile_bf16_latent_down_mxfp4_quant_k7168_n3584(mt: int, ks: int, shuf: int, with_bf16: int, xcd: int):
    MT, KS, SHUF, WB, XCDL = mt, ks, shuf, with_bf16, xcd
    SLAB = SC0 if XCDL else SC1
    assert 1 <= MT <= MT_MAX and KS == 4 and NB % KS == 0
    NBK = NB // KS  # K blocks per workgroup
    MR = 16 * MT
    R = min(ring_depth(MT), NBK)
    WF = NWV  # weight positions per slot
    SLOT = (WF + MT) * 2048
    LDS = R * SLOT
    XD = MT * 2  # activation DMAs per block
    XPW = (XD + NWV - 1) // NWV
    PER = 2 + XPW  # DMAs per wave per block
    TILE_BYTES = MR * TS
    assert LDS <= RING_BYTES and (R - 1) * PER <= 63 and TILE_BYTES <= LDS and 4 * MT * 1024 <= LDS, (LDS, R, PER)
    PBYTES = partial_bytes(MT, KS)
    
    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    name = f"bf16_latent_down_mxfp4_quant_k7168_n3584_mt{MT}_ks{KS}" + ("_shuf" if SHUF else "_row") \
        + ("_bf16" if WB else "") + ("_xcd" if XCDL else "")

    @flyc.kernel(name=name, known_block_size=[64 * NWV, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_q: fx.Int64, arg_s: fx.Int64, arg_o: fx.Int64,
               arg_p: fx.Int64, arg_c: fx.Int64, i32_m: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        if const_expr(XCDL):
            kz = (bid // XCDS) % KS
            tile = (bid % XCDS) * (TILES // XCDS) + (bid // XCDS) // KS
        else:
            kz = bid % KS
            tile = bid // KS
        n0 = tile * TN
        b0 = kz * NBK
        g = lane // 16
        l16 = lane % 16
        r8 = lane // 8

        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_m * (K * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K * 2)
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))

        # DMA jj of a fragment: rows 8 jj + r8, 16-byte piece (lane % 8) ^ ((r8 >> 1) | (jj << 2)) of the line
        w_src, w_dst = [], []
        for jj in range_constexpr(2):
            pc = (lane % 8) ^ ((r8 // 2) + jj * 4)
            if const_expr(SHUF):
                off = (tile * NWV + wave) * (K * 32) + (jj * 8 + r8) * 128 + pc * 16
            else:
                off = (n0 + wave * 16 + jj * 8 + r8) * (K * 2) + pc * 16
            w_src.append(off)
            w_dst.append((wave * 2 + jj) * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave + i * NWV
            d = (d < XD).select(d, fx.Int32(XD - 1))  # spare DMAs repeat the last one (same bytes, same place)
            jx = d % 2
            row = (d // 2) * 16 + jx * 8 + r8
            row = (row < i32_m).select(row, i32_m - 1)  # rows past M read the last row; never stored
            px = (lane % 8) ^ ((r8 // 2) + jx * 4)
            x_src.append(row * (K * 2) + px * 16)
            x_dst.append((WF * 2 + d) * 1024)
        aux_w = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), W_CM)
        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def issue(rs, dst, src, b, soff, aux):
            d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + (b % R) * SLOT + dst))
            lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
            _rocdl_ods.raw_ptr_buffer_load_async_lds(
                rs, lp, fx.Int32(16).ir_value(), fx.Int32(src).ir_value(),
                fx.Int32(soff).ir_value(), fx.Int32(0).ir_value(), aux=aux)

        def dma(b):
            bg = b0 + b
            for q in range_constexpr(2):
                issue(wr, w_dst[q], w_src[q], b, bg * (2048 if SHUF else 128), aux_w)
            for q in range_constexpr(XPW):
                issue(xr, x_dst[q], x_src[q], b, bg * 128, aux_x)

        r8r = l16 % 8
        hh = l16 // 8
        pair = wave // 2  # this wave's weight groups 2 pair, 2 pair + 1
        half = wave % 2  # and the K half (of each 64-block) it multiplies
        rd_off = hh * 1024 + (r8r * 8 + ((half * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16

        def rd(b, pos):
            p = _lds_ptr(lds_base + (b % R) * SLOT + pos * 2048 + rd_off)
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MT)] for _ in range_constexpr(2)]

        def read_ops(b):
            return [rd(b, pair * 2 + ig) for ig in range_constexpr(2)], [rd(b, WF + m) for m in range_constexpr(MT)]

        def mfmas(ops):
            wv, xv = ops
            for m in range_constexpr(MT):
                for ig in range_constexpr(2):
                    acc[ig][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [wv[ig], xv[m], acc[ig][m], 0, 0, 0]))

        def barrier():
            llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)

        # each wave multiplies one K half of every block for two weight groups, so an activation fragment read from
        # LDS feeds two MFMAs; the reads of block b + 1 overlap the MFMAs of block b, and slot b is refilled
        # (block b + R) once every wave has read it (lgkmcnt(0) before the barrier)
        for b in range_constexpr(R):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=(R - 1) * PER)
        barrier()
        _sb()
        ops_a = read_ops(0)
        for b in range_constexpr(NBK):
            _sb()
            if const_expr(b + 1 < NBK):
                fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=(min(b + R - 1, NBK - 1) - (b + 1)) * PER)
                barrier()
                _sb()
                ops_b = read_ops(b + 1)
                _sb()
                if const_expr(b + R < NBK):
                    dma(b + R)
                _sb()
            mfmas(ops_a)
            if const_expr(b + 1 < NBK):
                ops_a = ops_b

        # the K halves sum in a fixed order: half 0 + half 1
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        for ig in range_constexpr(2):
            for m in range_constexpr(MT):
                if half == 1:
                    llvm.store(acc[ig][m].bitcast(fx.Int32), _lds_ptr(
                        lds_base + (((pair * 2 + ig) * MT + m) * 64 + lane) * 16), alignment=16)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        for ig in range_constexpr(2):
            for m in range_constexpr(MT):
                other = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(
                    lds_base + (((pair * 2 + ig) * MT + m) * 64 + lane) * 16), alignment=16)).bitcast(fx.Float32)
                acc[ig][m] = acc[ig][m] + other

        pr = buffer_ops.create_buffer_resource_from_addr(arg_p, num_records_bytes=PBYTES)

        def slab_off(k, m, grp):  # dwords of this lane's 4 fp32 of fragment m, weight group grp, in slice k
            return ((((tile * KS + k) * NWV + grp) * MT + m) * 64 + lane) * 4

        for ig in range_constexpr(2):
            for m in range_constexpr(MT):
                buffer_ops.buffer_store(acc[ig][m].bitcast(fx.Int32), pr, slab_off(kz, m, pair * 2 + ig) * 4,
                                        offset_is_bytes=True, mask=(half == 0) & ((m * 16 + l16) < i32_m),
                                        cache_modifier=SLAB)
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        rh = kz % 2  # this workgroup reduces the fragments 2 i + rh of the 32-column block kz // 2
        cb = kz // 2
        NF = (fx.Int32(MT + 1) - rh) // 2
        need = (NF > 0).select(fx.Int32(KS), fx.Int32(0))
        if tx == 0:
            cnt_addr = arg_c + fx.Int64(tile) * CNT_STRIDE
            old = _atomic_add(cnt_addr, 1)
            ebase = old - (old & (KS - 1))
            arrived = old + 1 - ebase
            while arrived < need:
                arrived = _atomic_add(cnt_addr, 0) - ebase
        fx.gpu.barrier()

        def epilogue():
            tile_base = lds_base
            grp = cb * 2 + wave % 2
            for jj in range_constexpr((((MT + 1) // 2) + 1) // 2):
                i = jj * 2 + wave // 2
                m = i * 2 + rh
                m = (m < MT).select(m, fx.Int32(MT - 1))
                parts = [Vec(buffer_ops.buffer_load(pr, slab_off(k, m, grp), vec_width=4, dtype=fx.Int32,
                                                    cache_modifier=SLAB)).bitcast(fx.Float32)
                         for k in range_constexpr(KS)]
                v = parts[0]
                for k in range_constexpr(1, KS):
                    v = v + parts[k]
                pk = v.to(fx.BFloat16).bitcast(fx.Int32)
                p = _lds_ptr(tile_base + (m * 16 + l16) * TS + (grp * 16 + g * 4) * 2)
                llvm.store(pk, p, alignment=8)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=i32_m * QCOLS)
            sr = buffer_ops.create_buffer_resource_from_addr(arg_s, num_records_bytes=i32_m * QGROUPS)
            orr = buffer_ops.create_buffer_resource_from_addr(arg_o, num_records_bytes=i32_m * (N * 2))
            i = tx // 16
            m = i * 2 + rh
            r = m * 16 + tx % 16
            live = (tx < NF * 16) & (r < i32_m)
            src = tile_base + live.select(r, fx.Int32(0)) * TS + cb * 64
            words = [Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(src + 16 * q), alignment=16))
                     for q in range_constexpr(4)]
            vals = []
            for q in range_constexpr(4):
                for k in range_constexpr(4):
                    w = fx.Int32(words[q][k])
                    vals.append(_f32(w << 16))
                    vals.append(_f32(w & fx.Int32(-65536)))
            amax = _fmax3_abs(vals[0], vals[1], vals[2])
            for q in range_constexpr(3, 31, 2):
                amax = _fmax3_abs(vals[q], vals[q + 1], amax)
            amax = _fmax3_abs(vals[31], vals[31], amax)
            t = (((amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)) + fx.Int32(0x7FFFFF)) >> 23
            e8 = (t > SCALE_FLOOR).select(t, fx.Int32(SCALE_FLOOR))
            scale_f = _f32(e8 << 23)
            outw = []
            for d in range_constexpr(4):
                pw = fx.Int32(0)
                for k in range_constexpr(4):
                    pw = _cvt_pk_fp4(pw, vals[8 * d + 2 * k], vals[8 * d + 2 * k + 1], scale_f, k)
                outw.append(pw)
            buffer_ops.buffer_store(Vec.from_elements(outw, fx.Int32), qr, r * (QCOLS // 4) + tile * 8 + cb * 4,
                                    mask=live)
            buffer_ops.buffer_store(e8.to(fx.Int8), sr, r * QGROUPS + tile * 2 + cb, mask=live)
            if const_expr(WB):
                for q in range_constexpr(4):
                    buffer_ops.buffer_store(words[q], orr, r * (N // 2) + (n0 + cb * 32) // 2 + q * 4, mask=live)

        if NF > 0:
            epilogue()

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_q: fx.Int64, arg_s: fx.Int64, arg_o: fx.Int64,
               arg_p: fx.Int64, arg_c: fx.Int64, i32_m: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_w, arg_q, arg_s, arg_o, arg_p, arg_c, i32_m).launch(
            grid=(TILES * KS, 1, 1), block=(64 * NWV, 1, 1), stream=stream)

    return launch


def _xcc_id():
    return fx.Int32(llvm.inline_asm(T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)\n\ts_waitcnt lgkmcnt(0)", "=s",
                                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@flyc.kernel(name="xcc_probe", known_block_size=[64, 1, 1])
def _xcc_probe(arg_o: fx.Int64):
    bid = fx.block_idx.x
    out = buffer_ops.create_buffer_resource_from_addr(arg_o, num_records_bytes=TILES * 4 * 4)
    if fx.thread_idx.x == 0:
        buffer_ops.buffer_store(_xcc_id(), out, bid)


@flyc.jit
def xcc_probe(arg_o: fx.Int64, stream: fx.Stream):
    _xcc_probe(arg_o).launch(grid=(TILES * 4, 1, 1), block=(64, 1, 1), stream=stream)


def tile_peers_share_l2(xcc_ids) -> bool:
    """Whether the workgroups of each tile (placement ``XCDL``: bids x + 8 (4 j .. 4 j + 3)) ran on one XCC."""
    n = TILES * 4
    return all(len({xcc_ids[x + XCDS * (4 * j + q)] for q in range(4)}) == 1
               for x in range(XCDS) for j in range(n // XCDS // 4))
