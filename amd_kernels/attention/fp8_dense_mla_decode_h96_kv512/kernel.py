# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dense MLA decode partials of the DSpark draft: 96 latent heads, e4m3 q and e4m3 latent KV (576 B per token, the first 512
bytes also V), a causal block of ``qlen`` query rows over this rank's round-robin shard of the context.

Work: the key space is cut into ``nsets`` chunks (``request_layout``: each request boundary costs ``SEG_PAD`` virtual keys,
what a segment's prologue and epilogue cost a workgroup), a chunk meets one or more requests as segments, and each segment
writes one partial (``m``, ``l``, bf16 ``O``) per (query row, head). ``merge.py`` combines the partials of a request. Grid:
``8 * ceil(3 * nsets / 8)`` workgroups of 4 waves; workgroup p runs on XCD p % 8 and takes unit
``(p % 8) * ceil(units / 8) + p // 8`` = ``3 * chunk + g``: the three head groups (32 heads each) of a chunk share one XCD's L2.

Wave w owns query row w (rows >= qlen idle) of the group and keeps Q^T (9 fragments), the running (exponent, sum) and
O^T (16 x 32x32 fp32 blocks = its 32 heads x 512 latent dims) in registers. Per tile of 64 keys, in a pipeline
(QK of tile i + 1 | softmax of tile i | PV of tile i): S^T = K Q^T with the scaled MFMA 32x32x64 (A = K rows read from LDS),
then ``p = 2^(x - e + SHIFT)`` as e4m3 with ``e = ceil(max x)`` of the tile, which is the B operand of O^T += V^T P^T (A = V^T from
the same LDS tile through ds_read_b64_tr_b8) whose per-row scale operand is ``2^(e - e_ref)``: the first tile fixes
``e_ref``, so O is never rescaled. Tiles reach LDS (``NBUF`` stages, pitch 576 B in 16-row x 64-B blocks, 16-B pieces
XOR-swizzled by row) by global_load_lds DMA behind explicit vmcnt waits and barriers; the key rows come from the CSR list
through a small LDS ring filled by DMA too. Loads of that DMA carry alias scopes (``_alias_scopes``): without them LLVM
drains the DMA before every LDS read.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, buf_base_i64
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.warm_up import launcher

HEADS = 96
KV_LORA_RANK = 512
ROPE = 64
ROW = KV_LORA_RANK + ROPE
GROUPS = 3
TT = 64
NW = 4
XCDS = 8
SHIFT = 8.0
NEG = -1.0e30
LOG2E = 1.4426950408889634
STAGE = TT * ROW
PIECES = ROW // 16
NIB = 4
SEG_PAD = 256
EPITCH = 1040
NBUF = 4
DK = 3
DV = 3


def _lds_ptr(elem, addr, align):
    return fx.inttoptr(fx.PointerType.get(elem, fx.AddressSpace.Shared, align), fx.Int32(addr))


def _lds_load(elem, n, addr, align):
    return Vec(fx.ptr_load(_lds_ptr(elem, addr, align), result_type=T.vec(n, elem)))


def _lds_store(vec, elem, addr, align):
    fx.ptr_store(Vec(vec).ir_value(), _lds_ptr(elem, addr, align))


def _lds_llvm_ptr(addr):
    return fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(addr)))


def _global_llvm_ptr(addr):
    return fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Global), fx.Int64(addr)))


_AA = {}


def _alias_scopes():
    """(write scope, read scope) arrays of one alias domain: LDS DMA stores and LDS loads never alias for LLVM's waitcnt
    insertion (the kernel orders them with explicit vmcnt waits and barriers)."""
    from flydsl._mlir import ir

    ctx = ir.Context.current
    key = id(ctx)
    if key not in _AA:
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(
            '#dom = #llvm.alias_scope_domain<id = distinct[0]<>, description = "lds">\n'
            '#sw = #llvm.alias_scope<id = distinct[1]<>, domain = #dom, description = "dma">\n'
            '#sr = #llvm.alias_scope<id = distinct[2]<>, domain = #dom, description = "rd">\n'
            '"test.op"() {w = [#sw], r = [#sr]} : () -> ()',
            ctx,
        )
        op = m.body.operations[0]
        _AA.clear()
        _AA[key] = (op.attributes["w"], op.attributes["r"])
    return _AA[key]


def _tr8(addr, imm):
    w_, r_ = _alias_scopes()
    return Vec(
        rocdl.ds_read_tr8_b64_(
            T.vec(2, T.i32), _lds_llvm_ptr(fx.Int32(addr) + imm), alias_scopes=r_, noalias_scopes=w_
        )
    )


def _lds_load_na(n, addr):
    w_, r_ = _alias_scopes()
    return Vec(
        llvm.LoadOp(T.vec(n, T.i32), _lds_llvm_ptr(addr), alignment=16, alias_scopes=r_, noalias_scopes=w_).result
    )


def _lds_load_na_scalar(addr):
    w_, r_ = _alias_scopes()
    return llvm.LoadOp(T.i32, _lds_llvm_ptr(addr), alignment=4, alias_scopes=r_, noalias_scopes=w_).result


def _dma4(src_addr, dst_ptr):
    w_, r_ = _alias_scopes()
    rocdl.global_load_lds(_global_llvm_ptr(src_addr), dst_ptr, 4, 0, alias_scopes=w_, noalias_scopes=r_)


def _dma(src_ptr, dst_ptr):
    w_, r_ = _alias_scopes()
    rocdl.global_load_lds(src_ptr, dst_ptr, 16, 0, alias_scopes=w_, noalias_scopes=r_)


def _opaque(x):
    """x through a no-op instruction the optimizer cannot see through: a value recomputed here is not kept alive from
    where it was first computed."""
    return fx.Int32(llvm.inline_asm(T.i32, [_raw(fx.Int32(x))], "v_mov_b32 $0, $1", "=v,v", has_side_effects=True))


def _cat(a, b, n):
    return Vec(a).shuffle(Vec(b), list(range(2 * n)))


def _mfma32(a, b, c, sb=127):
    return Vec(
        rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            T.vec(16, T.f32),
            [
                _raw(Vec(a).ir_value()),
                _raw(Vec(b).ir_value()),
                _raw(Vec(c).ir_value()),
                0,
                0,
                0,
                _raw(fx.Int32(127)),
                0,
                _raw(fx.Int32(sb)),
            ],
        )
    )


def _pack_fp8x4(vals):
    w = rocdl.cvt_pk_fp8_f32(
        T.i32, _raw(fx.Float32(vals[0]).ir_value()), _raw(fx.Float32(vals[1]).ir_value()), llvm.mlir_poison(T.i32), False
    )
    w = rocdl.cvt_pk_fp8_f32(T.i32, _raw(fx.Float32(vals[2]).ir_value()), _raw(fx.Float32(vals[3]).ir_value()), w, True)
    return fx.Int32(w)


def _swap_pair(fn, x):
    st = llvm.StructType.get_literal([T.i32, T.i32])
    v = _raw(fx.Float32(x).bitcast(fx.Int32).ir_value())
    r = fn(st, v, v, False, False)
    a = fx.Int32(llvm.ExtractValueOp(T.i32, r, [0]).res).bitcast(fx.Float32)
    b = fx.Int32(llvm.ExtractValueOp(T.i32, r, [1]).res).bitcast(fx.Float32)
    return fx.Float32(a), fx.Float32(b)


def _fmax(a, b):
    return fx.maxnumf(a, b, fastmath="nnan")


def _xmax2(x):
    """Max over the two lane halves (lane, lane ^ 32)."""
    a, b = _swap_pair(rocdl.permlane32_swap, x)
    return _fmax(a, b)


def _xsum2(x):
    a, b = _swap_pair(rocdl.permlane32_swap, x)
    return a + b


def _readfirstlane(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(v))))


def _readlane(v, lane):
    return fx.Int32(rocdl.readlane(T.i32, _raw(fx.Int32(v)), _raw(fx.Int32(lane))))


def _imin(a, b):
    return (a < b).select(a, b)


def _imax(a, b):
    return (a > b).select(a, b)


def _popcount_ballot(pred):
    bits = rocdl.ballot(T.i64, _raw(pred.ir_value()))
    return fx.Int32(llvm.call_intrinsic(T.i64, "llvm.ctpop.i64", [bits], [], []))


def request_layout(ip_t, n_req, nsets, lane):
    """Layout of the split-KV work, shared by the partials and the merge. Request r owns the virtual interval
    ``[ip[r] + r * SEG_PAD, ip[r + 1] + r * SEG_PAD)``: the pad is what a request boundary costs a unit (a segment's
    prologue and epilogue), so units with two segments carry fewer keys. The virtual space is cut into ``nsets`` chunks;
    request r writes one partial per chunk it touches, at slot ``lo[r] + chunk - sf[r]``.
    Returns (chunk, vs, ve, sf, cnt, lo), the last five one value per lane (lane = request)."""
    tot = _readfirstlane(ip_t[n_req])
    tv = tot + n_req * SEG_PAD
    chunk = _readfirstlane(((tv + nsets - 1) // nsets + (TT - 1)) // TT * TT)
    lm = _imin(lane, n_req)
    live = lane < n_req
    r0 = live.select(fx.Int32(ip_t[lm]), fx.Int32(0))
    r1 = live.select(fx.Int32(ip_t[_imin(lane + 1, n_req)]), fx.Int32(0))
    vs = r0 + lane * SEG_PAD
    ve = r1 + lane * SEG_PAD
    sf = vs // chunk
    cnt = (live & (r1 > r0)).select((ve - 1) // chunk - sf + 1, fx.Int32(0))
    scan = cnt
    for sh in (1, 2, 4, 8, 16, 32):
        got = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(_imax((lane - sh) * 4, fx.Int32(0))), _raw(scan)))
        scan = scan + (lane >= sh).select(got, fx.Int32(0))
    return chunk, vs, ve, sf, cnt, scan - cnt


def _guarded(pred, fn):
    @flyc.jit
    def _do(_pred=pred, _fn=fn):
        if _pred:
            _fn()

    _do()


@launcher
def compile_partials(qlen: int, cp_world: int = 8):
    assert 1 <= qlen <= NW
    cp_shift = {1: 0, 2: 1, 4: 2, 8: 3}[cp_world]
    IB = NBUF * STAGE
    TB = IB + NW * 1024
    lds_bytes = TB + 1024
    name = f"fp8_dense_mla_decode_h96_kv512_q{qlen}_w{cp_world}"

    @fx.struct
    class Smem:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kern(
        Q: fx.Tensor,
        KV: fx.Tensor,
        IDX: fx.Tensor,
        IPTR: fx.Tensor,
        GIPTR: fx.Tensor,
        QSC: fx.Tensor,
        KSC: fx.Tensor,
        PO: fx.Tensor,
        PML: fx.Tensor,
        stride_qt: fx.Int32,
        stride_qh: fx.Int32,
        n_req: fx.Int32,
        nsets: fx.Int32,
        rank: fx.Int32,
        c_scale: fx.Float32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        w = _readfirstlane(tid // 64)
        lane = tid % 64
        r32 = lane % 32
        hh = lane // 32
        pid = _readfirstlane(fx.Int32(fx.block_idx.x))

        units = nsets * GROUPS
        per_xcd = (units + (XCDS - 1)) // XCDS
        u = (pid % XCDS) * per_xcd + pid // XCDS
        u = (u < units).select(u, units - 1)
        set_ = _readfirstlane(u // GROUPS)
        g = _readfirstlane(u % GROUPS)

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_base = buf_base_i64(KV)
        idx_base = buf_base_i64(IDX)
        ip_t = GTensor(IPTR, dtype=T.i32, shape=(-1,))
        gip_t = GTensor(GIPTR, dtype=T.i32, shape=(-1,))
        qsc_t = GTensor(QSC, dtype=T.f32, shape=(-1,))
        ksc_t = GTensor(KSC, dtype=T.f32, shape=(-1,))
        po_t = GTensor(PO, dtype=T.i32, shape=(-1,))
        pml_t = GTensor(PML, dtype=T.i32, shape=(-1,))

        lds = fx.SharedAllocator().allocate(Smem).peek()
        base = _readfirstlane(fx.ptrtoint(lds.raw.ptr))

        chunk, vs_l, ve_l, sf_l, cnt_l, lo_l = request_layout(ip_t, n_req, nsets, lane)
        s0 = _readfirstlane(set_ * chunk)
        e0 = s0 + chunk
        live_l = lane < n_req
        last_tok = _readfirstlane(fx.Int32(ip_t[n_req])) - 1
        b0 = _popcount_ballot(live_l & (ve_l <= s0))
        b1 = _popcount_ballot(live_l & (vs_l < e0)) - 1
        nseg = _imax(b1 - b0 + 1, fx.Int32(0))
        for n_, v_ in enumerate((vs_l, ve_l, sf_l, lo_l)):
            _lds_store(Vec.from_elements([v_], fx.Int32), T.i32, base + TB + n_ * 256 + lane * 4, 4)
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()

        dma_off = ((lane & 3) ^ ((lane >> 4) & 3)) * 16
        sw_k = (r32 >> 2) & 3
        k_base = (r32 >> 4) * 9 * 1024 + (r32 & 15) * 64
        k_off0 = k_base + ((2 * hh) ^ sw_k) * 16
        k_off1 = k_base + ((2 * hh + 1) ^ sw_k) * 16
        qd = lane // 16
        jj = lane % 16
        mm = jj // 2
        vh = qd // 2
        tt = 8 * (mm // 4) + 4 * vh + (mm % 4)
        sw_v = (2 * (mm // 4) + vh) & 3
        v_off = [tt * 64 + 16 * ((2 * par + (qd % 2)) ^ sw_v) + 8 * (jj % 2) for par in range(2)]

        def idx_dma(seg_s, seg_e, tile):
            pos = seg_s + tile * TT + lane
            # all requests empty (CUDA-graph capture dummies): last_tok is -1; never read kv_indices[-1]
            posc = _imax(_imin((pos < seg_e).select(pos, seg_s), last_tok), fx.Int32(0))
            _dma4(idx_base + fx.Int64(posc) * 4, _lds_llvm_ptr(base + IB + w * 1024 + (tile % NIB) * 256))

        def idx_read(tile):
            return fx.Int32(_lds_load_na_scalar(base + IB + w * 1024 + (tile % NIB) * 256 + 4 * (16 * w + (lane >> 2))))

        def dma_rows(ix, stage):
            ix = (last_tok >= 0).select(ix, fx.Int32(0))  # no key at all: row 0 (always mapped), results are masked
            gbase = kv_base + fx.Int64(ix) * ROW + fx.Int64(dma_off)
            for k in range_constexpr(9):
                _dma(_global_llvm_ptr(gbase + 64 * k), _lds_llvm_ptr(base + stage * STAGE + (w * 9 + k) * 1024))

        qs = fx.Float32(qsc_t[0])
        ks = fx.Float32(ksc_t[0])
        c_log2 = c_scale * qs * ks

        def kload(stage_addr, step):
            c, tb = step // 2, step % 2
            return _cat(
                _lds_load_na(4, stage_addr + k_off0 + 1024 * c + 18432 * tb),
                _lds_load_na(4, stage_addr + k_off1 + 1024 * c + 18432 * tb),
                4,
            )

        def vload(stage_addr, j):
            rd = [_tr8(stage_addr + v_off[j % 2] + 1024 * (j // 2), u_ * 9 * 1024) for u_ in range(4)]
            return _cat(_cat(rd[0], rd[1], 2), _cat(rd[2], rd[3], 2), 4)

        def segment(si):
            b = _readfirstlane(b0 + si)
            v_b = _readfirstlane(_lds_load_na_scalar(base + TB + 4 * b))
            v_e = _readfirstlane(_lds_load_na_scalar(base + TB + 256 + 4 * b))
            sb = v_b - b * SEG_PAD
            sn = v_e - b * SEG_PAD
            seg_s = _readfirstlane(_imax(s0, v_b) - b * SEG_PAD)
            seg_e = _readfirstlane(_imin(e0, v_e) - b * SEG_PAD)
            nt = _readfirstlane((seg_e - seg_s + (TT - 1)) // TT)
            slot = _readfirstlane(_lds_load_na_scalar(base + TB + 768 + 4 * b) + set_ - _lds_load_na_scalar(base + TB + 512 + 4 * b))
            gl = _readfirstlane(fx.Int32(gip_t[b + 1]) - fx.Int32(gip_t[b]))
            j0 = seg_s - sb

            pos = _readfirstlane(_imin(w, fx.Int32(qlen - 1)))
            num = gl - qlen + pos - rank
            nvg = (num >= 0).select((num >> cp_shift) + 1, fx.Int32(0))
            lim = _readfirstlane(_imin(_imin(nvg, sn - sb), seg_e - sb))

            qrow_i32 = ((b * qlen + pos) * stride_qt + (g * 32 + r32) * stride_qh) // 4
            qf = []
            for c in range_constexpr(9):
                a0 = Vec(q_t.load(qrow_i32 + 16 * c + 8 * hh, vec_size=4))
                a1 = Vec(q_t.load(qrow_i32 + 16 * c + 8 * hh + 4, vec_size=4))
                qf.append(_cat(a0, a1, 4))

            for t_ in range_constexpr(4):
                idx_dma(seg_s, seg_e, t_)
            rocdl.s_waitcnt(vmcnt=0)
            dma_rows(idx_read(0), 0)
            dma_rows(idx_read(1), 1)
            idx_dma(seg_s, seg_e, 4)
            dma_rows(idx_read(2), 2)
            idx_dma(seg_s, seg_e, 5)
            rocdl.s_waitcnt(vmcnt=20)
            rocdl.s_barrier()

            def qk_tile(stage_addr, hook):
                sacc = [Vec.filled(16, 0.0, fx.Float32) for _ in range(2)]
                pre = [kload(stage_addr, st_) for st_ in range(DK)]
                for step in range_constexpr(18):
                    if const_expr(step + DK < 18):
                        pre.append(kload(stage_addr, step + DK))
                    sacc[step % 2] = _mfma32(pre[step], qf[step // 2], sacc[step % 2])
                    if hook is not None:
                        hook(step)
                    rocdl.sched_barrier(0)
                return sacc

            s_first = qk_tile(base, None)

            init = [_raw(fx.Float32(NEG).ir_value()), _raw(fx.Float32(0.0).ir_value())]
            init += [_raw(Vec.filled(16, 0.0, fx.Float32).ir_value()) for _ in range(16)]
            init += [_raw(Vec(x).ir_value()) for x in s_first]
            for i, stt in range(0, nt, 1, init=init):
                e_old = fx.Float32(stt[0])
                l_old = fx.Float32(stt[1])
                o = [Vec(stt[2 + d]) for d in range(16)]
                s_cur = [Vec(stt[18]), Vec(stt[19])]
                ii = fx.Int32(i)
                rocdl.s_waitcnt(vmcnt=11)
                rocdl.s_barrier()
                ix = (last_tok >= 0).select(idx_read(ii + 3), fx.Int32(0))
                gbase = kv_base + fx.Int64(ix) * ROW + fx.Int64(dma_off)
                dst_stage = base + ((ii + (NBUF - 1)) % NBUF) * STAGE

                def dma_issue(k):
                    if k < 9:
                        _dma(_global_llvm_ptr(gbase + 64 * k), _lds_llvm_ptr(dst_stage + (w * 9 + k) * 1024))

                thr = lim - (j0 + ii * TT)
                if thr < TT:
                    s_cur = [
                        Vec.from_elements(
                            [
                                (4 * hh + 32 * tb + 8 * (i16 // 4) + (i16 % 4) < thr).select(
                                    fx.Float32(Vec(s_cur[tb])[i16]), fx.Float32(float("-inf"))
                                )
                                for i16 in range(16)
                            ],
                            fx.Float32,
                        )
                        for tb in range(2)
                    ]
                vals = [fx.Float32(Vec(s_cur[tb])[i16]) for tb in range(2) for i16 in range(16)]
                sm = {}
                ops = []
                sm["m"] = fx.Float32(NEG)

                def mk_max(k):
                    def f():
                        sm["m"] = _fmax(_fmax(sm["m"], vals[2 * k]), vals[2 * k + 1])

                    return f

                for k in range_constexpr(16):
                    ops.append(mk_max(k))

                def reduce_stage():
                    tmax = _xmax2(sm["m"]) * c_log2
                    e_t = fx.Float32(fx.ceil(tmax))
                    valid = tmax > fx.Float32(-1.0e29)
                    unset = e_old < fx.Float32(-1.0e29)
                    er = (unset & valid).select(e_t, e_old)
                    e_use = valid.select(e_t, er)
                    dl = e_use - er
                    sm["er"] = er
                    sm["bias"] = SHIFT - e_use
                    sm["scl"] = dl.exp2(fastmath="afn")
                    sm["sbyte"] = _imin(_imax(fx.Int32(dl) + 127, fx.Int32(0)), fx.Int32(254))

                ops.append(reduce_stage)
                sm["p"] = [None] * 32

                def mk_exp(k):
                    def f():
                        sm["p"][k] = fx.Float32(fmath.fma(vals[k], c_log2, sm["bias"])).exp2(fastmath="afn")

                    return f

                for k in range_constexpr(32):
                    ops.append(mk_exp(k))
                sm["acc"] = [fx.Float32(0.0), fx.Float32(0.0)]

                def mk_sum(k):
                    def f():
                        sm["acc"][k % 2] = sm["acc"][k % 2] + sm["p"][2 * k] + sm["p"][2 * k + 1]

                    return f

                for k in range_constexpr(16):
                    ops.append(mk_sum(k))

                def fin_l():
                    sm["l"] = l_old + (sm["acc"][0] + sm["acc"][1]) * sm["scl"]

                ops.append(fin_l)
                sm["pw"] = [None] * 8

                def mk_pack(k):
                    def f():
                        sm["pw"][k] = _pack_fp8x4(sm["p"][4 * k : 4 * k + 4])

                    return f

                for k in range_constexpr(8):
                    ops.append(mk_pack(k))
                per = (len(ops) + 16) // 17

                def hook(step):
                    if step >= 1:
                        [f() for f in ops[(step - 1) * per : step * per]]

                s_next = qk_tile(base + ((ii + 1) % NBUF) * STAGE, hook)
                [f() for f in ops[16 * per :]]

                pown = Vec.from_elements(sm["pw"], fx.Int32)
                v_stage = base + (ii % NBUF) * STAGE
                vpre = [vload(v_stage, j) for j in range(DV)]
                for j in range_constexpr(16):
                    if const_expr(j + DV < 16):
                        vpre.append(vload(v_stage, j + DV))
                    o[j] = _mfma32(vpre[j], pown, o[j], sm["sbyte"])
                    dma_issue(j // 2 if j % 2 == 0 else 99)
                    rocdl.sched_barrier(0)
                dma_issue(8)
                idx_dma(seg_s, seg_e, ii + 6)
                fin = yield [sm["er"], sm["l"]] + o + s_next

            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            e_f = fx.Float32(fin[0])
            l_f = _xsum2(fx.Float32(fin[1]))

            def _store_ml():
                def _st_ml():
                    ml_base = ((slot * qlen + pos) * HEADS + g * 32 + r32) * 2
                    pml_t.store(ml_base, (e_f - SHIFT).bitcast(fx.Int32))
                    pml_t.store(ml_base + 1, l_f.bitcast(fx.Int32))

                _guarded(hh == 0, _st_ml)

            _guarded((w < qlen) & (nt > 0), _store_ml)

            lane_e = _opaque(tid) % 64
            r32_e = lane_e % 32
            hh_e = lane_e // 32
            ebase = base + w * (32 * EPITCH)
            for db in range_constexpr(16):
                ov = Vec(fin[2 + db])
                for a_ in range_constexpr(4):
                    pk = Vec.from_elements([fx.Float32(ov[4 * a_ + k_]) for k_ in range(4)], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                    _lds_store(pk, T.i32, ebase + r32_e * EPITCH + 2 * (32 * db + 8 * a_ + 4 * hh_e), 8)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()

            def _store_o():
                po_row0 = ((slot * qlen + pos) * HEADS + g * 32) * KV_LORA_RANK
                st_po = po_t.store
                for i_ in range(0, 32, 1):
                    v4 = _lds_load(T.i32, 4, ebase + fx.Int32(i_) * EPITCH + lane_e * 16, 16)
                    st_po((po_row0 + fx.Int32(i_) * KV_LORA_RANK) // 2 + lane_e * 4, v4, vec_size=4)

            _guarded((w < qlen) & (nt > 0), _store_o)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()

        for si in range(0, nseg, 1):
            segment(si)

    @flyc.jit
    def launch(
        Q,
        KV,
        IDX,
        IPTR,
        GIPTR,
        QSC,
        KSC,
        PO,
        PML,
        stride_qt,
        stride_qh,
        n_req,
        nsets,
        rank,
        c_scale,
        grid_x,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kern(Q, KV, IDX, IPTR, GIPTR, QSC, KSC, PO, PML, stride_qt, stride_qh, n_req, nsets, rank, c_scale).launch(
            grid=(fx.Int64(grid_x), 1, 1), block=(64 * NW, 1, 1), stream=stream
        )

    launch.compile_hints = {"waves_per_eu": 1, "fast_fp_math": False}
    return launch
