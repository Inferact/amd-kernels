# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# SPDX-FileCopyrightText: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""FP4 MQA logits for ragged prefill rows over staged index key pages (gfx950, FlyDSL).

Adapted from AITER's ``pa_mqa_logits_fp4_prefill`` (aiter/ops/flydsl/kernels/mqa_logits, MIT).

A workgroup owns ``rows_per_cta`` consecutive query rows and a slice of their shared key window; each wave walks its slice
one 64-key page at a time (identity page table), so one load of a page feeds every row's MFMAs. Per 16-key tile and row
two scaled FP4 MFMAs give the 32 head scores; ``sum_h w_h relu(x_h)`` is evaluated as ``(sum_h w_h x_h + sum_h w_h |x_h|) / 2``
(two full-rate FMAs per score, |x| a free source modifier, the 1/2 folded into the weights). The four tiles of a page are
reduced across the lane groups together (``permlane32/16_swap``), so each lane ends with one key's logit and a row stores
256 contiguous bytes per page. Rows are software-pipelined: the next row's MFMAs issue while this row's epilogue runs.

Whole pages are stored: positions outside a row's window ``[ks, ke)`` inside the group's page range receive unspecified values.

Limit (measured on MI355X, ``benchmarks/.../benchmark_score.py``): a wave's VALU and MFMA work do
not overlap on a SIMD (1, 2 and 4 waves per SIMD), an independent ``v_fma_f32`` costs 2.3-2.5 cycles, ``v_max_f32`` 4, a
16x16x128 FP4 MFMA 17.8 (20 on random data). A page costs about 161 (8 MFMAs) + 245 (epilogue VALU) cycles per row plus
467 / rows_per_cta per row for the page's loads and the pipeline fill; the kernel runs at that sum, not at the MFMA peak.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import Float4E2M1FN, T, as_ir_value

from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher

I, F, I64 = fx.Int32, fx.Float32, fx.Int64
HEADS = 32
PAGE_KEY_BYTES = 4096
PAGE_SCALE_BYTES = 256
MAX_OUT_BYTES = 0x7FFFFFF0
VALU_PER_MFMA = 6  # epilogue VALU ops scheduled between two MFMAs (sweep: 6 > 8 > 10)
MFMA_MASK, VALU_MASK = 0x008, 0x002


def _permlane_swap(swap, x, y):
    pair = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(x.bitcast(I)),
        as_ir_value(y.bitcast(I)),
        False,
        False,
    )
    return tuple(I(llvm.extractvalue(T.i32, pair, [j])).bitcast(F) for j in range(2))


def _vop(asm, *args):
    return F(
        llvm.inline_asm(T.f32, [F(a).ir_value() for a in args], asm, "=v" + ",v" * len(args), has_side_effects=False)
    )


def _global_i32(addr, words, nontemporal=False):
    ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4 * words), I64(addr))
    return fx.generic_load(ptr, dtype=fx.Vector[I, words], nontemporal=nontemporal).ir_value()


def _add(a, b):
    return _vop("v_add_f32 $0, $1, $2", a, b)


def _fma(a, b, c):
    return _vop("v_fma_f32 $0, $1, $2, $3", a, b, c)


def _fma_abs(a, b, c):
    return _vop("v_fma_f32 $0, |$1|, $2, $3", a, b, c)


@launcher
def compile_score_rows(rows_per_cta: int, waves: int):
    R = rows_per_cta

    @flyc.kernel(known_block_size=[waves * 64, 1, 1])
    def score_rows_kernel(
        arg_q: fx.Int64,
        arg_qs: fx.Int64,
        arg_w: fx.Int64,
        arg_kv: fx.Int64,
        arg_kvs: fx.Int64,
        arg_ks: fx.Int64,
        arg_ke: fx.Int64,
        arg_out: fx.Int64,
        i32_stride: fx.Int32,
        i32_rows: fx.Int32,
        i32_groups: fx.Int32,
        i32_splits: fx.Int32,
        i32_last_page: fx.Int32,
        f_weight_scale: fx.Float32,
    ):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = I(rocdl.readfirstlane(T.i32, tx // 64))
        l = lane & 15
        g = (lane >> 4) & 3
        pid = I(fx.block_idx.x)
        group = pid % i32_groups
        split = pid // i32_groups
        r0 = group * R
        ks_res = buffer_ops.create_buffer_resource_from_addr(arg_ks)
        ke_res = buffer_ops.create_buffer_resource_from_addr(arg_ke)
        rowv, ksv, kev = [], [], []
        for r in range_constexpr(R):
            live = (r0 + r) < i32_rows
            row = fx.min(r0 + r, i32_rows - 1)
            s = I(buffer_ops.buffer_load(ks_res, row, vec_width=1, dtype=I))
            e = I(buffer_ops.buffer_load(ke_res, row, vec_width=1, dtype=I))
            rowv.append(row)
            ksv.append(live.select(s, I(0)))
            kev.append(live.select(e, I(0)))
        min_ks = None
        max_ke = kev[0]
        for r in range_constexpr(R):
            cand = (kev[r] > 0).select(ksv[r], I(0x7FFFFFFF))
            min_ks = cand if min_ks is None else fx.min(min_ks, cand)
            if r:
                max_ke = fx.max(max_ke, kev[r])
        page_lo = min_ks // 64
        page_hi = (max_ke + 63) // 64
        per_split = (fx.max(page_hi - page_lo, I(0)) + i32_splits - 1) // i32_splits
        p_lo = page_lo + split * per_split
        p_hi = fx.min(p_lo + per_split, page_hi)
        if p_lo < p_hi:
            q_r = buffer_ops.create_buffer_resource_from_addr(arg_q)
            qs_r = buffer_ops.create_buffer_resource_from_addr(arg_qs)
            w_r = buffer_ops.create_buffer_resource_from_addr(arg_w)
            out_r = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=MAX_OUT_BYTES)
            atoms = [
                fx.make_mma_atom(
                    fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float4E2M1FN, Float4E2M1FN, opsel_a=0, opsel_b=nt)
                )
                for nt in range(4)
            ]
            q_ops, q_sc, w_l = [], [], []
            for r in range_constexpr(R):
                row = rowv[r]
                ops = []
                for mi in range_constexpr(2):
                    head = row * HEADS + mi * 16 + l
                    frag = fx.make_rmem_tensor(4, I)
                    frag.store(fx.Vector(buffer_ops.buffer_load(q_r, head * 16 + g * 4, vec_width=4, dtype=I)))
                    ops.append(frag)
                q_ops.append(ops)
                dw = I(buffer_ops.buffer_load(qs_r, (row * 4 + g) * 16 + l, vec_width=1, dtype=I))
                q_sc.append([dw >> I(8 * mi) for mi in range(2)])
                wl = []
                for mi in range_constexpr(2):
                    w2 = fx.Vector(
                        buffer_ops.buffer_load(w_r, (row * HEADS + mi * 16 + g * 4) // 2, vec_width=2, dtype=I)
                    )
                    d0, d1 = I(w2[0]), I(w2[1])
                    vals = [
                        (d0 << I(16)).bitcast(F),
                        (d0 & I(-65536)).bitcast(F),
                        (d1 << I(16)).bitcast(F),
                        (d1 & I(-65536)).bitcast(F),
                    ]
                    wl.append([v * f_weight_scale * 0.5 for v in vals])
                w_l.append(wl)

            def load_page(page_raw):
                page = I(rocdl.readfirstlane(T.i32, fx.min(page_raw, i32_last_page)))
                keys = arg_kv + I64(page) * I64(PAGE_KEY_BYTES)
                scales = arg_kvs + I64(page) * I64(PAGE_SCALE_BYTES)
                kvs = fx.Vector(_global_i32(scales + I64(g * 64 + l * 4), 1))[0]
                kv = [_global_i32(keys + I64(g * 1024 + l * 16 + nt * 256), 4, True) for nt in range(4)]
                return kv, kvs

            def mfma_row(r, bfrags, kvs):
                accs = []
                for nt in range_constexpr(4):
                    pair = []
                    for mi in range_constexpr(2):
                        acc = fx.make_rmem_tensor(4, F)
                        acc.store(fx.full(4, 0.0, F))
                        fx.gemm(atoms[nt], acc, q_ops[r][mi], bfrags[nt], acc, scale_a=q_sc[r][mi], scale_b=kvs)
                        pair.append(acc)
                    accs.append(pair)
                return accs

            def epilogue_row(r, accs, key):
                tiles = []
                for nt in range_constexpr(4):
                    lin = None
                    mag = None
                    for mi in range_constexpr(2):
                        x4 = fx.Vector(accs[nt][mi].load())
                        for e in range_constexpr(4):
                            w = w_l[r][mi][e]
                            x = x4[e]
                            lin = _vop("v_mul_f32 $0, $1, $2", x, w) if lin is None else _fma(x, w, lin)
                            mag = _vop("v_mul_f32 $0, |$1|, $2", x, w) if mag is None else _fma_abs(x, w, mag)
                    tiles.append(_add(lin, mag))
                x, y = _permlane_swap(rocdl.permlane32_swap, tiles[0], tiles[2])
                low = _add(x, y)
                x, y = _permlane_swap(rocdl.permlane32_swap, tiles[1], tiles[3])
                high = _add(x, y)
                x, y = _permlane_swap(rocdl.permlane16_swap, low, high)
                buffer_ops.buffer_store(_add(x, y), out_r, rowv[r] * i32_stride + key)

            def score_page(page, kv, kvs):
                bfrags = []
                for nt in range_constexpr(4):
                    bf = fx.make_rmem_tensor(4, I)
                    bf.store(fx.Vector(kv[nt]))
                    bfrags.append(bf)
                key = page * 64 + g * 16 + l
                cur = mfma_row(0, bfrags, kvs)
                for r in range_constexpr(R):
                    nxt = mfma_row(r + 1, bfrags, kvs) if r + 1 < R else None
                    epilogue_row(r, cur, key)
                    if nxt is not None:
                        for _ in range_constexpr(8):
                            rocdl.sched_group_barrier(MFMA_MASK, 1, r)
                            rocdl.sched_group_barrier(VALU_MASK, VALU_PER_MFMA, r)
                    rocdl.sched_barrier(0)
                    cur = nxt

            n_pages = (p_hi - p_lo - wave + (waves - 1)) // waves
            kv0, kvs0 = load_page(p_lo + wave)
            if n_pages > 0:
                init_args = list(kv0) + [kvs0]
                for it_idx, state in range(0, fx.Int64(n_pages), 1, init=init_args):
                    kv_cur = [state[i] for i in range(4)]
                    kvs_cur = state[4]
                    page = p_lo + wave + fx.Int32(it_idx) * waves
                    kv_next, kvs_next = load_page(page + waves)
                    score_page(page, kv_cur, kvs_cur)
                    results = yield list(kv_next) + [kvs_next]

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_qs: fx.Int64,
        arg_w: fx.Int64,
        arg_kv: fx.Int64,
        arg_kvs: fx.Int64,
        arg_ks: fx.Int64,
        arg_ke: fx.Int64,
        arg_out: fx.Int64,
        i32_stride: fx.Int32,
        i32_rows: fx.Int32,
        i32_groups: fx.Int32,
        i32_splits: fx.Int32,
        i32_last_page: fx.Int32,
        f_weight_scale: fx.Float32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        score_rows_kernel(
            arg_q, arg_qs, arg_w, arg_kv, arg_kvs, arg_ks, arg_ke, arg_out,
            i32_stride, i32_rows, i32_groups, i32_splits, i32_last_page, f_weight_scale,
        ).launch(grid=(I64(grid_x), 1, 1), block=(waves * 64, 1, 1), stream=stream)

    return launch
