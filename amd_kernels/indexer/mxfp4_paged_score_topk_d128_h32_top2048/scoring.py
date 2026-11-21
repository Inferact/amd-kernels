# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# The MFMA and epilogue arithmetic of the scorer is AITER's (aiter/ops/flydsl/kernels/mqa_logits/
# pa_mqa_logits_fp4_rowgroup.py, MIT, Copyright (C) 2024-2026, Advanced Micro Devices, Inc.), page 64, 32 heads.
"""Scores one chunk of a sequence's keys for a tile of its query rows into LDS as order-preserving int32 keys.

A wave walks 64-key steps ``wave, wave + W, ...`` of the chunk (one 64-row page a step), every row of the tile off the same
key loads: lane ``(g, l)`` feeds MFMA column ``l`` of key tile ``nt`` with key ``16 nt + l`` (head dims
``[32 g, 32 g + 32)``), the two head tiles' ``w_h * relu(logit_h)`` sum in lane, and two permlane32 swaps and one
permlane16 swap leave group ``g`` holding the sum of tile ``g``, key ``16 g + l`` of the step. Keys at or past a row's bound
are stored as ``INT_MIN``. The logits equal AITER's ``flydsl_pa_mqa_logits_fp4_rowgroup`` bit for bit.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import Float4E2M1FN, T

from ._wave import F, I, I64, INT_MIN, global_i32, permlane_swap, relu2, score_key, store_lds, vop

HEADS = 32
SPEC_SIZES = (512, 1024, 1536, 2048)


@dataclass
class Scorer:
    """The kernel's values a scoring call reads: operand addresses, the sequence's page table row and rows."""

    wave: object
    lane: object
    q: object
    qs: object
    kv: object
    kvs: object
    bt: object
    w: object
    bt_stride: object
    page_stride: object
    last_page: object
    seq: object  # the sequence's index in the block table
    first: object  # its first flat row
    last: object  # one past its last flat row
    ends: list  # per tile row, its exclusive key bound (0 past the sequence's rows)
    max_end: object
    spec: object = None  # the first step's table entry for each chunk size, read before the size is known


@flyc.jit
def score_chunk(sc, rows, cmax, num_waves, cb, cc, field, base, hook=None):
    """Scores keys ``[cc * cb, (cc + 1) * cb)`` (``cb`` <= ``cmax``, a multiple of 64) of the tile's ``rows`` rows (flat
    rows ``first ..``) into ``field.ptr[base + r * cmax + key]``, ``key`` the position in the chunk; keys of a step
    past ``cb`` are not written. Every wave of the workgroup takes part."""
    wave, lane = sc.wave, sc.lane
    if hook is not None:
        hook(8)
    l = lane & 15
    g = (lane >> 4) & 3
    steps = max(1, cmax // 64 // num_waves)
    q_r = buffer_ops.create_buffer_resource_from_addr(sc.q)
    qs_r = buffer_ops.create_buffer_resource_from_addr(sc.qs)
    w_r = buffer_ops.create_buffer_resource_from_addr(sc.w)
    bt_r = buffer_ops.create_buffer_resource_from_addr(sc.bt)
    mfma = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float4E2M1FN, Float4E2M1FN, opsel_a=0, opsel_b=0))
    q_ops, q_sc, w_l = [], [], []
    for r in range_constexpr(rows):
        row = fx.min(sc.first + r, sc.last - 1)
        ops = []
        for mi in range_constexpr(2):
            frag = fx.make_rmem_tensor(4, I)
            frag.store(fx.Vector(buffer_ops.buffer_load(q_r, (row * HEADS + mi * 16 + l) * 16 + g * 4, vec_width=4, dtype=I)))
            ops.append(frag)
        q_ops.append(ops)
        dw = I(buffer_ops.buffer_load(qs_r, (row * 4 + g) * 16 + l, vec_width=1, dtype=I))
        q_sc.append([dw >> I(8 * mi) for mi in range(2)])
        wl = []
        for mi in range_constexpr(2):
            w4 = fx.Vector(buffer_ops.buffer_load(w_r, row * HEADS + mi * 16 + g * 4, vec_width=4, dtype=I)).bitcast(F)
            wl.append([w4[e] for e in range(4)])
        w_l.append(wl)
    nst = (fx.max(fx.min(sc.max_end - cc * cb, cb), I(0)) + 63) // 64
    step0 = cc * (cb // 64)
    if hook is not None:
        fx.rocdl.s_waitcnt(vmcnt=0)
        hook(9)

    def load_keys(k):
        st = fx.min(step0 + wave + k * num_waves, step0 + fx.max(nst - 1, I(0)))
        if fx.const_expr(k == 0 and sc.spec is not None):
            page = sc.spec[-1]
            for size, entry in reversed(list(zip(SPEC_SIZES, sc.spec))[:-1]):
                page = (cb == I(size)).select(entry, page)
        else:
            page = I(buffer_ops.buffer_load(bt_r, sc.seq * sc.bt_stride + st, vec_width=1, dtype=I))
        phys = I(rocdl.readfirstlane(T.i32, fx.min(fx.max(page, I(0)), sc.last_page)))
        kvb = sc.kv + I64(phys) * I64(sc.page_stride)
        kvsb = sc.kvs + I64(phys) * I64(sc.page_stride)
        kvs = fx.Vector(global_i32(kvsb + I64(g * 64 + l * 4), 1))[0]
        if fx.const_expr(hook is not None and k == 0):
            hook(10)
        kv = [global_i32(kvb + I64(g * 1024 + l * 16 + nt * 256), 4, True) for nt in range(4)]
        return kv, kvs

    def compute(k, kv, kvs):
        if fx.const_expr(hook is not None and k == 0):
            fx.rocdl.s_waitcnt(vmcnt=0)
            hook(11)
        per_tile = []
        for nt in range_constexpr(4):
            bfrag = fx.make_rmem_tensor(4, I)
            bfrag.store(fx.Vector(kv[nt]))
            ks = (I(kvs) >> I(8 * nt)) & 0xFF
            sums = []
            for r in range_constexpr(rows):
                total = None
                for mi in range_constexpr(2):
                    acc = fx.make_rmem_tensor(4, F)
                    acc.store(fx.full(4, 0.0, F))
                    fx.gemm(mfma, acc, q_ops[r][mi], bfrag, acc, scale_a=q_sc[r][mi], scale_b=ks)
                    logit = fx.Vector(acc.load())
                    for e in range_constexpr(4):
                        t = relu2(logit[e])
                        wv = w_l[r][mi][e]
                        total = vop("v_mul_f32 $0, $1, $2", t, wv) if total is None else vop("v_fma_f32 $0, $1, $2, $3", t, wv, total)
                sums.append(total)
            per_tile.append(sums)
        key = (wave + k * num_waves) * 64 + g * 16 + l
        for r in range_constexpr(rows):
            p = [per_tile[nt][r] for nt in range(4)]
            x, y = permlane_swap(rocdl.permlane32_swap, p[0], p[2])
            lo = vop("v_add_f32 $0, $1, $2", x, y)
            x, y = permlane_swap(rocdl.permlane32_swap, p[1], p[3])
            hi = vop("v_add_f32 $0, $1, $2", x, y)
            x, y = permlane_swap(rocdl.permlane16_swap, lo, hi)
            tot = vop("v_add_f32 $0, $1, $2", x, y)
            okey = ((cc * cb + key) < sc.ends[r]).select(score_key(tot.bitcast(I)), I(INT_MIN))
            store_lds(field, base + r * cmax + key, okey)
        if fx.const_expr(hook is not None and k == 0):
            hook(12)

    def fill_invalid(k):
        key = (wave + k * num_waves) * 64 + g * 16 + l
        for r in range_constexpr(rows):
            store_lds(field, base + r * cmax + key, I(INT_MIN))

    depth = 1 if rows >= 4 else 2  # key tiles loaded ahead: a deeper pipeline costs registers, and with them workgroups a CU
    bufs = [load_keys(k) for k in range(min(depth, steps))]
    for k in range_constexpr(steps):
        kv, kvs = bufs[k % depth]
        if (wave + k * num_waves) < nst:
            compute(k, kv, kvs)
        elif (wave + k * num_waves) * 64 < cb:
            fill_invalid(k)
        if k + depth < steps:
            bufs[k % depth] = load_keys(k + depth)
