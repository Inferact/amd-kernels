# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Merge of the split-KV partials of ``kernel.py``: one wave per (request, query row, head)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import GTensor
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.warm_up import launcher

from .kernel import HEADS, KV_LORA_RANK, NEG, _readlane, request_layout

LN2 = 0.6931471805599453


@launcher
def compile_merge(qlen: int):
    name = f"fp8_dense_mla_decode_h96_kv512_merge_q{qlen}"

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kern(
        PO: fx.Tensor,
        PML: fx.Tensor,
        IPTR: fx.Tensor,
        KSC: fx.Tensor,
        OUT: fx.Tensor,
        LSE: fx.Tensor,
        stride_ot: fx.Int32,
        stride_oh: fx.Int32,
        stride_lt: fx.Int32,
        n_req: fx.Int32,
        nsets: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        w = tid // 64
        lane = tid % 64
        row = fx.Int32(fx.block_idx.x) * 4 + w
        per_req = qlen * HEADS
        b = row // per_req
        rem = row - b * per_req
        t = rem // HEADS
        head = rem - t * HEADS

        po_t = GTensor(PO, dtype=T.i32, shape=(-1,))
        pml_t = GTensor(PML, dtype=T.i32, shape=(-1,))
        ip_t = GTensor(IPTR, dtype=T.i32, shape=(-1,))
        sc_t = GTensor(KSC, dtype=T.f32, shape=(-1,))
        o_t = GTensor(OUT, dtype=T.i32, shape=(-1,))
        lse_t = GTensor(LSE, dtype=T.i32, shape=(-1,))

        chunk, vs_l, ve_l, sf_l, cnt_l, lo_l = request_layout(ip_t, n_req, nsets, lane)
        lo = _readlane(lo_l, b)
        nb = _readlane(cnt_l, b)

        kscale = fx.Float32(sc_t[0])
        init = [_raw(fx.Float32(NEG).ir_value()), _raw(fx.Float32(0.0).ir_value())]
        init += [_raw(Vec.filled(4, 0.0, fx.Float32).ir_value()) for _ in range(2)]
        for s, st in range(0, nb, 1, init=init):
            mm_run = fx.Float32(st[0])
            ws_run = fx.Float32(st[1])
            a0 = Vec(st[2])
            a1 = Vec(st[3])
            slot = lo + fx.Int32(s)
            ml = ((slot * qlen + t) * HEADS + head) * 2
            mm_s = fx.Int32(pml_t[ml]).bitcast(fx.Float32)
            l_s = fx.Int32(pml_t[ml + 1]).bitcast(fx.Float32)
            base_o = (((slot * qlen + t) * HEADS + head) * KV_LORA_RANK) // 2
            ob = Vec(po_t.load(base_o + lane * 4, vec_size=4)).bitcast(fx.BFloat16)
            of = Vec(ob).to(fx.Float32)
            o0 = Vec.from_elements([fx.Float32(of[j]) for j in range(4)], fx.Float32)
            o1 = Vec.from_elements([fx.Float32(of[4 + j]) for j in range(4)], fx.Float32)
            valid = l_s > fx.Float32(0.0)
            take = valid & (mm_s > mm_run)
            m_new = take.select(mm_s, mm_run)
            f_old = (mm_run - m_new).exp2(fastmath="afn")
            f_new = valid.select((mm_s - m_new).exp2(fastmath="afn"), fx.Float32(0.0))
            ws_new = ws_run * f_old + l_s * f_new
            n0 = Vec.from_elements([fx.Float32(a0[j]) * f_old + fx.Float32(o0[j]) * f_new for j in range(4)], fx.Float32)
            n1 = Vec.from_elements([fx.Float32(a1[j]) * f_old + fx.Float32(o1[j]) * f_new for j in range(4)], fx.Float32)
            fin = yield [m_new, ws_new, n0, n1]

        m_f = fx.Float32(fin[0])
        ws_f = fx.Float32(fin[1])
        inv = (ws_f > fx.Float32(0.0)).select(kscale / ws_f, fx.Float32(0.0))
        vals = [fx.Float32(Vec(fin[2])[j]) * inv for j in range(4)] + [fx.Float32(Vec(fin[3])[j]) * inv for j in range(4)]
        bf = Vec.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        tok = b * qlen + t
        o_t.store((tok * stride_ot + head * stride_oh) // 2 + lane * 4, bf, vec_size=4)
        lse = (ws_f > fx.Float32(0.0)).select(
            (m_f + fx.Float32(fx.log2(ws_f))) * LN2, fx.Float32(float("-inf"))
        )
        lse_t.store(tok * stride_lt + head, lse.bitcast(fx.Int32))

    @flyc.jit
    def launch(
        PO,
        PML,
        IPTR,
        KSC,
        OUT,
        LSE,
        stride_ot,
        stride_oh,
        stride_lt,
        n_req,
        nsets,
        grid_x,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kern(PO, PML, IPTR, KSC, OUT, LSE, stride_ot, stride_oh, stride_lt, n_req, nsets).launch(
            grid=(fx.Int64(grid_x), 1, 1), block=(256, 1, 1), stream=stream
        )

    launch.compile_hints = {"waves_per_eu": 4, "fast_fp_math": False}
    return launch
