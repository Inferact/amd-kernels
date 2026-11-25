# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One wave selects a row's experts and seven other waves pack its latent activation."""
import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops as buf
from amd_kernels.warm_up import launcher
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .host import _bits, _run
from .topk import (LDS_WORDS, MAXC, NQ, RANK_UNROLL, topk_fast_write, topk_finish, topk_new, topk_pivot,
                   topk_q_scores, topk_rank_start, topk_rank_step, topk_slow_out, topk_slow_select, topk_tie_masks)

E, H, TOPK = 896, 3584, 16


def _quant_wave(latent, codes, scales, row, wave, lane, stride, fp32):
    group = wave * 16 + (lane >> 2)
    start = group * 32 + (lane & 3) * 8
    if const_expr(fp32):
        x = []
        for k in range_constexpr(2):
            v = Vec(buf.buffer_load(latent, row * stride + start + k * 4, vec_width=4, dtype=fx.Float32))
            x.extend([fx.Float32(v[j]).to(fx.BFloat16).to(fx.Float32) for j in range_constexpr(4)])
    else:
        v = Vec(buf.buffer_load(latent, (row * stride + start) // 2, vec_width=4, dtype=fx.Int32)).bitcast(fx.BFloat16)
        x = [fx.BFloat16(v[j]).to(fx.Float32) for j in range_constexpr(8)]
    peak = fx.Float32(1e-10)
    for j in range_constexpr(8):
        peak = fx.max(peak, fx.max(x[j], -x[j]))
    for delta in (1, 2):
        peer = fx.Int32(rocdl.ds_bpermute(T.i32, (lane ^ delta) * 4, peak.bitcast(fx.Int32))).bitcast(fx.Float32)
        peak = fx.max(peak, peer)
    bits = (peak * fx.Float32(1.0 / 6.0)).bitcast(fx.Uint32)
    exp = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).select(fx.Uint32(1), fx.Uint32(0))
    buf.buffer_store(exp.to(fx.Uint8), scales, row * (H // 32) + group, mask=(lane & 3) == 0)
    dequant = (exp << 23).bitcast(fx.Float32)
    packed = fx.Int32(0)
    for pair in range_constexpr(4):
        packed = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed, x[2*pair], x[2*pair+1], dequant, pair))
    buf.buffer_store(packed, codes, row * (H // 8) + start // 8)


@functools.lru_cache(maxsize=2)
@launcher
def compile_prefill(fp32: bool):
    @fx.struct
    class Shared:
        ent: fx.Array[fx.Int32, LDS_WORDS["ent"], 16]
        osig: fx.Array[fx.Float32, LDS_WORDS["osig"], 16]
        oid: fx.Array[fx.Int32, LDS_WORDS["oid"], 16]
        tie: fx.Array[fx.Int32, LDS_WORDS["tie"], 16]
        tex: fx.Array[fx.Int32, LDS_WORDS["tex"], 16]
        tmask: fx.Array[fx.Int32, LDS_WORDS["tmask"], 16]

    name = f"router_top16_mxfp4_quant_e896_h3584_f32{int(fp32)}"

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(g: fx.Int64, l: fx.Int64, b: fx.Int64, w: fx.Int64, ids: fx.Int64, q: fx.Int64, s: fx.Int64,
               rows: fx.Int32, valid: fx.Int32, gs: fx.Int32, ls: fx.Int32, scale: fx.Int32):
        lane, piece, row = fx.thread_idx.x, fx.block_idx.x, fx.block_idx.y
        if piece == 0:
            smem = fx.SharedAllocator().allocate(Shared).peek()
            out_bytes = fx.Int64(rows) * (TOPK * 4)
            wr = buf.create_buffer_resource_from_addr(w, num_records_bytes=out_bytes)
            ir = buf.create_buffer_resource_from_addr(ids, num_records_bytes=out_bytes)
            if row < valid:
                gr = buf.create_buffer_resource_from_addr(g + fx.Int64(row) * fx.Int64(gs) * 4, num_records_bytes=E * 4)
                br = buf.create_buffer_resource_from_addr(b, num_records_bytes=E * 4)
                st = topk_new(lane)
                for k in range_constexpr(NQ):
                    for sg, key in topk_q_scores(gr, br, lane, fx.Int32(0), fx.Int32(1), 1, 0, k):
                        st.sig.append(sg)
                        st.key.append(key)
                topk_pivot(st, smem, 0, lane)
                if st.nc <= MAXC:
                    topk_fast_write(st, smem, 0, lane)
                    topk_rank_start(st, smem, 0, lane)
                    rank = fx.Int32(0)
                    e = fx.Int32(0)
                    while e < st.nc:
                        rank = topk_rank_step(rank, e, st, smem, 0)
                        e = e + RANK_UNROLL
                    topk_finish(rank, lane < st.nc, st.mysig, st.myid, lane, smem, 0, wr, ir, row, fx.Int32(TOPK), scale)
                else:
                    ss = topk_slow_select(st)
                    if ss.total_ties != ss.need:
                        topk_tie_masks(ss, smem, 0, lane)
                    rank, mysig, myid = topk_slow_out(ss, smem, 0, lane)
                    topk_finish(rank, lane < TOPK, mysig, myid, lane, smem, 0, wr, ir, row, fx.Int32(TOPK), scale)
            else:
                buf.buffer_store(fx.Float32(0.), wr, row * TOPK + lane, mask=lane < TOPK)
                buf.buffer_store(fx.Int32(-1), ir, row * TOPK + lane, mask=lane < TOPK)
        else:
            element_bytes = 4 if fp32 else 2
            extent = ((fx.Int64(rows) - 1) * fx.Int64(ls) + H) * element_bytes
            lr = buf.create_buffer_resource_from_addr(l, num_records_bytes=extent)
            qr = buf.create_buffer_resource_from_addr(q, num_records_bytes=fx.Int64(rows) * (H // 2))
            sr = buf.create_buffer_resource_from_addr(s, num_records_bytes=fx.Int64(rows) * (H // 32))
            _quant_wave(lr, qr, sr, row, piece - 1, lane, ls, fp32)

    @flyc.jit
    def launch(g: fx.Int64, l: fx.Int64, b: fx.Int64, w: fx.Int64, ids: fx.Int64, q: fx.Int64, s: fx.Int64,
               rows: fx.Int32, valid: fx.Int32, gs: fx.Int32, ls: fx.Int32, scale: fx.Int32, stream: fx.Stream):
        kernel(g,l,b,w,ids,q,s,rows,valid,gs,ls,scale).launch(grid=(8,rows,1), block=(64,1,1), stream=stream)

    launch.kernel_name = name
    return launch


def prefill_top16_mxfp4(logits, latent, bias, routed_scaling_factor=1.0, valid_rows=None):
    """Return FP32 weights, int32 physical IDs, packed FP4 rows and row-major uint8 E8M0 scales."""
    if logits.ndim != 2 or logits.shape[1] != E or logits.dtype != torch.float32:
        raise ValueError("logits must be FP32 [rows, 896]")
    rows = logits.shape[0]
    if latent.shape != (rows, H) or latent.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("latent must be BF16 or FP32 [rows, 3584]")
    if bias.shape != (E,) or bias.dtype != torch.float32 or not bias.is_contiguous():
        raise ValueError("bias must be contiguous FP32 [896]")
    if not logits.is_cuda or latent.device != logits.device or bias.device != logits.device:
        raise ValueError("all inputs must be on the same GPU")
    valid = rows if valid_rows is None else valid_rows
    if not isinstance(valid, int) or not 0 <= valid <= rows <= 4096:
        raise ValueError("require 0 <= valid_rows <= rows <= 4096")
    for x in (logits, latent):
        if x.stride(1) != 1 or x.stride(0) < x.shape[1] or (x.stride(0)*x.element_size()) % 16 or x.data_ptr() % 16:
            raise ValueError("column stride must be one and rows must be 16-byte aligned and nonoverlapping")
        if rows and ((rows - 1) * x.stride(0) + x.shape[1]) * x.element_size() > 2**31 - 1:
            raise ValueError("input exceeds the buffer addressing range")
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")
    weights = logits.new_empty((rows, TOPK))
    ids = weights.new_empty(weights.shape, dtype=torch.int32)
    codes = ids.new_empty((rows, H // 2), dtype=torch.float4_e2m1fn_x2)
    scales = ids.new_empty((rows, H // 32), dtype=torch.uint8)
    if rows:
        with torch.cuda.device(logits.device):
            _run(compile_prefill(latent.dtype == torch.float32), logits.data_ptr(), latent.data_ptr(), bias.data_ptr(),
                 weights.data_ptr(), ids.data_ptr(), codes.data_ptr(), scales.data_ptr(), rows, valid, logits.stride(0),
                 latent.stride(0), _bits(routed_scaling_factor), torch.cuda.current_stream())
    return weights, ids, codes, scales
