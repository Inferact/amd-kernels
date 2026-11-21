# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Floor of the prologue: the same grid, table lookups, band / gate loads and record stores, no math."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.attention.kda_prefill_h12_d128 import prologue as P
from amd_kernels.common import buffer_ops
from amd_kernels.common.tensor_shim import _run_compiled
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec


def build():
    @flyc.kernel(name="kda_prefill_prologue_skeleton", known_block_size=[256, 1, 1])
    def kernel(arg_mixed: fx.Int64, i32_ldx: fx.Int32, i32_mixed_bytes: fx.Int32, arg_rawg: fx.Int64, i32_ldg: fx.Int32,
               arg_cu: fx.Int64, arg_ci: fx.Int64, arg_ws: fx.Int64, i32_nc: fx.Int32):
        tx = fx.Int32(fx.thread_idx.x)
        cid = fx.Int32(fx.block_idx.x)
        h = fx.Int32(fx.block_idx.y)
        ci_r = buffer_ops.create_buffer_resource_from_addr(arg_ci, num_records_bytes=fx.Int32(0x7FFFFFF0))
        cu_r = buffer_ops.create_buffer_resource_from_addr(arg_cu, num_records_bytes=fx.Int32(0x7FFFFFF0))
        n_seq = fx.Int32(buffer_ops.buffer_load(ci_r, cid * 2, vec_width=1, is_scalar=True))
        ci_loc = fx.Int32(buffer_ops.buffer_load(ci_r, cid * 2 + 1, vec_width=1, is_scalar=True))
        bos = fx.Int32(buffer_ops.buffer_load(cu_r, n_seq, vec_width=1, is_scalar=True))
        tn = fx.Int32(buffer_ops.buffer_load(cu_r, n_seq + 1, vec_width=1, is_scalar=True)) - bos
        t0 = ci_loc * 64
        mr = buffer_ops.create_buffer_resource_from_addr(arg_mixed, num_records_bytes=i32_mixed_bytes)
        gr = buffer_ops.create_buffer_resource_from_addr(arg_rawg, num_records_bytes=fx.Int32(0x7FFFFFF0))
        wsr = buffer_ops.create_buffer_resource_from_addr(
            arg_ws + fx.Int64(h * i32_nc + cid) * fx.Int64(P.RECORD_BYTES), num_records_bytes=P.RECORD_BYTES)
        row = tx // 4
        part = tx % 4
        acc = fx.Int32(0)
        for band in range_constexpr(3):
            for k in range_constexpr(2):
                s = t0 + row
                v = Vec(buffer_ops.buffer_load(mr, ((bos + s) * i32_ldx + band * P.LP + h * P.D + part * 32 + k * 8) // 2, vec_width=4,
                                               dtype=fx.Int32, mask=s < tn))
                acc = acc ^ v[0] ^ v[1] ^ v[2] ^ v[3]
        for k in range_constexpr(2):
            s = t0 + row
            v = Vec(buffer_ops.buffer_load(gr, ((bos + s) * i32_ldg + h * P.D + part * 32 + k * 8) // 2, vec_width=4, dtype=fx.Int32,
                                           mask=s < tn))
            acc = acc ^ v[0] ^ v[1] ^ v[2] ^ v[3]
        for k in range_constexpr(P.RECORD_BYTES // 4096):
            buffer_ops.buffer_store(Vec.from_elements([acc] * 4, fx.Int32), wsr, (k * 4096 + tx * 16) // 4)
        buffer_ops.buffer_store(Vec.from_elements([acc] * 4, fx.Int32), wsr, (18 * 4096 + (tx % 128) * 16) // 4, mask=tx < 128)

    @flyc.jit
    def launch(arg_mixed: fx.Int64, i32_ldx: fx.Int32, i32_mixed_bytes: fx.Int32, arg_rawg: fx.Int64, i32_ldg: fx.Int32,
               arg_cu: fx.Int64, arg_ci: fx.Int64, arg_ws: fx.Int64, i32_nc: fx.Int32, stream: fx.Stream):
        kernel(arg_mixed, i32_ldx, i32_mixed_bytes, arg_rawg, i32_ldg, arg_cu, arg_ci, arg_ws, i32_nc).launch(
            grid=(fx.Int64(i32_nc), P.NH, 1), block=(256, 1, 1), stream=stream)

    return launch


_launch = None


def skeleton_prologue(mixed, raw_g, beta_logit, A_log, dt_bias, lower_bound, scale, conv_weight, conv_state, state_indices,
                      has_initial_state, cu_seqlens, chunk_indices, ws, table=None):
    global _launch
    if _launch is None:
        _launch = build()
    _run_compiled(_launch, mixed.data_ptr(), mixed.stride(0), mixed.shape[0] * mixed.stride(0) * 2, raw_g.data_ptr(), raw_g.stride(0),
                  cu_seqlens.data_ptr(), chunk_indices.data_ptr(), ws.data_ptr(), chunk_indices.shape[0], torch.cuda.current_stream())
    return ws
