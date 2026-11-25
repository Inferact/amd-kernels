# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# _dcp_a2a_unpack_combine_kernel of Inferact/moo vllm/v1/attention/ops/dcp.py at a3b845e226, its base-2, packed-LSE
# (LSE_PACK_DIM=2), no-LSE-output case: the combine the kernel matches bit for bit, for CI without vLLM.
"""vLLM's DCP a2a unpack-combine in Triton, base-2 LSEs packed as two bf16 words."""
import triton
import triton.language as tl


@triton.jit
def unpack_combine_kernel(
    recv_ptr,
    out_ptr,
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_max = -float("inf")
    for rank_idx in tl.static_range(N):
        recv_base = rank_idx * recv_stride_N + batch_idx * recv_stride_B + head_idx * recv_stride_H
        lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
        hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
        lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where((lse_val != lse_val) | (lse_val == float("inf")), -float("inf"), lse_val)
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    lse_sum = 0.0
    for rank_idx in tl.static_range(N):
        recv_base = rank_idx * recv_stride_N + batch_idx * recv_stride_B + head_idx * recv_stride_H
        lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
        hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
        lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where((lse_val != lse_val) | (lse_val == float("inf")), -float("inf"), lse_val)
        lse_sum += tl.exp2(lse_val - lse_max)

    global_lse = tl.log2(lse_sum) + lse_max

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for rank_idx in tl.static_range(N):
        recv_base = rank_idx * recv_stride_N + batch_idx * recv_stride_B + head_idx * recv_stride_H
        lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
        hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
        lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where((lse_val != lse_val) | (lse_val == float("inf")), -float("inf"), lse_val)
        weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)
        partial = tl.load(recv_ptr + recv_base + d_offsets * recv_stride_D).to(tl.float32)
        partial = tl.where(weight == 0.0, 0.0, partial)
        acc += partial * weight

    final_offsets = batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    tl.store(out_ptr + final_offsets, acc)
