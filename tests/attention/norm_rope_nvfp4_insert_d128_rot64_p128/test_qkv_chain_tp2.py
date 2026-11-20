# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``qkv.mxfp8_gemm_qkv_norm_rope_nvfp4_insert`` at the TP2 QKV shape (N 4992 = (32 + 2 * 2 + 2 + 1) * 128, K 6144)
over 1-4096 rows, shuffled-code and row-major: bit for bit the explicit calls (the GEMM's deferred split-K slab into
``splitk_reduce_norm_rope_nvfp4_insert``, or its bf16 rows into ``norm_rope_nvfp4_insert``), the rows the one-launch
entry ``mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128`` takes included."""

import functools

import pytest
import torch
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
    norm_rope_nvfp4_insert,
    splitk_reduce_norm_rope_nvfp4_insert,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import qkv as chain
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8

from .test_norm_rope_insert import TP2, Batch, assert_bitwise, prefill_case, require_gfx950

N, K = 4992, 6144
KW = dict(num_heads=32, num_kv_heads=2, num_index_heads=2, eps=1e-6)
ROWS = [1, 8, 16, 17, 32, 64, 65, 128, 129, 192, 256, 288, 289, 369, 401, 512, 513, 1024, 1537, 2048, 4096]


@functools.cache
def weights():
    gen = torch.Generator(device="cuda").manual_seed(39)
    w = (torch.randn(N, K, device="cuda", generator=gen) / 64).to(torch.bfloat16)
    return g.preshuffle_mxfp8_weight(*quantize_mxfp8(w))


def codes_for(m, seed):
    """The producer's codes for ``m`` rows: the GEMM row's tile, or row-major codes where it is 0."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, K, device="cuda", generator=gen).to(torch.bfloat16)
    tile = g.prequant_code_tile_rows(m, N, K, torch.bfloat16)
    codes, scales = g.mxfp8_e4m3_quantize_shuffled(x, tile or 16)
    if not tile:
        codes = quantize_mxfp8(x)[0].contiguous()
    return codes, scales


def insert_args(m, seed):
    """Decode rows as requests of 8 tokens (DSpark), 512 and more as three prefill requests."""
    if m >= 512:
        case = prefill_case(m, 3)
    else:
        lens = [8] * (m // 8) + ([m % 8] if m % 8 else [])
        case = ([1000 + 211 * r for r in range(len(lens))], lens, 0)
    batch = Batch(*case, TP2, (0.37, 0.61), seed=seed, index_skip=0.01)
    return batch, (*batch.w, batch.cs, batch.pos, batch.slots, batch.islots)


def explicit(codes, scales, m, batch, ins):
    w, ws = weights()
    q, iq, kv, idx = batch.outputs()
    args = (*ins, batch.kv(kv), batch.idx(idx), batch.ks, batch.vs)
    if g.prequant_code_tile_rows(m, N, K, torch.bfloat16):
        out = g.mxfp8_gemm_shuffled(codes, scales, w, ws, torch.bfloat16, m, defer_reduce=True)
    else:
        out = g.mxfp8_gemm(codes, scales, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    if isinstance(out, g.SkPartials):
        rows = out.tmp.shape[1]
        q, iq = q.new_empty(rows, q.shape[1]), iq.new_empty(rows, iq.shape[1])
        splitk_reduce_norm_rope_nvfp4_insert(out.tmp, out.split_k, *args, q, iq, **KW)
    else:
        norm_rope_nvfp4_insert(out.reshape(m, N), *args, q, iq, **KW)
    return q, iq, kv, idx


def entry(codes, scales, m, batch, ins):
    w, ws = weights()
    _, _, kv, idx = batch.outputs()
    args = (*ins, batch.kv(kv), batch.idx(idx), batch.ks, batch.vs)
    q, iq = chain.mxfp8_gemm_qkv_norm_rope_nvfp4_insert(codes, scales, w, ws, m, torch.bfloat16, *args, **KW)
    return q, iq, kv, idx


@pytest.mark.parametrize("m", ROWS)
@torch.inference_mode()
def test_tp2_chain_bitwise_explicit_calls(m):
    require_gfx950()
    codes, scales = codes_for(m, 7 * m + 1)
    batch, ins = insert_args(m, m)
    a = explicit(codes, scales, m, batch, ins)
    b = entry(codes, scales, m, batch, ins)
    torch.cuda.synchronize()
    assert a[0].shape == b[0].shape == (a[0].shape[0], 4096) and b[1].shape[1] == 256
    assert_bitwise(a, b, f"tp2 chain m{m}")


@torch.inference_mode()
def test_tp2_zero_rows():
    require_gfx950()
    codes, scales = codes_for(8, 5)
    batch, ins = insert_args(8, 5)
    w, ws = weights()
    q, iq = chain.mxfp8_gemm_qkv_norm_rope_nvfp4_insert(
        codes[:0], scales, w, ws, 0, torch.bfloat16, *ins, batch.kv(batch.kv_full.clone()),
        batch.idx(batch.idx_full.clone()), batch.ks, batch.vs, **KW)  # fmt: skip
    assert q.shape == (0, 4096) and iq.shape == (0, 256)
