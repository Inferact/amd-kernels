# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The ``wkv`` chain on one GPU: ``NgramRows`` output (codes in the layout the table's row for the token count wants,
``gemm.prequant_code_tile_rows``, + preshuffled scales) straight into ``gemm.mxfp8_gemm_prequant`` (N 25600, K 6144, no
quantizer launch, no host shuffle), then ``gated_post``. The GEMM meets the
MXFP8 GEMM contract (exact dequant of both operands, fp32 sums) against the rows' dequantized values, and differs from
vLLM's path (the BF16 rows re-quantized per 1 x 32 block, then the GEMM) only where that re-quantization picks another
scale."""

import pytest
import torch
from amd_kernels.engram import ngram_rows_mxfp8_d256_c24_ranks8 as pkg
from amd_kernels.engram.gated_post_bf16_h5120_hc4 import gated_post, gated_post_reference
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.reference import COS_MIN, REL_L2_MAX, compare, dequantize_mxfp8, quantize_mxfp8

from . import workload as wl
from .test_ngram_rows import expected, setup  # noqa: F401

N, K = 25600, 6144


@pytest.fixture(scope="module")
def wkv():
    torch.manual_seed(3)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    codes, exps = quantize_mxfp8(w)
    return codes, exps, g.preshuffle_mxfp8_weight(codes, exps)


@pytest.mark.parametrize("m,requests", [(16, 2), (40, 6), (130, 21), (192, 32), (384, 64), (768, 128)])
def test_rows_into_gemm_into_post(setup, wkv, m, requests):  # noqa: F811
    rows = setup[0]
    codes_w, exps_w, (w_sh, s_sh) = wkv
    ids, pos, qsl, lb, _ = wl.decode_batch(requests, 6, m, 40 + m, slots=160)
    ins = [t.cuda() for t in (ids, pos, qsl, lb)]
    tile = g.prequant_code_tile_rows(m, N, K)
    rows(*ins, tile)
    idx, codes, _ = expected(setup, ids, pos, qsl, lb)
    tables = setup[1]
    w_dq = dequantize_mxfp8(codes_w, exps_w)
    for layer in range(2):
        act, act_scale = rows.activation(layer, m, tile)
        kv = g.mxfp8_gemm_prequant(act, act_scale, w_sh, s_sh, torch.bfloat16, m)
        sc = tables[layer][1].cuda()[idx[layer].flatten()].view(m, 192)
        x = dequantize_mxfp8(codes[layer].view(torch.float8_e4m3fn), sc)
        metrics = compare(kv, (x @ w_dq.T).to(torch.bfloat16))
        assert metrics["cos"] >= COS_MIN and metrics["rel_l2"] <= REL_L2_MAX, (layer, m, metrics)
        requant = g.mxfp8_linear(x.bfloat16(), w_sh, s_sh)
        assert compare(kv, requant)["cos"] >= COS_MIN
        h = torch.randn(m, 4, 5120, device="cuda").bfloat16()
        q, k = (torch.randn(4, 5120, device="cuda").bfloat16() * 0.3 for _ in range(2))
        out = gated_post(h, kv, q, k)
        ref = gated_post_reference(h, kv, q, k)
        assert torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0) >= 0.99999
