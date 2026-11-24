# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch

from amd_kernels.moe import prefill_top16_mxfp4


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("rows,valid", [(0, 0), (1, 0), (513, 511), (1024, 1024), (4096, 4093)])
def test_prefill(packed, rows, valid):
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")
    from aiter import biased_grouped_topk, dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    rng = torch.Generator(device="cuda").manual_seed(52)
    if packed:
        merged = torch.randn(rows, 4480, device="cuda", generator=rng)
        logits, latent = merged.split((896, 3584), dim=-1)
    else:
        logits = torch.randn(rows, 896, device="cuda", generator=rng)
        latent = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16, generator=rng)
    bias = torch.zeros(896, device="cuda")
    logits[::8] = 0
    latent[::19] = 0
    weights, ids, codes, scales = prefill_top16_mxfp4(logits, latent, bias, 0.75, valid)
    assert (weights.shape, ids.shape, codes.shape, scales.shape) == ((rows,16),(rows,16),(rows,1792),(rows,112))
    assert torch.all(ids[valid:] == -1) and torch.all(weights[valid:] == 0)
    if valid:
        ew = torch.empty_like(weights[:valid])
        ei = torch.empty_like(ids[:valid])
        biased_grouped_topk(logits[:valid].contiguous(), bias, ew, ei, 1, 1, True, .75)
        torch.testing.assert_close(ids[:valid], ei, rtol=0, atol=0)
        torch.testing.assert_close(weights[:valid], ew, rtol=0, atol=0)
    if rows:
        q,s = per_1x32_mx_quant_hip(latent.bfloat16().contiguous(), quant_dtype=dtypes.fp4x2, shuffle=False)
        torch.testing.assert_close(codes.view(torch.uint8), q.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(scales, s.view(torch.uint8), rtol=0, atol=0)



def test_prefill_graph_replay():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")
    logits = torch.randn(1024, 896, device="cuda")
    latent = torch.randn(1024, 3584, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(896, device="cuda")
    prefill_top16_mxfp4(logits, latent, bias)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = prefill_top16_mxfp4(logits, latent, bias)
    logits.normal_()
    latent.normal_()
    expected = prefill_top16_mxfp4(logits, latent, bias)
    graph.replay()
    for a, b in zip(actual, expected):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
