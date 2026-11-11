# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two-expert loads restore the four-expert argmax layout, including its ties."""

import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_topk as rt
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_topk_vec2 import compile_router_topk_vec2
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import (
    WEIGHT_RTOL, biased_topk_reference, decided, splitk_logits_reference,
)
from .workload import require_gfx950


@pytest.fixture(autouse=True)
def _gpu():
    require_gfx950()


def _run(fn, p, bias, wt, ids, scale_bits=0x40000000):
    m, s, e = p.shape
    end = (m - 1) * p.stride(0) + (s - 1) * p.stride(1) + e
    rt._run_compiled(fn, p.data_ptr(), bias.data_ptr(), wt.data_ptr(), ids.data_ptr(),
                     end * 4, m, p.stride(0), p.stride(1), ids.stride(0), scale_bits,
                     torch.cuda.current_stream())


# (split, tokens): S48 serves 1..64 tokens, S16 65..512 and 1025..2048, S8 2049..4096
SPLIT_M = ([(48, m) for m in (1, 2, 7, 8, 15, 16, 17, 26, 31, 32, 33, 48, 63, 64)]
           + [(16, m) for m in (1, 65, 97, 128, 129, 255, 256, 257, 511, 512, 1025, 2047, 2048)]
           + [(12, m) for m in (1, 2049, 2432, 2688)]
           + [(8, m) for m in (1, 2049, 2689, 3071, 4095, 4096)])


@pytest.mark.parametrize('split,m', SPLIT_M)
@pytest.mark.parametrize('case', ['random', 'ties', 'saturation'])
@pytest.mark.parametrize('strided', [False, True])
def test_vec2_bitwise_and_contract(split, m, case, strided):
    gen = torch.Generator(device='cuda').manual_seed(m)
    storage = torch.randn((split, m * (2 if strided else 1), 128), device='cuda', generator=gen) * 0.1
    p = storage[:, ::2 if strided else 1].permute(1, 0, 2)
    bias = torch.randn(128, device='cuda', generator=gen) * 0.01
    if case == 'ties':
        p.copy_(p[:, :, :8].repeat(1, 1, 16))
        p[0] = 0.0
        bias.zero_()
    if case == 'saturation':
        p[:, :, ::2] = 2.0
        p[:, :, 1::2] = -2.0
        bias.zero_()
    refs = (torch.empty((m, 4), device='cuda'), torch.empty((m, 4), device='cuda', dtype=torch.int32))
    _run(rt.compile_router_topk(128, 4, split), p, bias, *refs)
    # Both sides of the output view and the padding of every row are guards.
    weights = torch.full((m + 2, 8), float('nan'), device='cuda')
    ids = torch.full((m + 2, 8), -1, device='cuda', dtype=torch.int32)
    out = weights[1:-1, 2:6], ids[1:-1, 2:6]
    fn = compile_router_topk_vec2(128, 4, split)
    for _ in range(3):
        out[0].fill_(float('nan')); out[1].fill_(-1)
        _run(fn, p, bias, *out)
        assert torch.equal(out[0].view(torch.int32), refs[0].view(torch.int32))
        assert torch.equal(out[1], refs[1])
    assert torch.isfinite(out[0]).all()
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[1:-1, 2:6] = False
    assert torch.isnan(weights[mask]).all() and (ids[mask] == -1).all()
    rw, ri, vals = biased_topk_reference(splitk_logits_reference(p), bias, 4, 2.0)
    known = decided(vals)
    assert torch.equal(out[1][known].long(), ri[known])
    if case == 'random':
        # a row's weights are decided when its whole top-k set and order are
        rows = known.all(-1)
        torch.testing.assert_close(out[0][rows].double(), rw[rows], atol=0, rtol=WEIGHT_RTOL)


@pytest.mark.parametrize('split,m', [(48, 1), (48, 8), (48, 17), (48, 32), (48, 64),
                                     (16, 65), (16, 256), (16, 512), (16, 2048), (12, 2688), (8, 4096)])
def test_vec2_graph_changed_inputs(split, m):
    p = torch.randn((split, m, 128), device='cuda').permute(1, 0, 2) * 0.1
    bias = torch.zeros(128, device='cuda')
    out = torch.empty((m, 4), device='cuda'), torch.empty((m, 4), device='cuda', dtype=torch.int32)
    ref = torch.empty_like(out[0]), torch.empty_like(out[1])
    fn = compile_router_topk_vec2(128, 4, split)
    _run(fn, p, bias, *out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(fn, p, bias, *out)
    for _ in range(3):
        p.normal_(0, 0.1)
        bias.normal_(0, 0.01)
        _run(rt.compile_router_topk(128, 4, split), p, bias, *ref)
        out[0].fill_(float('nan')); out[1].fill_(-1)
        graph.replay()
        assert torch.equal(out[0].view(torch.int32), ref[0].view(torch.int32))
        assert torch.equal(out[1], ref[1])


def test_public_factory_uses_vec2_only_for_the_served_splits():
    assert '_vec2_' in rt._get_kernel(128, 4, 48).kernel_name
    assert '_vec2_' in rt._get_kernel(128, 4, 16).kernel_name
    assert '_vec2_' in rt._get_kernel(128, 4, 8).kernel_name
    for e, k, s, tpw in [(128, 4, 4, 1), (128, 4, 8, 2), (128, 4, 16, 2), (128, 4, 48, 2), (256, 4, 48, 1), (128, 2, 48, 1)]:
        assert '_vec2_' not in rt._get_kernel(e, k, s, tpw).kernel_name
