# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Router/quant placement retains partials, routing, quant bytes and replay writes."""
import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm as rg
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_gemm_topk_mxfp4_mxfp8_quant import router_gemm_topk_mxfp4_mxfp8_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_topk_mxfp4_mxfp8_quant import splitk_biased_topk_mxfp4_mxfp8_quant
from .workload import make_activation, require_gfx950


@pytest.mark.parametrize("m", [58, 59, 61, 63, 64, 65, 66, 80, 97, 127, 128, 193, 194, 223, 224, 255, 256])
@pytest.mark.parametrize("case", ["random", "ties", "skew", "no_choices"])
def test_router_quant_tasks_bytes_and_graph(m, case, monkeypatch):
    require_gfx950()
    torch.manual_seed(m)
    x = make_activation(m, seed=m)
    gate = torch.randn((128, 6144), device="cuda") / 6144**0.5
    bias = torch.randn(128, device="cuda") * 0.01
    if case == "ties":
        gate.zero_()
        bias.zero_()
    if case == "skew":
        bias.zero_()
        bias[[0, 31, 63, 127]] = 100.0

    if case == "no_choices":
        bias.fill_(float("-inf"))

    def reference():
        p = rg.router_gemm_partials(x, gate)
        ids = torch.empty((m, 4), device="cuda", dtype=torch.int32)
        wt = torch.empty((m, 4), device="cuda")
        q4, q8 = splitk_biased_topk_mxfp4_mxfp8_quant(p, bias, wt, ids, 2.0, x)
        return [*q4, *q8, wt, ids, p]

    def run():
        q4, q8, wt, ids, p = router_gemm_topk_mxfp4_mxfp8_quant(x, gate, bias, 2.0)
        return [*q4, *q8, wt, ids, p]

    expected = reference()
    original = torch.empty
    guards = []

    def alloc(*shape, **kw):
        shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        total = 1
        for n in shape:
            total *= n
        t = torch.full((total + 128,), 7, **kw)
        guards.append(t)
        return t[64:-64].view(shape)

    monkeypatch.setattr(torch, "empty", alloc)
    actual = run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    monkeypatch.setattr(torch, "empty", original)
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual, expected))
    for seed in (m, m + 1, m):
        x.copy_(make_activation(m, seed=seed))
        expected = reference()
        for t in out:
            t.view(torch.uint8).fill_(0xA5)
        graph.replay()
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, expected))
        assert all((t[:64] == 7).all() and (t[-64:] == 7).all() for t in guards)


def test_quant_tiers():
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm_topk_mxfp4_mxfp8_quant as q
    placement = [q.router_quant_placement(m) for m in range(1, 300)]
    assert placement == [0] * 57 + [1] * 7 + [2] * 64 + [0] * 64 + [1] * 64 + [0] * 43
    assert all(rg.nblocks_config(m) is not None for m in range(1, 300) if q.router_dual_quant_for(m))
