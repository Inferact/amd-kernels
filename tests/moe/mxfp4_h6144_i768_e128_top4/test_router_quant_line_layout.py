# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The fused BF16x3 router + MXFP4 quant on the line-layout gate: codes and scales against ``quant.mxfp4_quant`` bit
for bit, the router partials within the router contract, repeated and replayed bit for bit; the K split table."""
import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_mxfp4_quant_line_layout import (
    router_mxfp4_quant_line_layout,
)
from .test_entry import weights  # noqa: F401
from .workload import make_activation


# every K-split tier (router_k_split) and its first row past a bound, incl. row tails
@pytest.mark.parametrize("m", [1025, 1536, 2047, 2048, 2049, 2689, 4096, 4097, 5377,
                              8192, 8193, 10881, 16384, 16385, 21761, 24576, 24577, 32768])
def test_router_quant_line_layout(weights, m):
    _, w = weights
    x = make_activation(m, seed=m)
    x[0, :32] = 0
    expected = router_mxfp4_quant_line_layout(x, w.router_weight_lines)
    q, s, p = expected
    rq, rs = mxfp4_quant(x)
    assert torch.equal(q, rq) and torch.equal(s, rs)
    assert torch.isfinite(p).all()
    logits = sum(p[:, k] for k in range(p.shape[1]))
    reference = x.double() @ w.router_weight.double().t()
    error = (logits.double() - reference).abs()
    bound = x.double().abs() @ w.router_weight.double().abs().t()
    assert (error <= 2**-18 * bound).all(), error.max().item()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = router_mxfp4_quant_line_layout(x, w.router_weight_lines)
    for _ in range(2):
        for t in captured[:2]:
            t.fill_(255)
        captured[2].fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(captured, expected))


# the K split per row count (the table's bounds and the first row past each)
@pytest.mark.parametrize("m,ks", [(2048, 16), (2049, 12), (2688, 12), (2689, 8), (4096, 8), (4097, 6),
                                  (5376, 6), (5377, 4), (8192, 4), (8193, 3), (10880, 3), (10881, 2),
                                  (16384, 2), (16385, 3), (21760, 3), (21761, 2), (32768, 2)])
def test_router_k_split_table(m, ks):
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_mxfp4_quant import router_k_split

    assert router_k_split(m) == ks
