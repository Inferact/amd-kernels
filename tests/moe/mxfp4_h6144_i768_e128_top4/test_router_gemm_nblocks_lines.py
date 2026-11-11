# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The two-wave line-layout router writes the nblocks kernel's split-48 partials bit for bit: partial tiles, strided
inputs, guarded outputs, changed-input replay and the public dispatch."""
import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm as rg
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm_nblocks as old
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm_nblocks_lines as new
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.layout import router_line_layout
from .workload import make_activation, require_gfx950


@pytest.mark.parametrize("m", [17, 18, 23, 24, 30, 31, 32])
@pytest.mark.parametrize("strided", [False, True])
def test_line_partial_bytes_guards_and_replay(m, strided):
    require_gfx950()
    g = torch.Generator(device="cuda").manual_seed(1107 + m)
    x = torch.empty((m, 6146 if strided else 6144), dtype=torch.bfloat16, device="cuda")[:, :6144]
    x.copy_(make_activation(m, seed=m))
    weight = torch.randn((128, 6144), generator=g, device="cuda") / 6144**0.5
    lines = router_line_layout(weight)
    ref = torch.empty((48, m, 128), device="cuda")
    guard = torch.full((ref.numel() + 64,), 17.0, device="cuda")
    out = guard[32:-32].view_as(ref)
    baseline = old._get_kernel(*rg.nblocks_config(m))
    candidate = new._get_kernel()

    def reference():
        rg._run_compiled(baseline, x.data_ptr(), weight.data_ptr(), ref.data_ptr(), m, x.stride(0),
                         0, 0, 0, 0, torch.cuda.current_stream())

    def run():
        rg._run_compiled(candidate, x.data_ptr(), lines.data_ptr(), out.data_ptr(), m, x.stride(0),
                         torch.cuda.current_stream())

    reference()
    run()
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for seed in (901, 902, 901):
        x.copy_(make_activation(m, seed=seed))
        reference()
        for _ in range(2):
            out.fill_(float("nan"))
            graph.replay()
            assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
        public = rg.router_gemm_partials(x, weight, lines)
        assert torch.equal(public.permute(1, 0, 2).contiguous().view(torch.uint8), ref.view(torch.uint8))
    assert (guard[:32] == 17).all() and (guard[-32:] == 17).all()
