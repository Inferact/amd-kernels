# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Caller-owned o_proj output at K 4096: every row family writes only its [M, 6144] arena view (guard rows around
it), the same bits as an allocated output, eager and in graph replay."""
import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as g
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950


@pytest.mark.parametrize("m", [8, 32, 33, 256, 768, 769, 1024, 1025, 1281, 2048, 2049, 3329, 4096, 5121, 6145])
@torch.inference_mode()
def test_oproj_out_k4096(m):
    require_gfx950()
    if (6144, 4096) not in getattr(g, "_OPROJ_OUT_SHAPES", ()):
        pytest.skip("the dispatcher takes out= only at K 2048")
    x, _, _, (w, ws) = make_inputs(m, 6144, 4096, seed=m)
    a, s = g.mxfp8_e4m3_quantize_aiter(x)
    arena = torch.full((m + 2, 6144), -13, device="cuda", dtype=torch.bfloat16)
    out = arena[1:-1]
    for run in (
        lambda dst: g.mxfp8_linear(x, w, ws, out=dst),
        lambda dst: g.mxfp8_gemm(a, s, w, ws, torch.bfloat16, activation_scale_preshuffled=True, out=dst),
    ):
        expected = run(None)
        out.fill_(float("nan"))
        assert run(out) is out
        assert torch.equal(out, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(out)
        for _ in range(2):
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected)
        assert (arena[0] == -13).all() and (arena[-1] == -13).all()
