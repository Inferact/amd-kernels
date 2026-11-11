# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Guard rows around a caller-owned output at terminal-pipeline boundaries.

The 128x192 output transpose uses 24576 BF16 elements of a 73728-element
operand region. Its XOR only permutes columns within each 64-column block.
Input and output buffer descriptors retain
logical M*K / M*N extents; padded rows read zero and drop their output stores.
"""
import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950


@pytest.mark.parametrize('m', [769, 770, 831, 1023, 1024])
@torch.inference_mode()
def test_terminal_pipeline_output_guards(m):
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, 6144, 2048, seed=m)
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    expected = g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    arena = torch.empty((m + 2, 6144), dtype=torch.bfloat16, device='cuda')
    arena.view(torch.int16).fill_(0x3555)
    before, after = arena[0].clone(), arena[-1].clone()
    out = arena[1:-1]
    run = lambda: g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16,
                               activation_scale_preshuffled=True, out=out)
    assert run() is out
    assert torch.equal(expected, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(8):
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(expected, out)
        assert torch.equal(arena[0], before)
        assert torch.equal(arena[-1], after)
