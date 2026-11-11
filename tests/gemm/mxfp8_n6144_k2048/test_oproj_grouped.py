# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""96-column o_proj tiles share their four row panels within each XCD (M385-512)."""
import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950


@pytest.mark.parametrize('m', [320, 321, 352, 384, 385, 416, 448, 449, 480, 512])
@torch.inference_mode()
def test_oproj_grouped_rows(m):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 6144, 2048, seed=m)
    x[0].zero_()
    c = g._select_prequant_config(m, 6144, 2048, torch.bfloat16)
    name = g._decode_launcher(6144, 2048, c, False).kernel_name
    assert name.removesuffix('_ownread').removesuffix('_ip').endswith('_g4') == (m >= 385)
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, c.tile_m)
    got = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws, accumulate=torch.float64)
    mt = ref.compare(got, expected)
    assert mt['cos'] >= ref.COS_MIN and mt['rel_l2'] <= ref.REL_L2_MAX
    active = expected.double().norm(dim=1) > 0
    rel = (got.double() - expected.double()).norm(dim=1) / expected.double().norm(dim=1).clamp_min(1e-30)
    assert rel[active].max() <= ref.REL_L2_MAX
    assert got[0].count_nonzero() == 0
    assert torch.equal(got, g.mxfp8_linear(x, w, s))
    assert torch.equal(got, g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    for _ in range(3):
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, got)
