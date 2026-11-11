# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Caller-owned o_proj output: arena views, eager/capture replay, and rejection before writes."""
import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as g
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950


@pytest.mark.parametrize('m', [8, 128, 1024, 1025, 1536, 2048, 2049, 3072, 3073, 4095])
@torch.inference_mode()
def test_oproj_out(m, monkeypatch):
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, 6144, 2048, seed=m)
    a, s = g.mxfp8_e4m3_quantize_aiter(x)
    arena = torch.full((m + 2, 6144), -13, device='cuda', dtype=torch.bfloat16)
    out = arena[1:-1]
    for run in (
        lambda dst: g.mxfp8_linear(x, w, ws, out=dst),
        lambda dst: g.mxfp8_gemm(a, s, w, ws, torch.bfloat16,
                                 activation_scale_preshuffled=True, out=dst),
    ):
        expected = run(None)
        for _ in range(3):
            out.fill_(float('nan'))
            assert run(out) is out
            assert torch.equal(out, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = run(out)
        assert got is out
        for _ in range(3):
            out.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected)
        assert (arena[0] == -13).all() and (arena[-1] == -13).all()
    # Already warmed entry variants must work with new arenas without compilation.
    import flydsl.compiler as flyc
    monkeypatch.setattr(flyc, 'compile', lambda *a, **kw: pytest.fail('new compile for output pointer'))
    assert g.mxfp8_linear(x, w, ws, out=out) is out


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'stride', 'device', 'overlap', 'scale_overlap'])
@torch.inference_mode()
def test_out_rejected_before_write(bad):
    require_gfx950()
    m = 1024
    x, _, _, (w, ws) = make_inputs(m, 6144, 2048)
    a, s = g.mxfp8_e4m3_quantize_aiter(x)
    out = torch.full((m, 6144), -13, device='cuda', dtype=torch.bfloat16)
    if bad == 'shape': out = out[:m-1]
    if bad == 'dtype': out = out.float()
    if bad == 'stride': out = out.t().contiguous().t()
    if bad == 'device': out = out.cpu()
    if bad == 'overlap': a = out.view(torch.float8_e4m3fn).view(-1)[:m*2048].view(m,2048)
    if bad == 'scale_overlap': s = out.view(torch.uint8).view(-1)[:m*64].view(m,64)
    with pytest.raises(ValueError, match='out='):
        g.mxfp8_gemm(a, s, w, ws, torch.bfloat16, activation_scale_preshuffled=True, out=out)
    assert (out == -13).all()
    if bad in ('shape', 'dtype', 'stride', 'device', 'overlap'):
        original = out[:, :2048] if bad == 'overlap' else x
        with pytest.raises(ValueError, match='out='):
            g.mxfp8_linear(original, w, ws, out=out)
        assert (out == -13).all()
