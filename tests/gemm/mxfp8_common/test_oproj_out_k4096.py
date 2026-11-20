# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Caller-owned output arena (``out=``) of the o_proj N 6144 x K 4096: arena views at decode and prefill row counts,
the same bytes as a fresh output, graph replay, and rejection before any write (other shapes, bad arenas)."""
import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as g

from .test_gemm import make_inputs, require_gfx950

N, K = 6144, 4096


@pytest.mark.parametrize("m", [1, 8, 64, 200, 512, 1024, 1025, 2048, 4095])
@torch.inference_mode()
def test_oproj_out(m):
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, N, K, seed=m)
    a, s = g.mxfp8_e4m3_quantize_aiter(x)
    arena = torch.full((m + 2, N), -13, device="cuda", dtype=torch.bfloat16)
    out = arena[1:-1]
    for run in (
        lambda dst: g.mxfp8_linear(x, w, ws, out=dst),
        lambda dst: g.mxfp8_gemm(a, s, w, ws, torch.bfloat16, activation_scale_preshuffled=True, out=dst),
    ):
        expected = run(None)
        for _ in range(2):
            out.fill_(float("nan"))
            assert run(out) is out
            assert torch.equal(out, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = run(out)
        assert got is out
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        assert (arena[0] == -13).all() and (arena[-1] == -13).all()


@pytest.mark.parametrize("bad", ["shape", "dtype", "stride", "overlap", "other_shape"])
@torch.inference_mode()
def test_out_rejected_before_write(bad):
    require_gfx950()
    m, n = 256, N
    if bad == "other_shape":
        n = 4608
    x, _, _, (w, ws) = make_inputs(m, n, K if bad != "other_shape" else 6144)
    a, s = g.mxfp8_e4m3_quantize_aiter(x)
    out = torch.full((m, n), -13, device="cuda", dtype=torch.bfloat16)
    if bad == "shape":
        out = out[: m - 1]
    if bad == "dtype":
        out = out.float()
    if bad == "stride":
        out = out.t().contiguous().t()
    if bad == "overlap":
        a = out.view(torch.float8_e4m3fn).view(-1)[: m * K].view(m, K)
    with pytest.raises(ValueError, match="out="):
        g.mxfp8_gemm(a, s, w, ws, torch.bfloat16, activation_scale_preshuffled=True, out=out)
    assert (out == -13).all()
    if bad != "overlap":
        with pytest.raises(ValueError, match="out="):
            g.mxfp8_linear(x, w, ws, out=out)
        assert (out == -13).all()
