# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Fused fp32 / bf16 GEMM against ``reference.py`` (cosine similarity, bf16 columns within one ulp), repeat runs
bit-identical, and nothing compiles after a warm-up at each M tile."""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.gemm.bf16_k5120_n1056_f32_n32 import gemm, shuffle_weight, supports
from amd_kernels.gemm.bf16_k5120_n1056_f32_n32.reference import gemm_ref

ROWS = [1, 16, 17, 48, 96, 192, 384, 768, 769, 848, 1024]
SPLITS = [(1024, 32), (512, 32), (0, 32)]


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("needs gfx950")


def make(m, na, nb, seed=0):
    g = torch.Generator().manual_seed(seed + m)
    x = torch.randn((m, 5120), generator=g).bfloat16().cuda()
    w = (torch.randn((na + nb, 5120), generator=g) * 0.03).bfloat16().cuda()
    return x, w


def cosine(a, b):
    a, b = a.double(), b.double()
    return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-30)).min().item()


@pytest.mark.parametrize("na,nb", SPLITS)
@pytest.mark.parametrize("m", ROWS)
@torch.inference_mode()
def test_matches_reference(m, na, nb):
    require_gfx950()
    x, w = make(m, na, nb)
    ws = shuffle_weight(w[:na] if na else None, w[na:])
    assert supports(x, ws, na)
    out_a = torch.empty((m, na), dtype=torch.float32, device="cuda") if na else None
    out_b = torch.empty((m, nb), dtype=torch.bfloat16, device="cuda")
    gemm(x, ws, out_a, out_b)
    ref_a, ref_b = gemm_ref(x, w, na)
    if na:
        assert cosine(out_a, ref_a) >= 0.99999
        assert (out_a - ref_a).abs().max() <= 1e-5 * ref_a.abs().max() + 1e-6
    assert cosine(out_b.float(), ref_b.float()) >= 0.99999
    ulp = (out_b.float() - ref_b.float()).abs() <= ref_b.float().abs() * 2**-7 + 1e-6
    assert ulp.all()
    again_a = torch.empty_like(out_a) if na else None
    again_b = torch.empty_like(out_b)
    gemm(x, ws, again_a, again_b)
    assert torch.equal(again_b, out_b) and (not na or torch.equal(again_a, out_a))


@torch.inference_mode()
def test_no_compile_after_warmup(monkeypatch):
    require_gfx950()
    x, w = make(1024, 1024, 32)
    ws = shuffle_weight(w[:1024], w[1024:])
    out_a = torch.empty((1024, 1024), dtype=torch.float32, device="cuda")
    out_b = torch.empty((1024, 32), dtype=torch.bfloat16, device="cuda")
    tiles = [16, 32, 96, 192, 384, 768, 1024]
    for m in tiles:
        gemm(x[:m], ws, out_a[:m], out_b[:m])
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile
    monkeypatch.setattr(flyc, "compile", lambda fn, *a, **k: compiled.append(fn) or real(fn, *a, **k))
    for m in (1, 7, 40, 100, 250, 500, 700, 850, 1000):
        gemm(x[:m], ws, out_a[:m], out_b[:m])
    assert not compiled


@pytest.mark.parametrize("na,nb,m", [(0, 32, 16), (0, 32, 192), (0, 32, 768), (512, 32, 96), (1024, 32, 384), (1024, 32, 768), (0, 32, 1024), (512, 32, 848), (1024, 32, 1024)])
@torch.inference_mode()
def test_k_split_deterministic_and_graph_safe(na, nb, m):
    """The K split over workgroups (partials + ticket, the last workgroup of a tile adds the splits in K order): 30
    repeats and a replayed graph over changing activations are bit-identical to a fresh call, the tickets return to
    zero (a stale ticket would corrupt the next launch), and outputs poisoned before a replay are fully rewritten."""
    require_gfx950()
    x, w = make(m, na, nb, seed=5)
    ws = shuffle_weight(w[:na] if na else None, w[na:])
    out_a = torch.empty((m, na), dtype=torch.float32, device="cuda") if na else None
    out_b = torch.empty((m, nb), dtype=torch.bfloat16, device="cuda")
    gemm(x, ws, out_a, out_b)
    ref_a = None if out_a is None else out_a.clone()
    ref_b = out_b.clone()
    for _ in range(30):
        if out_a is not None:
            out_a.fill_(float("nan"))
        out_b.fill_(float("nan"))
        gemm(x, ws, out_a, out_b)
        assert torch.equal(out_b, ref_b) and (out_a is None or torch.equal(out_a, ref_a))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gemm(x, ws, out_a, out_b)
    for i in range(5):
        x.mul_(0.5).add_(0.125)
        e_a = None if out_a is None else torch.empty_like(out_a)
        e_b = torch.empty_like(out_b)
        gemm(x, ws, e_a, e_b)
        if out_a is not None:
            out_a.fill_(float("nan"))
        out_b.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_b, e_b) and (out_a is None or torch.equal(out_a, e_a))
    from amd_kernels.gemm.bf16_k5120_n1056_f32_n32.host import _split_buffers

    assert not _split_buffers(x.device)[1].any(), "a ticket was left set"
