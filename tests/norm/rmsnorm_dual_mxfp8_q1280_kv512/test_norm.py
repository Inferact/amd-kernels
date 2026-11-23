# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``rmsnorm_dual_mxfp8`` against its contract (``reference.py``) and the served Triton path (``fused_q_kv_rmsnorm`` +
``mxfp8_e4m3_quantize``) at every padded decode row count: q / kv taken from a fused [m, 1792] projection output, rows
scaled over +-8 binades, zero rows; graph replay with poisoned outputs; determinism; nothing compiles after the
warm-up. The kernel's square sum is the served Triton norm's reduction tree, so against the served path every output
is bit for bit equal; torch's sum differs in the last fp32 bit, so against the reference a bf16 value can differ by an
ulp (bounded below); codes and scales are exact wherever the bf16 norm is."""

import pytest
import torch
from amd_kernels.norm import rmsnorm_dual_mxfp8_q1280_kv512 as nm
from amd_kernels.norm.rmsnorm_dual_mxfp8_q1280_kv512 import (
    dequantize_mxfp8,
    rmsnorm_dual_mxfp8,
    rmsnorm_dual_mxfp8_reference,
    warmup_rmsnorm_dual_mxfp8,
)

EPS = 1e-20
ROWS = (1, 7, 16, 17, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024)
MAX_ELEM_MISMATCH = 1e-4  # fraction of bf16 / code elements


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950 with FlyDSL")


def make(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, 1792), generator=g, device="cuda") * torch.exp2(
        torch.randint(-8, 9, (m, 1), generator=g, device="cuda").float()
    )
    if m > 1:
        x[0].zero_()
        x[1, ::7] = 0.0
    qw = (torch.rand(1280, generator=g, device="cuda") * 2 + 0.25).bfloat16()
    kw = (torch.rand(512, generator=g, device="cuda") * 2 + 0.25).bfloat16()
    return x.bfloat16(), qw, kw


def check(got, ref):
    codes, scale, kv = got
    rc, rs, rkv = ref
    assert torch.isfinite(kv.float()).all()
    assert (kv != rkv).float().mean().item() <= MAX_ELEM_MISMATCH
    assert (codes.view(torch.uint8) != rc.view(torch.uint8)).float().mean().item() <= MAX_ELEM_MISMATCH
    assert (scale != rs).float().mean().item() <= MAX_ELEM_MISMATCH
    a, b = dequantize_mxfp8(codes, scale), dequantize_mxfp8(rc, rs)
    live = b.double().norm(dim=1) > 0
    assert not a[~live].any()
    if live.any():
        x, y = a[live].double(), b[live].double()
        cos = (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1))
        assert cos.min().item() >= 0.99999


@pytest.mark.parametrize("m", ROWS)
def test_contract(m):
    require_gfx950()
    x, qw, kw = make(m, m)
    q, kv = x.split([1280, 512], dim=-1)
    got = rmsnorm_dual_mxfp8(q, kv, qw, kw, EPS)
    torch.cuda.synchronize()
    assert got[0].shape == (m, 1280) and got[1].shape == (m, 40) and got[2].shape == (m, 512)
    check(got, rmsnorm_dual_mxfp8_reference(q, kv, qw, kw, EPS))
    again = rmsnorm_dual_mxfp8(q, kv, qw, kw, EPS)
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(got, again))


@pytest.mark.parametrize("m", ROWS + (2048,))
def test_served_triton(m):
    require_gfx950()
    ops = pytest.importorskip("vllm.models.common.ops")
    quant = pytest.importorskip("vllm.model_executor.layers.quantization.utils.mxfp8_utils")
    for seed in range(3):
        x, qw, kw = make(m, m + seed)
        q, kv = x.split([1280, 512], dim=-1)
        got = rmsnorm_dual_mxfp8(q, kv, qw, kw, EPS)
        nq, nkv = ops.fused_q_kv_rmsnorm(q, kv, qw, kw, EPS)
        bc, bs = quant.mxfp8_e4m3_quantize(nq)
        assert torch.equal(got[0].view(torch.uint8), bc.view(torch.uint8))
        assert torch.equal(got[1], bs.view(torch.uint8))
        assert torch.equal(got[2], nkv)


def test_extreme_rows():
    require_gfx950()
    m = 16
    x = torch.zeros((m, 1792), device="cuda", dtype=torch.bfloat16)
    x[1, :] = 3.0e38
    x[2, :] = 1.0e-30
    x[3, ::2] = -448.0
    x[4, 5] = 7.0
    x[5] = torch.arange(1792, device="cuda").bfloat16() - 900
    q, kv = x.split([1280, 512], dim=-1)
    qw, kw = torch.ones(1280, device="cuda", dtype=torch.bfloat16), torch.ones(512, device="cuda", dtype=torch.bfloat16)
    got = rmsnorm_dual_mxfp8(q, kv, qw, kw, EPS)
    check(got, rmsnorm_dual_mxfp8_reference(q, kv, qw, kw, EPS))


def test_caller_buffers_and_graph():
    require_gfx950()
    m = 192
    x, qw, kw = make(m, 5)
    q, kv = x.split([1280, 512], dim=-1)
    codes = torch.empty((m, 1280), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty((m, 40), device="cuda", dtype=torch.uint8)
    kv_out = torch.empty((m, 512), device="cuda", dtype=torch.bfloat16)
    rmsnorm_dual_mxfp8(q, kv, qw, kw, EPS, codes=codes, scale=scale, kv_out=kv_out)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = rmsnorm_dual_mxfp8(q, kv, qw, kw, EPS, codes=codes, scale=scale, kv_out=kv_out)
    assert out[0] is codes and out[1] is scale and out[2] is kv_out
    for i in range(3):
        x.mul_(0.75).add_(0.0625)
        eager = rmsnorm_dual_mxfp8(q, kv, qw, kw, EPS)
        codes.view(torch.uint8).fill_(255)
        scale.fill_(255)
        kv_out.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(codes.view(torch.uint8), eager[0].view(torch.uint8))
        assert torch.equal(scale, eager[1]) and torch.equal(kv_out, eager[2])


def test_warmup_no_compile_after():
    require_gfx950()
    from flydsl.compiler.jit_function import MlirCompiler

    warmup_rmsnorm_dual_mxfp8()
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        for m in ROWS:
            x, qw, kw = make(m, 3)
            q, kv = x.split([1280, 512], dim=-1)
            rmsnorm_dual_mxfp8(q, kv, qw, kw, EPS)
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig
    assert nm.__name__.endswith("q1280_kv512")
