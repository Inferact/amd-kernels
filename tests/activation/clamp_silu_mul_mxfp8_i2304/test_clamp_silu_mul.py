# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``clamp_silu_mul_mxfp8`` against its contract (``reference.py``) and the served ops (``SiluAndMulWithClamp``'s HIP
op + ``mxfp8_e4m3_quantize``) at every padded decode row count: gate_up scaled over +-6 binades, values at and beyond
the limit, zero rows; graph replay with poisoned outputs; determinism; nothing compiles after the warm-up. Codes and
scales equal the served path bit for bit; the reference's exp may differ in the last fp32 bit (bounded below)."""

import pytest
import torch
from amd_kernels.activation.clamp_silu_mul_mxfp8_i2304 import (
    clamp_silu_mul_mxfp8,
    clamp_silu_mul_mxfp8_reference,
    dequantize_mxfp8,
    warmup_clamp_silu_mul_mxfp8,
)

I = 2304
LIMIT = 10.0
ROWS = (1, 7, 16, 17, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024)


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950 with FlyDSL")


def make(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, 2 * I), generator=g, device="cuda") * 3.0
    x *= torch.exp2(torch.randint(-6, 7, (m, 1), generator=g, device="cuda").float())
    x[:, :8] = torch.tensor([10.0, 11.0, -10.0, -30.0, 0.0, -0.0, 1e-30, 300.0], device="cuda")
    x[:, I : I + 8] = torch.tensor([10.0, 11.0, -10.0, -30.0, 0.0, -0.0, 1e-30, 300.0], device="cuda")
    if m > 1:
        x[0].zero_()
    return x.bfloat16()


def served(x):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    out = torch.empty((x.shape[0], I), dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul_with_clamp(out, x, LIMIT, 1.0, 0.0)
    c, s = mxfp8_e4m3_quantize(out)
    return c, s.view(torch.uint8)


@pytest.mark.parametrize("m", ROWS)
def test_contract(m):
    require_gfx950()
    x = make(m, m)
    c, s = clamp_silu_mul_mxfp8(x, LIMIT)
    torch.cuda.synchronize()
    assert c.shape == (m, I) and s.shape == (m, I // 32)
    rc, rs = clamp_silu_mul_mxfp8_reference(x, LIMIT)
    assert (c.view(torch.uint8) != rc.view(torch.uint8)).float().mean().item() <= 1e-5
    assert (s != rs).float().mean().item() <= 1e-5
    if m > 1:
        cos = torch.nn.functional.cosine_similarity(
            dequantize_mxfp8(c, s)[1:].double(), dequantize_mxfp8(rc, rs)[1:].double(), dim=1
        )
        assert cos.min().item() >= 0.99999
    c2, s2 = clamp_silu_mul_mxfp8(x, LIMIT)
    assert torch.equal(c.view(torch.uint8), c2.view(torch.uint8)) and torch.equal(s, s2)


@pytest.mark.parametrize("m", ROWS)
def test_served_bitwise(m):
    require_gfx950()
    pytest.importorskip("vllm")
    x = make(m, m + 7)
    c, s = clamp_silu_mul_mxfp8(x, LIMIT)
    bc, bs = served(x)
    assert torch.equal(c.view(torch.uint8), bc.view(torch.uint8))
    assert torch.equal(s, bs)


def test_row_stride():
    require_gfx950()
    x = make(48, 3)
    wide = torch.zeros((48, 2 * I + 64), device="cuda", dtype=torch.bfloat16)
    wide[:, : 2 * I] = x
    c, s = clamp_silu_mul_mxfp8(wide[:, : 2 * I], LIMIT)
    c0, s0 = clamp_silu_mul_mxfp8(x, LIMIT)
    assert torch.equal(c.view(torch.uint8), c0.view(torch.uint8)) and torch.equal(s, s0)


def test_caller_buffers_and_graph():
    require_gfx950()
    m = 192
    x = make(m, 5)
    codes = torch.empty((m, I), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty((m, I // 32), device="cuda", dtype=torch.uint8)
    clamp_silu_mul_mxfp8(x, LIMIT, codes=codes, scale=scale)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = clamp_silu_mul_mxfp8(x, LIMIT, codes=codes, scale=scale)
    assert out[0] is codes and out[1] is scale
    for i in range(3):
        x.mul_(0.75).add_(0.0625)
        ec, es = clamp_silu_mul_mxfp8(x, LIMIT)
        codes.view(torch.uint8).fill_(255)
        scale.fill_(255)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(codes.view(torch.uint8), ec.view(torch.uint8)) and torch.equal(scale, es)


def test_warmup_no_compile_after():
    require_gfx950()
    from flydsl.compiler.jit_function import MlirCompiler

    warmup_clamp_silu_mul_mxfp8()
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        for m in ROWS:
            clamp_silu_mul_mxfp8(make(m, 3), LIMIT)
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig
