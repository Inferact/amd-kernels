# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Indexer query RoPE + MXFP4 quant against ``reference.py`` and AITER's ``q_rope_mxfp4_quant`` (bit for bit), at the
served row counts of the decode graph sizes; nothing compiles after one call."""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.indexer.q_rope_mxfp4_quant_h32_d128_rot64 import q_rope_mxfp4_quant, supports_q_rope_mxfp4_quant
from amd_kernels.indexer.q_rope_mxfp4_quant_h32_d128_rot64.reference import q_rope_mxfp4_quant_ref

ROWS = [1, 16, 48, 96, 192, 384, 768]
WSCALE = 0.0883883476483184 * 32**-0.5


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("needs gfx950")


def make(n, cs_dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn((n, 32, 128), generator=g) * 2).bfloat16().cuda()
    q[:, 3, 40:72] = 0  # an all-zero group
    q[0, 1, 5] = 3e4
    pos = torch.randint(0, 4000, (n,), generator=g).cuda()
    cs = torch.randn((4096, 64), generator=g).to(cs_dtype).cuda()
    w = torch.randn((n, 32), generator=g).bfloat16().cuda()
    return q, pos, cs, w


@pytest.mark.parametrize("n", ROWS)
@pytest.mark.parametrize("cs_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_matches_reference_and_aiter(n, cs_dtype):
    require_gfx950()
    q, pos, cs, w = make(n, cs_dtype)
    assert supports_q_rope_mxfp4_quant(q, cs)
    want = q_rope_mxfp4_quant_ref(q, pos, cs, w, WSCALE)
    got = q_rope_mxfp4_quant(q, pos, cs, w, WSCALE)
    for g, r in zip(got, want):
        assert torch.equal(g.view(torch.uint8) if g.dtype == torch.uint8 else g, r)
    again = q_rope_mxfp4_quant(q, pos, cs, w, WSCALE)
    for a, b in zip(again, got):
        assert torch.equal(a, b)
    try:
        from aiter.ops.triton.rope.q_rope_mxfp4_quant import q_rope_mxfp4_quant as aiter_op
    except ImportError:
        pytest.skip("AITER op not available")
    for g, r in zip(got, aiter_op(q, pos, cs, w, WSCALE)):
        assert torch.equal(g, r)


@torch.inference_mode()
def test_no_compile_after_warmup(monkeypatch):
    require_gfx950()
    q, pos, cs, w = make(ROWS[-1])
    q_rope_mxfp4_quant(q[:1], pos[:1], cs, w[:1], WSCALE)
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile
    monkeypatch.setattr(flyc, "compile", lambda fn, *a, **k: compiled.append(fn) or real(fn, *a, **k))
    for n in ROWS[1:]:
        q_rope_mxfp4_quant(q[:n], pos[:n], cs, w[:n], WSCALE)
    assert not compiled
