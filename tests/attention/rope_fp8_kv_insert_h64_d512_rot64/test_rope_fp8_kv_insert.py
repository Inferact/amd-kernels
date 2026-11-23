# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Fused query RoPE / e4m3 quant and KV RoPE / e4m3 insert against ``reference.py`` (bit for bit), at the served row
counts of the decode graph sizes, and against the vLLM C op when it is importable. Nothing compiles after one call."""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.attention.rope_fp8_kv_insert_h64_d512_rot64 import rope_fp8_kv_insert, supports_rope_fp8_kv_insert
from amd_kernels.attention.rope_fp8_kv_insert_h64_d512_rot64.reference import rope_fp8_kv_insert_ref

ROWS = [1, 16, 48, 96, 192, 384, 768]
BS = 128


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("needs gfx950")


def make(n, kv_scale=0.7, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn((n, 64, 512), generator=g) * 3).bfloat16().cuda()
    kv = (torch.randn((n, 512), generator=g) * 3).bfloat16().cuda()
    q[0, 0, 5] = 3e4  # saturates
    kv[0, 7] = -3e4
    pos = torch.randint(0, 4000, (n,), generator=g).cuda()
    cs = torch.randn((4096, 64), generator=g).cuda()
    pages = (2 * n + BS - 1) // BS + 1
    slots = torch.randperm(pages * BS, generator=g)[:n].cuda()
    slots[n // 2] = -1
    cache = torch.randint(0, 256, (pages, BS, 512), generator=g, dtype=torch.uint8).cuda().view(torch.float8_e4m3fn)
    ks = torch.full((1,), kv_scale, device="cuda")
    qsi = torch.full((1,), 1.0 / 0.37, device="cuda")
    return q, kv, slots, pos, cs, cache, ks, qsi


def bits(x):
    return x.view(torch.uint8)


@pytest.mark.parametrize("n", ROWS)
@pytest.mark.parametrize("kv_scale", [0.7, 0.013, 3.0])
@torch.inference_mode()
def test_matches_reference(n, kv_scale):
    require_gfx950()
    q, kv, slots, pos, cs, cache, ks, qsi = make(n, kv_scale)
    assert supports_rope_fp8_kv_insert(q, cache, cs)
    want_q, want_cache = rope_fp8_kv_insert_ref(q, kv, cache, slots, pos, cs, ks, qsi)
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    rope_fp8_kv_insert(q, kv, q_fp8, cache, slots, pos, cs, ks, qsi)
    assert torch.equal(bits(q_fp8), bits(want_q))
    assert torch.equal(bits(cache), bits(want_cache))
    again = torch.empty_like(q_fp8)
    cache2 = want_cache.clone()
    rope_fp8_kv_insert(q, kv, again, cache2, slots, pos, cs, ks, qsi)
    assert torch.equal(bits(again), bits(q_fp8)) and torch.equal(bits(cache2), bits(want_cache))


@pytest.mark.parametrize("n", [16, 192, 768])
@torch.inference_mode()
def test_matches_vllm_op(n):
    require_gfx950()
    try:
        import vllm._custom_ops  # noqa: F401

        op = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
    except (ImportError, AttributeError):
        pytest.skip("vLLM C op not available")
    q, kv, slots, pos, cs, cache, ks, qsi = make(n, 0.013)
    want_cache = cache.clone()
    want_q = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    op(q.clone(), kv, want_q, want_cache, slots, pos, cs, ks, qsi, 1e-6, BS, False)
    q_fp8 = torch.empty_like(want_q)
    rope_fp8_kv_insert(q, kv, q_fp8, cache, slots, pos, cs, ks, qsi)
    assert torch.equal(bits(q_fp8), bits(want_q))
    assert torch.equal(bits(cache), bits(want_cache))


@torch.inference_mode()
def test_no_compile_after_warmup(monkeypatch):
    require_gfx950()
    compiled = []
    real = flyc.compile
    q, kv, slots, pos, cs, cache, ks, qsi = make(ROWS[-1])
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)

    def run(n):
        rope_fp8_kv_insert(q[:n], kv[:n], q_fp8[:n], cache, slots[:n], pos[:n], cs, ks, qsi)

    run(ROWS[0])
    torch.cuda.synchronize()
    monkeypatch.setattr(flyc, "compile", lambda fn, *a, **k: compiled.append(fn) or real(fn, *a, **k))
    for n in ROWS[1:]:
        run(n)
    assert not compiled
