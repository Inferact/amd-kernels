# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

``qkv_norm_rope_insert`` compiles one variant per (query heads, KV heads, page size,
insert), ``kv_norm_rope_insert_multilayer`` one per (layers, KV heads, page size, RoPE
form); a deployment fixes all of them, and the token count, the slots given and the
per-layer slot mappings are runtime arguments. The engine's warm-up (the profile run
without slots, then the CUDA-graph captures with slots) reaches both qkv variants.
Warm each variant once, then probe other token counts; a probe that compiles means the
token count reaches a compile key.
"""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import (
    kv_norm_rope_insert_multilayer,
)
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.qkv_norm_rope_insert import qkv_norm_rope_insert

from .test_norm_rope_insert import EPS, make_multilayer, require_gfx950, rope_cache, scale_pair

D = 128
# (id, query heads, KV heads, page, served token counts)
QKV_SERVED = [
    ("tp4_p128", 16, 1, 128, [4, 12, 100, 400, 1024]),
]
# (id, KV heads, page, served context-row counts)
ML_SERVED = [
    ("tp4_p128", 1, 128, [4, 64, 384, 4096]),
]


def count_compiles(monkeypatch):
    compiled = []
    real = flyc.compile

    def counting(fn, *args, **kwargs):
        compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    return compiled


@pytest.mark.parametrize("name,nq,nkv,page,served", QKV_SERVED, ids=[c[0] for c in QKV_SERVED])
@torch.inference_mode()
def test_qkv_no_compile_after_warmup(name, nq, nkv, page, served, monkeypatch):
    require_gfx950()
    gen = torch.Generator().manual_seed(nq)
    n_max = served[-1]
    qkv = torch.randn((n_max, (nq + 2 * nkv) * D), generator=gen).bfloat16().cuda()
    pos = torch.randint(0, 4096, (n_max,), generator=gen).cuda()
    slots = torch.randperm(2 * n_max, generator=gen).cuda()
    w = torch.randn(D, generator=gen).bfloat16().cuda()
    cs = rope_cache(gen)
    cache = torch.zeros(((2 * n_max + page - 1) // page, nkv, page, 2 * D), dtype=torch.uint8, device="cuda")
    q = torch.empty((n_max, nq * D), dtype=torch.bfloat16, device="cuda")
    ks, vs = scale_pair(1.0)

    def run(n, insert):
        args = (qkv[:n], pos[:n], w, w, cs, EPS, nq, nkv, q[:n])
        if insert:
            qkv_norm_rope_insert(*args, cache, slots[:n], ks, vs)
        else:
            qkv_norm_rope_insert(*args)

    run(served[0], False)  # the profile run (no slots)
    run(served[0], True)  # the graph captures
    torch.cuda.synchronize()
    compiled = count_compiles(monkeypatch)
    for n in served[1:]:
        run(n, True)
        run(n, False)
        assert not compiled, f"{name}: {n} tokens compiled {compiled}"


@pytest.mark.parametrize("name,nkv,page,served", ML_SERVED, ids=[c[0] for c in ML_SERVED])
@torch.inference_mode()
def test_multilayer_no_compile_after_warmup(name, nkv, page, served, monkeypatch):
    require_gfx950()
    gen = torch.Generator().manual_seed(nkv)
    kv, pos, weights, cs, caches, slots, ks, vs = make_multilayer(served[-1], nkv, page, True, gen)
    shared = [slots[0]] * len(slots)

    def run(n, layer_slots, n_insert=None):
        maps = [s[:n] for s in (slots if layer_slots else shared)]
        kv_norm_rope_insert_multilayer(
            kv[:n], pos[:n], weights, cs, caches, maps, ks, vs, EPS, n_insert=n_insert
        )

    run(served[0], False, 0)  # the dummy run (no valid slots)
    torch.cuda.synchronize()
    compiled = count_compiles(monkeypatch)
    for n in served[1:]:
        run(n, False)
        run(n, True)
        run(n, False, n // 2)
        assert not compiled, f"{name}: {n} rows compiled {compiled}"
