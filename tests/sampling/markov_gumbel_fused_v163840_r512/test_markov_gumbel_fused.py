# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``markov_gumbel_fused`` against ``reference.py``, step by step on the kernel's own chain: the bias and logits (every
cache element within two bf16 ulps of the bias and base magnitude of the reference, 99.9% equal: the GEMV's sum order), the draw (the sampled token's
noised value within 2e-4 of the row maximum, and the argmax wherever the top two differ by more than 1e-3), invalid
rows (no cache write, plain argmax), the cache row mapping, uniform and mixed temperatures (every loop body),
1-48 rows, 1-8 steps, one scratch across row and step counts, bitwise repeats, a graph replay on new inputs."""

import pytest
import torch

from amd_kernels.sampling import markov_gumbel_fused_v163840_r512 as mk
from amd_kernels.sampling.markov_gumbel_fused_v163840_r512.reference import (
    RANK,
    V,
    bias_reference,
    logits_reference,
    noised_reference,
)

SENTINEL = 7.0


def require_gfx950():
    if not mk.available():
        pytest.skip("requires gfx950 with FlyDSL")


def make(n, k, seed, temps, invalid=(), mapped=False, dev="cuda"):
    gen = torch.Generator().manual_seed(seed)
    d = {}
    d["base"] = (torch.randn(n, k, V, generator=gen) * 3).bfloat16().to(dev)
    d["w1"] = (torch.randn(V, RANK, generator=gen) * 0.05).bfloat16().to(dev)
    d["w2"] = (torch.randn(V, RANK, generator=gen) * 0.05).bfloat16().to(dev)
    d["ids"] = torch.randint(0, V, (n + 5,), generator=gen, dtype=torch.int32).to(dev)
    d["anchor"] = torch.randperm(n + 5, generator=gen)[:n].to(dev)
    slots = n + 3
    st = torch.randperm(slots, generator=gen)[:n]
    d["slots"] = st
    temp_full = torch.ones(slots)
    temp_full[st] = torch.tensor([temps[i % len(temps)] for i in range(n)], dtype=torch.float32)
    d["temp"] = temp_full.to(dev)
    d["seeds"] = torch.randint(0, 2**62, (slots,), generator=gen).to(dev)
    idx = st.view(n, 1).repeat(1, k).contiguous()
    for r in invalid:
        idx[r, :] = -1
    d["idx"] = idx.to(dev)
    d["pos"] = torch.randint(1, 200000, (n, k), generator=gen).to(dev)
    if mapped:
        rm = torch.full((slots,), -1, dtype=torch.int32)
        shard = torch.randperm(slots, generator=gen)[: slots - 2]
        rm[shard] = torch.arange(slots - 2, dtype=torch.int32)
        d["rowmap"] = rm.to(dev)
        d["cache"] = torch.full((slots - 2, k + 1, V), SENTINEL, dtype=torch.bfloat16, device=dev)
    else:
        d["rowmap"] = None
        d["cache"] = torch.full((slots, k + 1, V), SENTINEL, dtype=torch.bfloat16, device=dev)
    d["toks"] = torch.zeros(n + 2, k, dtype=torch.int64, device=dev)
    d["scratch"] = torch.zeros(mk.scratch_dwords(n), dtype=torch.int32, device=dev)
    return d


def run(d):
    mk.markov_gumbel_fused(d["base"], d["w1"], d["w2"], d["ids"], d["anchor"], d["toks"], d["idx"], d["pos"],
                           d["temp"], d["seeds"], d["cache"], d["scratch"], d["rowmap"])


def cache_row(d, r, s):
    slot = int(d["idx"][r, s])
    if slot < 0:
        return None
    if d["rowmap"] is None:
        return slot
    row = int(d["rowmap"][slot])
    return None if row < 0 else row


def check(d, n, k):
    torch.cuda.synchronize()
    for s in range(k):
        prev = d["ids"][d["anchor"]].long() if s == 0 else d["toks"][:n, s - 1]
        bias = bias_reference(d["w1"], d["w2"], prev)
        ref = logits_reference(d["base"][:, s], bias)
        bound = (d["base"][:, s].float().abs() + bias.float().abs()) * 2**-6 + 1e-30
        ys = torch.empty_like(ref)
        for r in range(n):
            row = cache_row(d, r, s)
            if row is None:
                continue
            ys[r] = d["cache"][row, s]
            diff = (ys[r].float() - ref[r].float()).abs()
            assert bool((diff <= bound[r]).all()), f"row {r} step {s}: logits differ by more than two bf16 ulps"
            assert float((diff > 0).float().mean()) < 1e-3, f"row {r} step {s}: bias sum order moved too many logits"
            assert bool((d["cache"][row, k] == SENTINEL).all()), "wrote past the step columns"
        for r in range(n):
            slot = int(d["idx"][r, s])
            row = cache_row(d, r, s)
            y = ys[r] if row is not None else ref[r]
            temp = float(d["temp"][slot]) if slot >= 0 else 0.0
            nz = noised_reference(y, temp, int(d["seeds"][max(slot, 0)]), s, int(d["pos"][r, s]) - 1)
            tok = int(d["toks"][r, s])
            top2 = nz.topk(2).values
            assert float(nz[tok]) >= float(top2[0]) - 2e-4, f"row {r} step {s}: token {tok} is not the draw"
            if float(top2[0] - top2[1]) > 1e-3:
                assert tok == int(nz.argmax()), f"row {r} step {s}"
    return True


@pytest.mark.parametrize(
    "n,k,temps,invalid,mapped",
    [
        (1, 4, [0.7], (), False),
        (8, 4, [0.0, 1.0, 0.7, 1.3], (), False),
        (13, 4, [1.0], (), False),
        (16, 3, [0.0], (), False),
        (17, 3, [0.5, 2.0], (2,), False),
        (20, 3, [1.0, 0.0], (), True),
        (33, 5, [0.7, 1.0, 0.0, 1e-3], (0, 5), True),
        (48, 3, [1.0], (), False),
        (48, 3, [0.0, 1.0, 0.8, 1.2, 1.0, 0.6], (1, 47), True),
        (20, 1, [0.7], (), False),
        (20, 8, [1.0, 0.7], (3,), True),
    ],
)
def test_chain(n, k, temps, invalid, mapped):
    require_gfx950()
    d = make(n, k, 100 + n, temps, invalid, mapped)
    run(d)
    assert check(d, n, k)


def test_scratch_reused_across_variants():
    """One scratch and call after call of different row counts and step counts (generation tags, rank counters)."""
    require_gfx950()
    scratch = torch.zeros(mk.scratch_dwords(48), dtype=torch.int32, device="cuda")
    for i, (n, k) in enumerate([(8, 3), (48, 4), (20, 2), (8, 5), (48, 1), (33, 3)]):
        d = make(n, k, 40 + i, [0.7, 1.0, 0.0])
        d["scratch"] = scratch
        run(d)
        assert check(d, n, k)


def test_repeats_bitwise_and_graph_replay():
    require_gfx950()
    n, k = 20, 3
    d = make(n, k, 5, [0.7, 1.0, 0.0])
    run(d)
    first = (d["toks"].clone(), d["cache"].clone())
    d["cache"].fill_(SENTINEL)
    d["toks"].zero_()
    run(d)
    assert torch.equal(first[0], d["toks"]) and torch.equal(first[1], d["cache"])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(d)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(d)
    new = make(n, k, 6, [0.7, 1.0, 0.0])
    for key in ("base", "ids", "anchor", "temp", "seeds", "idx", "pos", "w1", "w2"):
        d[key].copy_(new[key])
    d["cache"].fill_(SENTINEL)
    g.replay()
    assert check(d, n, k)
    g.replay()
    assert check(d, n, k)


def test_no_compile_after_warm_up():
    require_gfx950()
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels import warm_up

    warm_up.set_auto(True)
    try:
        d = make(48, 3, 9, [0.7])
        run(d)
        torch.cuda.synchronize()
        real, compiles = MlirCompiler.compile, []

        def counting(*a, **kw):
            compiles.append(1)
            return real(*a, **kw)

        MlirCompiler.compile = counting
        try:
            for n in (1, 7, 8, 13, 16, 17, 20, 32, 33, 40, 48):
                dn = make(n, 3, n, [0.7, 1.0, 0.0])
                run(dn)
                torch.cuda.synchronize()
        finally:
            MlirCompiler.compile = real
        assert not compiles, f"{len(compiles)} compiles after the warm-up"
    finally:
        warm_up.set_auto(False)
