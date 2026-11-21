# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``markov_chain`` against ``reference.py``, following the kernel's own chain (teacher forcing): per step the bf16
cache rows from the previous token (all but a few last-bit elements exact: the fp32 bias sum has the kernel's order),
and the token as the argmax of the fp64-noised cache rows (ids exact except where the top two differ by less than
1e-4 + 1e-6 |top|); LM-like logits, temperatures 0 / 1 / 0.7 / 1.3 / 2, inert rows (slot -1: no cache write, greedy);
every row-count variant (1-200 rows, chunks above 160); ties (lowest column, temperature 0) and tile boundaries;
int32 / int64 slot maps and ids; bitwise determinism; graph replay with new inputs; the first call's warm-up compiles
every variant and nothing compiles later. With vLLM installed, the Gumbel draws are compared with its Triton
``gumbel_sample`` bit for bit."""

import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.sampling import markov_chain_gumbel_bf16_v129280_r256_s5 as mc
from amd_kernels.sampling.markov_chain_gumbel_bf16_v129280_r256_s5 import NSPEC, R, V, markov_chain
from amd_kernels.sampling.markov_chain_gumbel_bf16_v129280_r256_s5.reference import (
    bias_reference,
    logits_reference,
    sample_reference,
)

SLOTS = 512
TEMPS = (0.0, 1.0, 0.7, 1.3, 2.0)


def require_gfx950():
    if not mc.available():
        pytest.skip("requires gfx950 with FlyDSL")


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator().manual_seed(1234)
    w1 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).cuda()
    w2 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).cuda()
    return w1, w2


def lm_rows(n, gen):
    """bf16 logits of n * 5 rows: a Gaussian body and a heavy head, as an LM's."""
    x = torch.randn(n * NSPEC, V, generator=gen) * 2.0
    head = torch.randint(0, V, (n * NSPEC, 48), generator=gen)
    x.scatter_(1, head, torch.rand(n * NSPEC, 48, generator=gen) * 10 + 4)
    return x.to(torch.bfloat16).cuda()


def make(n, gen, inert=7, idx_dtype=torch.int64, ids_dtype=torch.int32, temps=TEMPS):
    slots = torch.randperm(SLOTS, generator=gen)[:n]
    idx = slots.view(n, 1).expand(n, NSPEC).clone()
    if inert:
        idx[torch.arange(n) % inert == inert - 1] = -1
    d = dict(
        base=lm_rows(n, gen),
        ids=torch.randint(0, V, (4096,), generator=gen).to(ids_dtype),
        anchor=torch.randint(0, 4096, (n,), generator=gen),
        idx=idx.to(idx_dtype),
        pos=131072 + torch.randint(0, 4096, (n, 1), generator=gen) + torch.arange(NSPEC),
        temp=torch.tensor([temps[i % len(temps)] for i in range(SLOTS)], dtype=torch.float32),
        seeds=torch.randint(-(2**62), 2**62, (SLOTS,), generator=gen),
    )
    return {k: v.cuda() for k, v in d.items()}


def run(d, w, cache=None, tokens=None, fill=0.0):
    n = d["anchor"].shape[0]
    cache = torch.full((SLOTS, NSPEC, V), fill, dtype=torch.bfloat16, device="cuda") if cache is None else cache
    tokens = torch.full((n, NSPEC), -1, dtype=torch.int64, device="cuda") if tokens is None else tokens
    markov_chain(d["base"], w[0], w[1], d["ids"], d["anchor"], d["idx"], d["pos"], d["temp"], d["seeds"], cache, tokens)
    return cache, tokens


def check(d, w, cache, tokens, fill=0.0):
    n = d["anchor"].shape[0]
    w1, w2 = w
    base = d["base"].view(n, NSPEC, V)
    idx = d["idx"].long()
    prev = d["ids"][d["anchor"]].long()
    touched = torch.zeros(SLOTS, NSPEC, dtype=torch.bool, device="cuda")
    clear_total = clear_ok = 0
    for s in range(NSPEC):
        valid = idx[:, s] >= 0
        bias = bias_reference(prev, w1, w2)
        y = logits_reference(base[:, s], bias)
        slot = idx[:, s].clamp(min=0)
        yk = cache[slot, s]
        diff = (yk.float() - y.float()).abs()
        bad = (diff > 0) & valid[:, None]
        assert float(bad.float().mean()) < 1e-4, (s, int(bad.sum()))
        # a last-bit difference of the bias, then of the sum
        assert bool((diff <= 2.0 ** -7 * (y.float().abs() + bias.float().abs()) + 1e-30)[valid].all()), s
        touched[slot[valid], s] = True
        temp = torch.where(valid, d["temp"][slot], torch.zeros_like(d["temp"][slot]))
        # valid rows from the kernel's cache values; inert rows (greedy) from the reference's
        src = torch.where(valid[:, None], yk, y)
        ids, gap = sample_reference(src, temp, d["seeds"][slot], s, d["pos"][:, s] - 1)
        top = src.float().abs().max(dim=1).values
        clear = gap > 1e-4 + 1e-6 * top
        got = tokens[:, s]
        assert bool((got[clear] == ids[clear]).all()), (s, int((got != ids).sum()))
        clear_total += clear.numel()
        clear_ok += int(clear.sum())
        prev = got
    assert clear_ok > 0.9 * clear_total
    untouched = ~touched
    assert bool((cache[untouched] == fill).all()), "rows of inert requests or unused slots were written"


@pytest.mark.parametrize("n", [1, 5, 11, 16, 17, 32, 48, 64, 65, 100, 128, 160])
def test_chain_rows(weights, n):
    require_gfx950()
    d = make(n, torch.Generator().manual_seed(n))
    cache, tokens = run(d, weights, fill=7.0)
    torch.cuda.synchronize()
    check(d, weights, cache, tokens, fill=7.0)


@pytest.mark.parametrize("n", [161, 200])
def test_chunked_rows(weights, n):
    require_gfx950()
    d = make(n, torch.Generator().manual_seed(n))
    cache, tokens = run(d, weights)
    check(d, weights, cache, tokens)
    # the same bits as separate calls of the chunks
    c2 = torch.zeros_like(cache)
    t2 = torch.empty_like(tokens)
    for r0 in range(0, n, mc.MAX_ROWS):
        m = min(mc.MAX_ROWS, n - r0)
        e = dict(d, base=d["base"][r0 * NSPEC:(r0 + m) * NSPEC], anchor=d["anchor"][r0:r0 + m],
                 idx=d["idx"][r0:r0 + m], pos=d["pos"][r0:r0 + m])
        run(e, weights, c2, t2[r0:r0 + m])
    assert torch.equal(tokens, t2) and torch.equal(cache.view(torch.int16), c2.view(torch.int16))


@pytest.mark.parametrize("idx_dtype,ids_dtype", [(torch.int32, torch.int64), (torch.int64, torch.int32)])
def test_index_dtypes(weights, idx_dtype, ids_dtype):
    require_gfx950()
    gen = torch.Generator().manual_seed(21)
    a = make(24, gen, idx_dtype=torch.int64, ids_dtype=torch.int32)
    b = dict(a, idx=a["idx"].to(idx_dtype), ids=a["ids"].to(ids_dtype))
    ca, ta = run(a, weights)
    cb, tb = run(b, weights)
    assert torch.equal(ta, tb) and torch.equal(ca.view(torch.int16), cb.view(torch.int16))


def test_strided_inputs(weights):
    """Rows of base logits with a larger stride, a token buffer wider than the batch, a view of idx_map."""
    require_gfx950()
    n = 20
    d = make(n, torch.Generator().manual_seed(8))
    wide = torch.zeros(n * NSPEC, V + 64, dtype=torch.bfloat16, device="cuda")
    wide[:, :V] = d["base"]
    e = dict(d, base=wide[:, :V])
    tokens = torch.full((n + 9, NSPEC), -1, dtype=torch.int64, device="cuda")
    ca, ta = run(d, weights)
    cb, tb = run(e, weights, tokens=tokens)
    assert torch.equal(ta, tb[:n]) and torch.equal(ca.view(torch.int16), cb.view(torch.int16))
    assert bool((tb[n:] == -1).all())


def test_ties_and_boundaries(weights):
    """Zero bias head: the logits are the base rows; at temperature 0 the token is the lowest column of the maximum."""
    require_gfx950()
    w1, _ = weights
    w2 = torch.zeros_like(weights[1])
    n = 16
    base = torch.full((n, NSPEC, V), -1.0, dtype=torch.bfloat16)
    peaks = [[0], [511], [512], [129279], [129023, 129024], [5, 50000, 129279], [1023, 1024, 4096], [255, 256]]
    for r in range(n):
        for s in range(NSPEC):
            cols = peaks[(r + s) % len(peaks)]
            base[r, s, cols] = 3.0
    base[3] = 0.0  # all equal: column 0
    d = make(n, torch.Generator().manual_seed(2), inert=0, temps=(0.0,))
    d["base"] = base.view(n * NSPEC, V).cuda()
    cache, tokens = run(d, (w1, w2))
    for r in range(n):
        for s in range(NSPEC):
            exp = 0 if r == 3 else min(peaks[(r + s) % len(peaks)])
            assert int(tokens[r, s]) == exp, (r, s, int(tokens[r, s]), exp)


@pytest.mark.parametrize("temp", [0.0, 1.0])
def test_degenerate_rows_stay_in_range(weights, temp):
    """Rows without a winner (all NaN, all -inf) give token 0 at every step; a NaN or +inf never wins, and the
    chain goes on (next step's embedding row exists): every token is in [0, V)."""
    require_gfx950()
    w1, _ = weights
    w2 = torch.zeros_like(weights[1])
    n = 40
    d = make(n, torch.Generator().manual_seed(9), inert=0, temps=(temp,))
    base = d["base"].view(n, NSPEC, V)
    nan, ninf, pinf = float("nan"), float("-inf"), float("inf")
    base[0] = nan
    base[1] = ninf
    base[2, :, :] = nan
    base[2, :, 70001] = 0.5
    base[2, :, 129279] = 0.25
    base[3] = ninf
    base[3, :, 100000] = -3.0
    base[4] = nan
    base[4, :, ::2] = ninf
    base[5] = -1.0
    base[5, :, 129279] = pinf
    base[5, :, 33] = pinf
    base[6, 2] = nan  # one step only
    cache, tokens = run(d, (w1, w2))
    torch.cuda.synchronize()
    assert int(tokens[:n].min()) >= 0 and int(tokens[:n].max()) < V
    assert bool((tokens[0] == 0).all()) and bool((tokens[1] == 0).all()) and bool((tokens[4] == 0).all())
    assert bool(((tokens[2] == 70001) | (tokens[2] == 129279)).all()) and bool((tokens[3] == 100000).all())
    if temp == 0.0:
        assert bool((tokens[2] == 70001).all())
    assert int(tokens[6, 2]) == 0
    if temp == 0.0:
        assert bool((tokens[5] == 0).all())  # +inf never wins: all other values equal, lowest column
    t2 = run(d, (w1, w2))[1]
    assert torch.equal(tokens, t2)


@pytest.mark.parametrize("n", [3, 40])
def test_whole_chain_against_torch(weights, n):
    require_gfx950()
    d = make(n, torch.Generator().manual_seed(n + 50))
    cache, tokens = run(d, weights)
    c2 = torch.zeros_like(cache)
    t2 = torch.empty_like(tokens)
    mc.markov_chain_reference(d["base"], weights[0], weights[1], d["ids"], d["anchor"], d["idx"], d["pos"], d["temp"],
                              d["seeds"], c2, t2)
    assert bool((tokens[:, 0] == t2[:, 0]).float().mean() > 0.95)
    assert bool((tokens == t2).float().mean() > 0.9)
    written = (d["idx"][:, 0] >= 0).nonzero().flatten()
    assert float((cache[d["idx"][written, 0], 0] != c2[d["idx"][written, 0], 0]).float().mean()) < 1e-4


def test_deterministic(weights):
    require_gfx950()
    d = make(40, torch.Generator().manual_seed(5))
    c1, t1 = run(d, weights)
    c2, t2 = run(d, weights)
    assert torch.equal(t1, t2) and torch.equal(c1.view(torch.int16), c2.view(torch.int16))


def test_graph_replay_new_inputs(weights):
    require_gfx950()
    gen = torch.Generator().manual_seed(9)
    d = make(48, gen)
    cache = torch.zeros(SLOTS, NSPEC, V, dtype=torch.bfloat16, device="cuda")
    tokens = torch.zeros(48, NSPEC, dtype=torch.int64, device="cuda")
    mc.warmup_markov_chain(torch.device("cuda"))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(d, weights, cache, tokens)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(d, weights, cache, tokens)
    for seed in (10, 11):
        e = make(48, torch.Generator().manual_seed(seed))
        for k in d:
            d[k].copy_(e[k])
        cache.zero_()
        g.replay()
        g.replay()
        torch.cuda.synchronize()
        c2, t2 = run(e, weights)
        assert torch.equal(tokens, t2) and torch.equal(cache.view(torch.int16), c2.view(torch.int16))
        check(e, weights, cache, tokens)


def test_matches_vllm_gumbel(weights):
    """The kernel's draws are vLLM's: ``gumbel_sample`` on the kernel's cache rows picks the same tokens."""
    require_gfx950()
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    n = 64
    d = make(n, torch.Generator().manual_seed(33), inert=9)
    cache, tokens = run(d, weights)
    idx = d["idx"].long()
    scratch = torch.zeros(SLOTS, NSPEC, V, dtype=torch.bfloat16, device="cuda")
    cols = torch.arange(NSPEC, dtype=torch.int32, device="cuda")
    for s in range(NSPEC):
        slot = idx[:, s].clamp(min=0)
        rows = cache[slot, s].clone()
        rows[idx[:, s] < 0] = weights[0][:1].new_zeros(1, V)  # an inert row: no cache row; its logits are not compared
        ids = gumbel_sample(rows, d["idx"][:, s].contiguous(), d["temp"], d["seeds"], d["pos"][:, s] - 1,
                            apply_temperature=True, is_drafting=True, logits_cache=scratch, logits_cache_col=cols[s])
        ok = idx[:, s] >= 0
        assert torch.equal(ids[ok], tokens[ok, s]), s


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels import warm_up

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    gen = torch.Generator().manual_seed(14)
    w1 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).cuda()
    w2 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).cuda()
    bad = []
    for n in (3, 1, 16, 17, 33, 64, 100, 160, 161):
        before = len(late)
        run(make(n, gen), (w1, w2))
        if len(late) > before:
            bad.append(n)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 1, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
