# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``gumbel_partials_bias`` against ``reference.py``: the bf16 cache rows exactly (slot -1 rows write nothing);
temperature-0 rows (no noise) exactly by Triton's block tree, with NaN (one, many, whole blocks, all, inf - inf),
+-inf, all -inf but one, exact ties and -0.0 / +0.0; noised rows (T 0.7 / 1.0 / 1.3 / 2.0 / 1e-3) within 1e-5 of the
fp64 noise reference with the same argmax wherever the block's top two differ by more than 1e-4 + 1e-6 |top|; both
tilings; 1-65 rows, every step column; the position offset; vocab ids 0 / 50015 / 50016 / 200063; graph replay with
new inputs; bitwise determinism; no compile after the warm-up. Bit-equality with Triton's kernel is checked outside
these tests (no vLLM here)."""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.sampling import gumbel_partials_bias_bf16_v200064 as gp
from amd_kernels.sampling.gumbel_partials_bias_bf16_v200064 import host
from amd_kernels.sampling.gumbel_partials_bias_bf16_v200064.reference import (
    NB,
    V,
    block_tree_reference,
    cache_rows_reference,
    noise_approx,
    uniform_reference,
)


def require_gfx950():
    if not gp.available():
        pytest.skip("requires gfx950 with FlyDSL")


def make(n, gen, temps, nan_rows=False):
    base = (torch.randn(n, V, generator=gen) * 3).to(torch.bfloat16)
    bias = (torch.randn(n + 2, V, generator=gen) * 0.5).to(torch.bfloat16)
    brows = torch.randperm(n + 2, generator=gen)[:n]
    if nan_rows:
        for r in range(n):
            k = r % 10
            b = brows[r]
            if k == 1:
                base[r, 5::997] = float("nan")
            elif k == 2:
                base[r, [0, 1023, 1024, 200063]] = float("nan")
            elif k == 3:
                base[r, 4096:5120] = float("nan")
            elif k == 4:
                base[r] = float("nan")
            elif k == 5:
                base[r, ::3] = float("inf")
                bias[b, ::7] = float("-inf")  # inf + -inf = NaN
            elif k == 6:
                base[r] = float("-inf")
                base[r, 77777] = 1.0
            elif k == 7:  # exact ties and signed zeros
                base[r, :] = -1.0
                bias[b, :] = 0.0
                base[r, [3, 50015, 50016, 200063]] = 0.0
                base[r, [9, 1500]] = -0.0
            elif k == 8:
                base[r, torch.rand(V, generator=gen) < 0.3] = float("nan")
            elif k == 9:
                base[r, [0, 50015, 50016, 200063]] = 100.0
    slots = (torch.randperm(256, generator=gen)[:n] if n <= 256 else torch.arange(n) % 256).to(torch.int32)
    if n > 2:
        slots[1] = -1
    temp = torch.tensor([temps[i % len(temps)] for i in range(256)], dtype=torch.float32)
    seeds = torch.randint(-(2**62), 2**62, (256,), generator=gen)
    pos = 131072 + torch.randint(0, 1 << 20, (n,), generator=gen)
    d = dict(base=base, bias=bias, brows=brows, idx=slots, temp=temp, seeds=seeds, pos=pos)
    return {k: v.cuda() for k, v in d.items()}


def run(d, step, tiles=None, cache=None):
    n = d["base"].shape[0]
    cache = torch.zeros(256, 7, V, dtype=torch.bfloat16, device="cuda") if cache is None else cache
    lm = torch.full((n, NB), 7.0, device="cuda")
    la = torch.full((n, NB), -7, dtype=torch.int64, device="cuda")
    gp.gumbel_partials_bias(d["base"], d["idx"], d["temp"], d["seeds"], d["pos"], cache, step, d["bias"], d["brows"],
                            lm, la, tiles)
    return cache, lm, la


def reference_values(d, step):
    """fp32 [n, V] noised values (approximate noise) and the T-0 mask."""
    y = cache_rows_reference(d["base"], d["bias"], d["brows"]).float()
    valid = d["idx"] >= 0
    slot = d["idx"].clamp(min=0).long()
    t = torch.where(valid, d["temp"][slot], torch.zeros_like(d["temp"][slot]))
    cols = torch.arange(V, device="cuda")
    out = y.clone()
    for r in range(y.shape[0]):
        tr = float(t[r])
        if tr == 0.0:
            continue
        v = y[r] / t[r] if tr != 1.0 else y[r]
        u = uniform_reference(int(d["seeds"][slot[r]]), step, int(d["pos"][r]), cols)
        out[r] = v + noise_approx(u)
    return out, t == 0.0


def check_cache(d, step, cache):
    exp = cache_rows_reference(d["base"], d["bias"], d["brows"])
    ref = torch.zeros_like(cache)
    for r in range(exp.shape[0]):
        s = int(d["idx"][r])
        if s >= 0:
            ref[s, step] = exp[r]
    assert torch.equal(ref.view(torch.int16), cache.view(torch.int16))


@pytest.mark.parametrize("n", [1, 3, 8, 17, 64])
@pytest.mark.parametrize("tiles", [1, 2])
def test_exact_tree_without_noise(n, tiles):
    require_gfx950()
    gen = torch.Generator().manual_seed(100 * n + tiles)
    d = make(n, gen, temps=(0.0,), nan_rows=True)
    for step in (0, 6):
        cache, lm, la = run(d, step, tiles)
        check_cache(d, step, cache)
        v = cache_rows_reference(d["base"], d["bias"], d["brows"]).float()
        rm, ra = block_tree_reference(v)
        assert torch.equal(rm.view(torch.int32), lm.view(torch.int32))
        assert torch.equal(ra, la)


@pytest.mark.parametrize("n", [1, 4, 16, 33, 65])
@pytest.mark.parametrize("tiles", [1, 2])
def test_noised_rows(n, tiles):
    require_gfx950()
    gen = torch.Generator().manual_seed(7 * n + tiles)
    d = make(n, gen, temps=(1.0, 0.7, 1.3, 2.0, 1e-3, 0.0))
    for step in (1, 5):
        cache, lm, la = run(d, step, tiles)
        check_cache(d, step, cache)
        vals, _ = reference_values(d, step)
        rm, ra = block_tree_reference(vals)
        torch.testing.assert_close(lm, rm, rtol=0, atol=1e-5 * max(1.0, float(rm.abs().max())))
        full = torch.full((n, NB * 1024), float("-inf"), device="cuda")
        full[:, :V] = vals
        top2 = full.view(n, NB, 1024).topk(2, dim=2).values
        # our noise is within a few ulps of the fp64 one: compare ids where the top two are further apart
        clear = (top2[..., 0] - top2[..., 1]) > 1e-4 + 1e-6 * top2[..., 0].abs()
        assert clear.float().mean() > 0.9
        assert torch.equal(ra[clear], la[clear])


def test_tilings_agree_and_deterministic():
    require_gfx950()
    gen = torch.Generator().manual_seed(3)
    d = make(40, gen, temps=(1.0, 0.7, 0.0), nan_rows=True)
    outs = [run(d, 4, t) for t in (1, 2, 2)]
    for c, m, a in outs[1:]:
        assert torch.equal(outs[0][0].view(torch.int16), c.view(torch.int16))
        assert torch.equal(outs[0][1].view(torch.int32), m.view(torch.int32))
        assert torch.equal(outs[0][2], a)


def test_position_offset():
    """pos_offset=-1 on pos + 1 (vLLM's sample_pos) gives the bits of pos (vLLM's draw_pos = sample_pos - 1)."""
    require_gfx950()
    gen = torch.Generator().manual_seed(6)
    d = make(20, gen, temps=(1.0, 0.7, 1.3))
    c0, m0, a0 = run(d, 3)
    e = dict(d, pos=d["pos"] + 1)
    n = d["base"].shape[0]
    c1 = torch.zeros_like(c0)
    m1 = torch.empty_like(m0)
    a1 = torch.empty_like(a0)
    gp.gumbel_partials_bias(e["base"], e["idx"], e["temp"], e["seeds"], e["pos"], c1, 3, e["bias"], e["brows"], m1,
                            a1, pos_offset=-1)
    assert torch.equal(c0.view(torch.int16), c1.view(torch.int16))
    assert torch.equal(m0.view(torch.int32), m1.view(torch.int32)) and torch.equal(a0, a1)
    assert n == 20


def test_graph_replay_new_inputs():
    require_gfx950()
    gen = torch.Generator().manual_seed(4)
    d = make(24, gen, temps=(1.0, 0.7))
    cache = torch.zeros(256, 7, V, dtype=torch.bfloat16, device="cuda")
    lm = torch.empty(24, NB, device="cuda")
    la = torch.empty(24, NB, dtype=torch.int64, device="cuda")
    host.warmup_gumbel_partials_bias(torch.device("cuda"))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        gp.gumbel_partials_bias(d["base"], d["idx"], d["temp"], d["seeds"], d["pos"], cache, 2, d["bias"],
                                d["brows"], lm, la)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gp.gumbel_partials_bias(d["base"], d["idx"], d["temp"], d["seeds"], d["pos"], cache, 2, d["bias"],
                                d["brows"], lm, la)
    e = make(24, gen, temps=(1.3, 1.0, 0.0), nan_rows=True)
    for k in d:
        d[k].copy_(e[k])
    cache.zero_()
    g.replay()
    torch.cuda.synchronize()
    c2, m2, a2 = run(e, 2)
    assert torch.equal(cache.view(torch.int16), c2.view(torch.int16))
    assert torch.equal(lm.view(torch.int32), m2.view(torch.int32))
    assert torch.equal(la, a2)


def test_no_compile_after_warmup(monkeypatch):
    require_gfx950()
    host._kernel_exe.cache_clear()
    host.warmup_gumbel_partials_bias(torch.device("cuda"))
    compiled = []
    real = flyc.compile
    monkeypatch.setattr(flyc, "compile", lambda *a, **k: compiled.append(a[0]) or real(*a, **k))
    gen = torch.Generator().manual_seed(5)
    d = make(300, gen, temps=(1.0, 0.0, 0.7))
    cache = torch.zeros(256, 7, V, dtype=torch.bfloat16, device="cuda")
    lm = torch.empty(300, NB, device="cuda")
    la = torch.empty(300, NB, dtype=torch.int64, device="cuda")
    for rows in list(range(1, 70)) + [100, 255, 256, 300]:
        dd = {k: (v[:rows] if k in ("base", "brows", "idx", "pos") else v) for k, v in d.items()}
        gp.gumbel_partials_bias(dd["base"], dd["idx"], dd["temp"], dd["seeds"], dd["pos"], cache, rows % 7,
                                dd["bias"], dd["brows"], lm[:rows], la[:rows])
    torch.cuda.synchronize()
    assert compiled == []
