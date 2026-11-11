# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``top_p_mask`` against ``reference.py``, bit for bit: LM-like rows (bf16 logits at temperatures 0.6-1.5),
integer ties (the boundary duplicates), -inf columns, flat rows (kept), p 1 rows (untouched), per-row p, both slice
counts (1-8 and 9-64 rows); the same rows at every row count of a slice count (workspace and counters reused);
graph replay with new inputs; the first call's warm-up compiles both variants and nothing compiles later.
Bit-equality with vLLM's Triton chain needs vLLM, so it is checked outside these tests."""

import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.sampling import top_p_mask_f32_v200064 as tp
from amd_kernels.sampling.top_p_mask_f32_v200064 import V, top_p_mask, top_p_mask_reference


def require_gfx950():
    if not tp.available():
        pytest.skip("requires gfx950 with FlyDSL")


def lm_rows(n, gen, temp=1.0):
    """bf16 logits with a heavy head (as an LM's): a few large values over a Gaussian body."""
    x = torch.randn(n, V, generator=gen) * 2.0
    head = torch.randint(0, V, (n, 64), generator=gen)
    x.scatter_(1, head, torch.rand(n, 64, generator=gen) * 12 + 6)
    return (x.to(torch.bfloat16).float() / temp).cuda()


def case(name, n, gen):
    p = torch.full((n,), 0.95, device="cuda")
    if name == "lm":
        return lm_rows(n, gen), p
    if name == "cold":
        return lm_rows(n, gen, 0.6), p
    if name == "hot":
        return lm_rows(n, gen, 1.5), torch.rand(n, generator=gen).cuda() * 0.5 + 0.5
    if name == "ties":
        return torch.randint(-3, 4, (n, V), generator=gen).float().cuda(), torch.full((n,), 0.9, device="cuda")
    if name == "masked":
        x = lm_rows(n, gen)
        x[:, ::3] = float("-inf")
        x[: max(1, n // 4), 1::3] = float("-inf")
        return x, p
    if name == "mixed":  # p 1 rows, a flat row, a one-finite row
        x = lm_rows(n, gen)
        p[::2] = 1.0
        x[-1] = 0.0
        if n > 2:
            x[1] = float("-inf")
            x[1, 1234] = 3.0
        return x, p
    raise ValueError(name)


def same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize("n", [1, 3, 8, 9, 16])
@pytest.mark.parametrize("name", ["lm", "cold", "hot", "ties", "masked", "mixed"])
def test_matches_reference(n, name):
    require_gfx950()
    gen = torch.Generator().manual_seed(n * 7 + len(name))
    x, p = case(name, n, gen)
    y = top_p_mask(x.clone(), p)
    r = top_p_mask_reference(x, p)
    torch.cuda.synchronize()
    assert same(y, r), (name, n, (y.view(torch.int32) != r.view(torch.int32)).any(1).nonzero().flatten().tolist())
    keep = p >= 1
    assert same(y[keep], x[keep])


def test_rows_independent_of_row_count():
    require_gfx950()
    gen = torch.Generator().manual_seed(11)
    x = torch.cat([lm_rows(48, gen), case("ties", 8, gen)[0], case("masked", 8, gen)[0]])
    p = torch.rand(64, generator=gen).cuda() * 0.4 + 0.6
    full = {s: top_p_mask(x[:n].clone(), p[:n]) for s, n in ((32, 8), (16, 64))}
    for n in range(1, 65):
        s = tp.splits_for(n)
        y = top_p_mask(x[:n].clone(), p[:n])
        assert same(y, full[s][:n]), n
    a = top_p_mask(x.clone(), p)
    b = top_p_mask(x.clone(), p)
    assert same(a, b) and same(a, full[16])


def test_row_stride():
    require_gfx950()
    gen = torch.Generator().manual_seed(12)
    x, p = case("lm", 12, gen)
    big = torch.full((12, V + 64), 7.0, device="cuda")
    big[:, :V] = x
    top_p_mask(big[:, :V], p)
    assert same(big[:, :V], top_p_mask(x.clone(), p)) and bool((big[:, V:] == 7.0).all())


def test_graph_replay_new_inputs():
    require_gfx950()
    gen = torch.Generator().manual_seed(13)
    for n in (5, 24):
        x, p = case("lm", n, gen)
        buf = x.clone()
        tp.warmup_top_p_mask(torch.device("cuda"))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            top_p_mask(buf, p)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            top_p_mask(buf, p)
        for name in ("hot", "ties", "mixed"):
            x2, p2 = case(name, n, gen)
            buf.copy_(x2)
            p.copy_(p2)
            g.replay()
            torch.cuda.synchronize()
            assert same(buf, top_p_mask(x2.clone(), p2)), (n, name)


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
    bad = []
    for n in (3, 1, 8, 9, 64, 33):
        before = len(late)
        x, p = case("lm", n, gen)
        top_p_mask(x, p)
        if len(late) > before:
            bad.append(n)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=900,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 1, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
