# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``rejection_sample`` against ``reference.py``: LM-like target rows (fp32 processed, or bf16 raw as the lm_head GEMM
writes them) and draft rows of three noise levels (accepted counts from 0 to 5), temperatures 0 / 0.7 / 1 / 1.3 mixed,
placeholder draft tokens, masked (-inf) columns, synthetic acceptance, request counts of the served batches; tokens
equal wherever the reference's margin exceeds 1e-3. Also: a request's result does not depend on the batch around it, two runs match bit for bit, the ticket
counters are zero after a launch, graph replay with new inputs, and the first call's warm-up compiles both modes."""

import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.sampling import rejection_standard_v129280_s5 as rs
from amd_kernels.sampling.rejection_standard_v129280_s5 import NSPEC, ROWS, V, rejection_sample, rejection_sample_reference

MARGIN = 1e-3


def require_gfx950():
    if not rs.available():
        pytest.skip("requires gfx950 with FlyDSL")


def make(b, gen, temps=(0.0, 0.7, 1.0, 1.3), noise=1.0, placeholders=0.0, extra_slots=3, bf16_target=False):
    """Inputs of ``b`` requests at random request states; draft tokens are drawn from the draft rows. ``bf16_target``:
    raw bf16 target logits (the kernel applies the temperature)."""
    r = b + extra_slots
    base = torch.randn(b * ROWS, V, generator=gen) * 2
    base.scatter_(1, torch.randint(0, V, (b * ROWS, 16), generator=gen), torch.rand(b * ROWS, 16, generator=gen) * 10 + 6)
    idx = torch.randperm(r, generator=gen)[:b].to(torch.int32)
    temp = torch.tensor(temps)[torch.randint(0, len(temps), (r,), generator=gen)].float()
    scale = temp[idx.long()].repeat_interleave(ROWS).unsqueeze(1)
    target = base.bfloat16() if bf16_target else torch.where(scale > 0, base / scale.clamp_min(1e-6), base)
    draft = torch.randn(r, NSPEC, V, generator=gen).to(torch.bfloat16)
    tokens = torch.full((b * ROWS,), -1, dtype=torch.int64)
    for i in range(b):
        s = int(idx[i])
        for j in range(NSPEC):
            d = (base[i * ROWS + j] + torch.randn(V, generator=gen) * noise).to(torch.bfloat16)
            draft[s, j] = d
            t = float(temp[s])
            tokens[i * ROWS + j + 1] = int(d.float().argmax()) if t == 0 else int(
                torch.multinomial(torch.softmax(d.float() / t, 0), 1, generator=gen))
    if placeholders:
        tokens[torch.rand(b * ROWS, generator=gen) < placeholders] = -1
    pos = torch.arange(b * ROWS, dtype=torch.int64) + int(torch.randint(0, 2**20, (1,), generator=gen))
    seed = torch.randint(-(2**62), 2**62, (r,), generator=gen)
    return tuple(x.cuda() for x in (target, draft, tokens, pos, idx, temp, seed))


def check(args, rates=None):
    sampled, num = rejection_sample(*args, rates)
    ref_s, ref_n, margin = rejection_sample_reference(*args, rates)
    torch.cuda.synchronize()
    sampled, num = sampled.cpu(), num.cpu()
    for i in range(sampled.shape[0]):
        if margin[i] < MARGIN:
            continue
        assert int(num[i]) == int(ref_n[i]) and torch.equal(sampled[i], ref_s[i]), (
            i, sampled[i].tolist(), ref_s[i].tolist(), float(margin[i]))
    return num


@pytest.mark.parametrize("bf16_target", [False, True])
@pytest.mark.parametrize("b", [1, 3, 8])
@pytest.mark.parametrize("noise", [0.3, 1.0, 3.0])
def test_matches_reference(b, noise, bf16_target):
    require_gfx950()
    gen = torch.Generator().manual_seed(b * 31 + int(noise * 10))
    num = check(make(b, gen, noise=noise, placeholders=0.05, bf16_target=bf16_target))
    assert num.min() >= 1 and num.max() <= ROWS


def test_accepted_counts_cover_the_range():
    require_gfx950()
    seen = set()
    for noise, seed in ((0.2, 1), (1.0, 2), (4.0, 3)):
        gen = torch.Generator().manual_seed(seed)
        seen |= set(check(make(12, gen, noise=noise, temps=(0.0, 1.0))).tolist())
    assert {1, 2, ROWS} <= seen, seen


@pytest.mark.parametrize("bf16_target", [False, True])
def test_served_batch_sizes(bf16_target):
    require_gfx950()
    for b in (32, 64):
        gen = torch.Generator().manual_seed(b)
        check(make(b, gen, noise=1.0, bf16_target=bf16_target))


@pytest.mark.parametrize("bf16_target", [False, True])
@pytest.mark.parametrize("rates", [[1.0] * 5, [0.0] * 5, [0.9, 0.8, 0.6, 0.4, 0.2], [0.5, 0.5, 0.5, 0.5, 0.5]])
def test_synthetic_acceptance(rates, bf16_target):
    require_gfx950()
    gen = torch.Generator().manual_seed(5)
    num = check(make(8, gen, bf16_target=bf16_target), torch.tensor(rates).cuda())
    if rates[0] == 0.0:
        assert bool((num == 1).all())
    if rates[-1] == 1.0:
        assert bool((num == ROWS).all())


@pytest.mark.parametrize("bf16_target", [False, True])
def test_masked_columns_and_ties(bf16_target):
    require_gfx950()
    gen = torch.Generator().manual_seed(6)
    args = list(make(6, gen, temps=(0.0, 1.0), bf16_target=bf16_target))
    target = args[0]
    target[:, ::2] = float("-inf")  # top-k style mask: the odd columns survive
    target[3] = float("-inf")
    target[3, 777] = 1.0
    target[4] = 0.0  # a flat row: greedy picks id 0, a draw picks anything
    args[0] = target
    for i in range(6 * ROWS):  # drafts on surviving columns
        if args[2][i] >= 0:
            args[2][i] |= 1
    check(tuple(args))


@pytest.mark.parametrize("bf16_target", [False, True])
def test_degenerate_rows_return_in_range_ids(bf16_target):
    """All-NaN and all -inf target / draft rows: every returned id is in [0, V) or the -1 pad."""
    require_gfx950()
    gen = torch.Generator().manual_seed(8)
    args = list(make(8, gen, temps=(0.0, 1.0), bf16_target=bf16_target))
    target, draft = args[0], args[1]
    target[0:ROWS] = float("nan")
    target[ROWS:2 * ROWS] = float("-inf")
    target[2 * ROWS + 1] = float("nan")
    draft[args[4][3].long()] = float("nan")
    draft[args[4][4].long()] = float("-inf")
    sampled, num = rejection_sample(*args, None)
    torch.cuda.synchronize()
    assert int(sampled.min()) >= -1 and int(sampled.max()) < V
    assert int(num.min()) >= 1 and int(num.max()) <= ROWS
    for i in range(sampled.shape[0]):
        n = int(num[i])
        assert bool((sampled[i, :n] >= 0).all()) and bool((sampled[i, n:] == -1).all())


def test_request_independent_of_batch():
    require_gfx950()
    gen = torch.Generator().manual_seed(7)
    args = make(16, gen, noise=0.6)
    full_s, full_n = rejection_sample(*args)
    target, draft, tokens, pos, idx, temp, seed = args
    for i in (0, 7, 15):
        one = (target[i * ROWS:(i + 1) * ROWS].contiguous(), draft, tokens[i * ROWS:(i + 1) * ROWS].contiguous(),
               pos[i * ROWS:(i + 1) * ROWS].contiguous(), idx[i:i + 1].contiguous(), temp, seed)
        s, n = rejection_sample(*one)
        assert torch.equal(s[0], full_s[i]) and int(n[0]) == int(full_n[i]), i


def test_repeat_bitwise_and_counters_reset():
    require_gfx950()
    gen = torch.Generator().manual_seed(8)
    args = make(16, gen, noise=0.6)
    a = rejection_sample(*args)
    b = rejection_sample(*args)
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    from amd_kernels.sampling.rejection_standard_v129280_s5 import host, kernel

    ws = host._workspace(args[0].device)
    counters = ws[kernel.STATS_BYTES + kernel.PART_BYTES:].view(torch.int32)
    assert not bool(counters.any())


def test_graph_replay_new_inputs():
    require_gfx950()
    gen = torch.Generator().manual_seed(9)
    bufs = make(8, gen, noise=0.6)
    rs.warmup_rejection_sample(bufs[0].device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        rejection_sample(*bufs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = rejection_sample(*bufs)
    for noise in (0.3, 2.0):
        new = make(8, gen, noise=noise)
        for buf, x in zip(bufs, new):
            buf.copy_(x)
        g.replay()
        torch.cuda.synchronize()
        want = rejection_sample(*new)
        assert torch.equal(out[0], want[0]) and torch.equal(out[1], want[1])


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
    rates = torch.full((NSPEC,), 0.5).cuda()
    for b, rates_, bf16 in ((3, None, False), (1, None, True), (16, None, False), (5, rates, True), (2, rates, False)):
        before = len(late)
        rejection_sample(*make(b, gen, bf16_target=bf16), rates_)
        if len(late) > before:
            bad.append(b)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=900,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 4, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
