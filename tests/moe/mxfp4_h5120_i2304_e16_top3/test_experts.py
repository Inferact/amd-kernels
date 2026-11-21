# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``mxfp4_experts`` against ``reference.py`` at the draft's token counts (5 rows per request, 16-640, plus the
smallest and largest dispatch tiers): cosine >= 0.99999 on the sum and on the slots, uniform and skewed routing
(an empty expert, one hot expert), two runs bit for bit, graph replay with new inputs, and the first call's warm-up
compiles every dispatch tier."""

import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e16_top3 import ExpertWeights, mxfp4_experts
from amd_kernels.moe.mxfp4_h5120_i2304_e16_top3.host import MAX_TOKENS, WARM_UP_TOKENS
from amd_kernels.moe.mxfp4_h5120_i2304_e16_top3.reference import E, TOPK, experts_reference

from .workload import make_activation, make_experts, make_routes, require_gfx950

COS = 0.99999


@pytest.fixture(scope="module")
def weights():
    require_gfx950()
    raw = make_experts()
    return raw, ExpertWeights.from_row_major(*raw)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))


@pytest.mark.parametrize("n", [1, 16, 33, 80, 160, 320, 640, 1024])
@pytest.mark.parametrize("dist", ["uniform", "skew"])
def test_matches_reference(weights, n, dist):
    raw, w = weights
    codes, scales = make_activation(n, seed=n)
    ids, wts = make_routes(n, dist, seed=n + 1)
    out = mxfp4_experts(codes, scales, w, wts, ids)
    slots = mxfp4_experts(codes, scales, w, wts, ids, return_slots=True)
    ref = experts_reference(codes, scales, *raw, wts, ids)
    ref_slots = experts_reference(codes, scales, *raw, wts, ids, return_slots=True)
    torch.cuda.synchronize()
    assert out.shape == (n, 5120) and slots.shape == (n, TOPK, 5120)
    assert cos(out, ref) >= COS and cos(slots, ref_slots) >= COS, (cos(out, ref), cos(slots, ref_slots))


def test_all_tokens_on_one_expert(weights):
    raw, w = weights
    n = 64
    codes, scales = make_activation(n, seed=3)
    ids = torch.tensor([[0, 5, 15]], dtype=torch.int32, device="cuda").repeat(n, 1)
    wts = torch.softmax(torch.randn(n, TOPK, device="cuda"), -1)
    out = mxfp4_experts(codes, scales, w, wts, ids)
    assert cos(out, experts_reference(codes, scales, *raw, wts, ids)) >= COS


def test_repeat_bitwise(weights):
    _, w = weights
    codes, scales = make_activation(160, seed=4)
    ids, wts = make_routes(160, "skew", seed=4)
    a = mxfp4_experts(codes, scales, w, wts, ids)
    b = mxfp4_experts(codes, scales, w, wts, ids)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


def test_graph_replay_new_inputs(weights):
    raw, w = weights
    n = 160
    codes, scales = make_activation(n, seed=5)
    ids, wts = make_routes(n, "uniform", seed=5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mxfp4_experts(codes, scales, w, wts, ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mxfp4_experts(codes, scales, w, wts, ids)
    for seed in (6, 7):
        c2, s2 = make_activation(n, seed=seed)
        i2, w2 = make_routes(n, "skew", seed=seed)
        codes.copy_(c2)
        scales.copy_(s2)
        ids.copy_(i2)
        wts.copy_(w2)
        graph.replay()
        torch.cuda.synchronize()
        assert cos(out, experts_reference(c2, s2, *raw, w2, i2)) >= COS


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
    w = ExpertWeights.from_row_major(*make_experts())
    bad = []
    for n in (80, 1, MAX_TOKENS, 17, 129, 700):
        before = len(late)
        codes, scales = make_activation(n, seed=n)
        ids, wts = make_routes(n, "uniform", seed=n)
        mxfp4_experts(codes, scales, w, wts, ids)
        if len(late) > before:
            bad.append(n)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_tier(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 1, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
