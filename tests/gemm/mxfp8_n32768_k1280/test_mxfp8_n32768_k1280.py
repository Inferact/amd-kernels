# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The N 32768 x K 1280 MXFP8 GEMM entries against ``reference.py`` at the padded decode row counts and between them,
bitwise repeats, graph replay on changed inputs, the 32 x 32 block-scale preshuffle, and the first-call warm-up
(nothing compiles afterwards)."""

import json
import os
import subprocess
import sys

import pytest
import torch

from amd_kernels.gemm.mxfp8_n32768_k1280 import K, N, kernel, mxfp8_linear, preshuffle_weight, serves
from amd_kernels.gemm.mxfp8_n32768_k1280.reference import COS_MIN, REL_L2_MAX, compare, dequantize_mxfp8, mxfp8_linear_reference, quantize_mxfp8

MS = [16, 32, 64, 128, 192, 256, 384, 512, 768]
BETWEEN = [1, 5, 17, 50, 100, 150, 200, 300, 500, 700]
BEYOND = [769, 1100]
_W = {}


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


def weights():
    if not _W:
        gen = torch.Generator(device="cuda").manual_seed(N + K)
        q, s = quantize_mxfp8((torch.randn(N, K, generator=gen, device="cuda") * 0.05).bfloat16())
        _W.update(q=q, s=s, packed=preshuffle_weight(q, s))
    return _W


def activation(m, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(1000 * m + seed)
    x = torch.randn(m, K, generator=gen, device="cuda")
    e = torch.randint(-4, 5, (m, K // 32, 1), generator=gen, device="cuda")
    return (x * torch.exp2(e.expand(m, K // 32, 32).reshape(m, K).float())).bfloat16()


def test_rows_cover():
    for table in [kernel.LINEAR]:
        bounds = [row[1] for row in table]
        assert bounds == sorted(set(bounds)) and bounds[-1] == max(MS)
    assert kernel.row(kernel.LINEAR, 1) is not None and kernel.row(kernel.LINEAR, max(MS) + 1) is None
    assert serves(1) in (True, False)


def test_block_scale_preshuffle():
    require_gfx950()
    w = weights()
    block = w["s"][::32].contiguous()  # a weight whose 32 rows share their scales
    q = w["q"]
    s_rows = block.repeat_interleave(32, dim=0)
    a = preshuffle_weight(q, block)
    b = preshuffle_weight(q, s_rows)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("m", MS + BETWEEN + BEYOND)
@torch.inference_mode()
def test_linear_matches_reference(m):
    require_gfx950()
    w = weights()
    x = activation(m)
    out = mxfp8_linear(x, *w["packed"])
    assert out.shape == (m, N) and out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    r = compare(out, mxfp8_linear_reference(x, w["q"], w["s"]))
    assert r["cos"] >= COS_MIN and r["rel_l2"] <= REL_L2_MAX, (m, r)


@pytest.mark.parametrize("m", [16, 64, 192, 768])
@torch.inference_mode()
def test_repeat_bitwise_and_zero_row(m):
    require_gfx950()
    w = weights()
    x = activation(m)
    x[0].zero_()
    a = mxfp8_linear(x, *w["packed"])
    b = mxfp8_linear(x, *w["packed"])
    assert torch.equal(a, b)
    assert not a[0].float().abs().max() > 0


@pytest.mark.parametrize("m", MS)
@torch.inference_mode()
def test_graph_replay_changed_inputs(m):
    require_gfx950()
    w = weights()
    x = activation(m, 1)
    mxfp8_linear(x, *w["packed"])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mxfp8_linear(x, *w["packed"])
    for seed in (2, 3):
        x.copy_(activation(m, seed))
        out.fill_(float("nan"))
        graph.replay()
        r = compare(out, mxfp8_linear_reference(x, w["q"], w["s"]))
        assert r["cos"] >= COS_MIN and r["rel_l2"] <= REL_L2_MAX, (m, seed, r)


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
    w = weights()
    bad = []
    for m in [1, 16, 33, 64, 100, 192, 300, 384, 512, 768]:
        before = len(late)
        mxfp8_linear(activation(m), *w["packed"])
        if len(late) > before and m != 1:
            bad.append(m)
    torch.cuda.synchronize()
    return {"late": bad, "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1500,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["warmed"] >= 1, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
