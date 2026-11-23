# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``main_proj`` against ``reference.py`` at every row tier edge, the draft's served counts (5 and 6 rows per request:
160 / 192 / 320 / 384 / 640 / 768) and odd counts: cosine >= 0.99999 and relative L2 <= 1e-3 with 32 x 32 block scales
of random exponents; two runs bit for bit; graph replay with new inputs; the first call's warm-up compiles every tier."""

import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.gemm import mxfp8_n5120_k15360 as mp
from amd_kernels.gemm.mxfp8_n5120_k15360 import DISPATCH, K, N, MainProjWeights, main_proj, main_proj_reference
from amd_kernels.gemm.mxfp8_n5120_k15360.reference import COS_MIN, REL_L2_MAX, compare, quantize_mxfp8


def require_gfx950():
    if not mp.available():
        pytest.skip("requires gfx950 with FlyDSL")


@pytest.fixture(scope="module")
def weights():
    require_gfx950()
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(N, K, generator=g, device="cuda") * 0.05).bfloat16()
    codes, _ = quantize_mxfp8(w)
    block = torch.randint(118, 126, (N // 32, K // 32), generator=g, device="cuda", dtype=torch.uint8)
    # codes at their own per-row scales, replaced by one scale per 32 x 32 block: the codes stay valid values
    return codes, block, MainProjWeights.from_block_scales(codes, block)


def activation(m, seed=1, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed + m)
    return (torch.randn(m, K, generator=g, device="cuda") * scale).bfloat16()


ROWS = sorted({1, 5, 16, 17, 33, 100, 160, 192, 257, 320, 384, 500, 640, 700, 768, 1000, 1025, 2048}
              | {bound for bound, _ in DISPATCH[:-2]} | {bound + 1 for bound, _ in DISPATCH[:-2]})


@pytest.mark.parametrize("m", ROWS)
def test_matches_reference(weights, m):
    codes, block, w = weights
    x = activation(m)
    out = main_proj(x, w)
    ref = main_proj_reference(x, codes, block)
    torch.cuda.synchronize()
    assert out.shape == (m, N) and out.dtype == torch.bfloat16 and bool(torch.isfinite(out).all())
    c = compare(out, ref)
    assert c["cos"] >= COS_MIN and c["rel_l2"] <= REL_L2_MAX, (m, c)


@pytest.mark.parametrize("m", [4096, 4097, 8192])
def test_prefill_rows(weights, m):
    codes, block, w = weights
    x = activation(m, scale=3.0)
    c = compare(main_proj(x, w), main_proj_reference(x, codes, block))
    assert c["cos"] >= COS_MIN and c["rel_l2"] <= REL_L2_MAX, (m, c)


def test_repeat_bitwise(weights):
    _, _, w = weights
    for m in (160, 640, 2048):
        x = activation(m)
        a, b = main_proj(x, w), main_proj(x, w)
        torch.cuda.synchronize()
        assert torch.equal(a, b), m


def test_graph_replay_new_inputs(weights):
    codes, block, w = weights
    x = activation(160)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        main_proj(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = main_proj(x, w)
    for seed in (7, 8):
        x2 = activation(160, seed=seed, scale=2.0)
        x.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        c = compare(out, main_proj_reference(x2, codes, block))
        assert c["cos"] >= COS_MIN and c["rel_l2"] <= REL_L2_MAX, c


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
    codes = torch.zeros(N, K, dtype=torch.float8_e4m3fn, device="cuda")
    w = MainProjWeights.from_block_scales(codes, torch.full((N // 32, K // 32), 127, dtype=torch.uint8, device="cuda"))
    bad = []
    for m in (160, 1, 5000, 640, 33, 2048):
        before = len(late)
        main_proj(torch.zeros(m, K, dtype=torch.bfloat16, device="cuda"), w)
        if len(late) > before:
            bad.append(m)
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


@pytest.mark.parametrize("m", (5, 160, 640))
def test_output_feeds_the_norm_and_quant(weights, m):
    """The hand-off: ``main_proj``'s bf16 output goes straight into ``norm/rmsnorm_mxfp8_h5120`` (``main_norm`` and the
    quantization of the draft layers' ``wkv`` GEMMs in one launch), whose codes and scales equal the reference
    chain's."""
    from amd_kernels.norm.rmsnorm_mxfp8_h5120 import dequantize_mxfp8, rmsnorm_mxfp8, rmsnorm_mxfp8_reference

    codes, block, w = weights
    x = activation(m, seed=4)
    g = torch.Generator(device="cuda").manual_seed(5)
    nw = (torch.rand(N, generator=g, device="cuda") + 0.5).bfloat16()
    out = main_proj(x, w)
    c, s = rmsnorm_mxfp8(out, nw, 1e-20)
    rc, rs = rmsnorm_mxfp8_reference(main_proj_reference(x, codes, block), nw, 1e-20)
    torch.cuda.synchronize()
    a, b = dequantize_mxfp8(c, s), dequantize_mxfp8(rc, rs)
    assert float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min()) >= 0.9999
