# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``grouped_gemm`` and ``grouped_gemm_mxfp8`` against ``reference.py`` at every row window (decode pad rows
16..768, odd rows, the window edges); bitwise repeats; the MXFP8 epilogue bit-equal to ``quantize_mxfp8`` of the same
config's bf16 output; a caller-owned ``out``; graph replay with new inputs; the first call's warm-up compiles every
variant and nothing compiles later."""

import json
import os
import subprocess
import sys

import pytest
import torch
from aiter.ops.shuffle import shuffle_weight
from amd_kernels.gemm import mxfp8_grouped_g8_n1024_k4096 as gg
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096 import host
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096.reference import (
    BLOCK, COS_MIN, G, K, N, QUANT_REL_L2_MAX, REL_L2_MAX, compare, dequantize_mxfp8, grouped_gemm_mxfp8_reference,
    grouped_gemm_reference, quantize_mxfp8,
)

ROWS = sorted({1, 7, 17, 100, 250, 500, 1000, host.MAX_M} | {b for b, _ in host._ROWS}
              | {b + 1 for b, _ in host._ROWS[:-1]})


def require_gfx950():
    if not gg.available():
        pytest.skip("requires gfx950 with FlyDSL")


def make_weight(gen):
    """Row-major codes [G * N, K] and 32 x 32 exponents [G, N // 32, K // 32] of a Gaussian weight."""
    w = torch.randn(G, N // BLOCK, BLOCK, K // BLOCK, BLOCK, generator=gen) * 0.05
    amax = w.abs().amax(dim=(2, 4)).clamp_min(1e-30)
    exponent = (amax / 448.0).log2().ceil().add(127).clamp(0, 254)
    codes = (w * torch.exp2(127 - exponent)[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return codes.view(G * N, K).cuda(), exponent.to(torch.uint8).cuda()


def make_x(m, gen, scale=1.0):
    """What the producer writes: quantized attention rows [M, G * K] with 1 x 32 exponents."""
    x = (torch.randn(m, G * K, generator=gen) * scale).to(torch.bfloat16).cuda()
    return quantize_mxfp8(x)


@pytest.fixture(scope="module")
def weight():
    require_gfx950()
    codes, exponents = make_weight(torch.Generator().manual_seed(5))
    return codes, exponents, shuffle_weight(codes, (16, 16)).view(G * N, K)


@pytest.mark.parametrize("m", ROWS)
def test_matches_reference(m, weight):
    codes, exponents, shuffled = weight
    x, xs = make_x(m, torch.Generator().manual_seed(m))
    got = gg.grouped_gemm(x, xs, shuffled, exponents)
    ref = grouped_gemm_reference(x, xs, codes, exponents)
    c = compare(got, ref)
    assert torch.isfinite(got.float()).all() and got.shape == (m, G * N)
    assert c["cos"] >= COS_MIN and c["rel_l2"] <= REL_L2_MAX, (m, c)
    again = gg.grouped_gemm(x, xs, shuffled, exponents)
    assert torch.equal(got.view(torch.int16), again.view(torch.int16)), m


@pytest.mark.parametrize("m", ROWS)
def test_mxfp8_epilogue(m, weight):
    codes, exponents, shuffled = weight
    x, xs = make_x(m, torch.Generator().manual_seed(100 + m), scale=3.0)
    q, qs = gg.grouped_gemm_mxfp8(x, xs, shuffled, exponents)
    assert q.dtype == torch.float8_e4m3fn and q.shape == (m, G * N) and qs.dtype == torch.uint8
    assert qs.shape == (m, G * N // BLOCK)
    rq, rs = grouped_gemm_mxfp8_reference(x, xs, codes, exponents)
    c = compare(dequantize_mxfp8(q.view(m, G * N), qs), dequantize_mxfp8(rq, rs))
    assert c["cos"] >= COS_MIN and c["rel_l2"] <= QUANT_REL_L2_MAX, (m, c)
    q2, qs2 = gg.grouped_gemm_mxfp8(x, xs, shuffled, exponents)
    assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(qs, qs2), m


@pytest.mark.parametrize("m", ROWS)
def test_epilogue_is_quantize_of_own_bf16(m, weight):
    """The same config with and without the epilogue, outputs pre-filled with NaN patterns: every code and byte is
    written and equals ``quantize_mxfp8`` of the bf16 tile."""
    codes, exponents, shuffled = weight
    x, xs = make_x(m, torch.Generator().manual_seed(200 + m), scale=3.0)
    bf16 = torch.full((m, G * N), float("nan"), dtype=torch.bfloat16, device="cuda")
    q = torch.full((m, G * N), 0xFF, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    qs = torch.full((m, G * N // BLOCK), 0xFF, dtype=torch.uint8, device="cuda")
    config = host.select(m)
    host._launch(config, x, xs, shuffled, exponents, bf16, bf16, bf16)
    host._launch(config._replace(mxfp8_out=True), x, xs, shuffled, exponents, qs, q, qs)
    rq, rs = quantize_mxfp8(bf16)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8)) and torch.equal(qs, rs), m


def test_out_arena(weight):
    codes, exponents, shuffled = weight
    x, xs = make_x(96, torch.Generator().manual_seed(7))
    arena = torch.full((96, G * N), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = gg.grouped_gemm(x, xs, shuffled, exponents, out=arena)
    assert out.data_ptr() == arena.data_ptr()
    assert torch.equal(out.view(torch.int16), gg.grouped_gemm(x, xs, shuffled, exponents).view(torch.int16))


def test_rejects_bad_inputs(weight):
    codes, exponents, shuffled = weight
    x, xs = make_x(16, torch.Generator().manual_seed(8))
    with pytest.raises(AssertionError):
        gg.grouped_gemm(x.view(torch.uint8), xs, shuffled, exponents)
    with pytest.raises(AssertionError):
        gg.grouped_gemm(x, xs[:, :-1].contiguous(), shuffled, exponents)
    with pytest.raises(AssertionError):
        gg.grouped_gemm(x, xs, shuffled, exponents, out=torch.empty(16, G * N, dtype=torch.float16, device="cuda"))
    big, bigs = make_x(gg.MAX_M + 1, torch.Generator().manual_seed(9))
    with pytest.raises(AssertionError):
        gg.grouped_gemm(big, bigs, shuffled, exponents)


def test_graph_replay_new_inputs(weight):
    codes, exponents, shuffled = weight
    gg.warmup_grouped_gemm(torch.device("cuda"))
    for m in (16, 192, 768):
        x, xs = make_x(m, torch.Generator().manual_seed(m))
        bx, bxs = x.clone(), xs.clone()
        for entry in (gg.grouped_gemm, gg.grouped_gemm_mxfp8):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                entry(bx, bxs, shuffled, exponents)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                res = entry(bx, bxs, shuffled, exponents)
            for seed in (1, 2):
                x2, xs2 = make_x(m, torch.Generator().manual_seed(seed * 1000 + m))
                bx.copy_(x2)
                bxs.copy_(xs2)
                g.replay()
                torch.cuda.synchronize()
                want = entry(x2, xs2, shuffled, exponents)
                got, ref = (res if isinstance(res, tuple) else (res,)), (want if isinstance(want, tuple) else (want,))
                for a, b in zip(got, ref):
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (m, entry.__name__)


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
    codes, exponents = make_weight(torch.Generator().manual_seed(5))
    shuffled = shuffle_weight(codes, (16, 16)).view(G * N, K)
    bad = []
    for m in (3, 1, 16, 17, 192, 768, 1024, 40):
        for entry in (gg.grouped_gemm, gg.grouped_gemm_mxfp8):
            before = len(late)
            x, xs = make_x(m, torch.Generator().manual_seed(m))
            entry(x, xs, shuffled, exponents)
            if len(late) > before:
                bad.append((m, entry.__name__))
    x, xs = make_x(24, torch.Generator().manual_seed(24))
    gg.grouped_gemm(x, xs, shuffled, exponents, out=torch.empty(24, G * N, dtype=torch.bfloat16, device="cuda"))
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 2, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
