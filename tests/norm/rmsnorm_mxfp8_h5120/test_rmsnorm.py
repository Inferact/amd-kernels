# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``rmsnorm_mxfp8`` against ``reference.py`` at the draft's row counts (the context rows, 1-1024 and a prefill
chunk) and odd ones: dequantized rows' cosine >= 0.99999, scale bytes and (all but ~1e-5 of) codes equal, extreme rows
(zero, overflow, underflowing squares, one hot column), row strides, caller buffers, graph replay with poisoned outputs, two runs
bit for bit, and the first call's warm-up compiles the kernel."""

import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.norm import rmsnorm_mxfp8_h5120 as nm
from amd_kernels.norm.rmsnorm_mxfp8_h5120 import dequantize_mxfp8, rmsnorm_mxfp8, rmsnorm_mxfp8_reference

H = 5120
EPS = 1e-20
ROWS = (1, 2, 7, 16, 17, 32, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 4097)


def require_gfx950():
    if not nm.available():
        pytest.skip("requires gfx950 with FlyDSL")


def make(m, seed, scale=3.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, H, generator=g, device="cuda") * scale).bfloat16()
    w = (torch.rand(H, generator=g, device="cuda") + 0.5).bfloat16()
    return x, w


def check(codes, scale, x, w):
    rc, rs = rmsnorm_mxfp8_reference(x, w, EPS)
    torch.cuda.synchronize()
    a, b = dequantize_mxfp8(codes, scale), dequantize_mxfp8(rc, rs)
    live = b.abs().amax(-1) > 0
    a, b = a.double(), b.double()
    cos = (a[live] * b[live]).sum(-1) / (a[live].norm(dim=-1) * b[live].norm(dim=-1))  # no eps: rows can be tiny
    assert float(cos.min()) >= 0.99999, float(cos.min())
    assert bool((a[~live] == 0).all())
    assert (codes.view(torch.uint8) != rc.view(torch.uint8)).float().mean().item() <= 2e-5
    assert (scale != rs).float().mean().item() <= 1e-5


@pytest.mark.parametrize("m", ROWS)
def test_matches_reference(m):
    require_gfx950()
    x, w = make(m, m)
    codes, scale = rmsnorm_mxfp8(x, w, EPS)
    assert codes.shape == (m, H) and codes.dtype == torch.float8_e4m3fn and scale.shape == (m, H // 32)
    check(codes, scale, x, w)
    c2, s2 = rmsnorm_mxfp8(x, w, EPS)
    assert torch.equal(codes.view(torch.uint8), c2.view(torch.uint8)) and torch.equal(scale, s2)


def test_extreme_rows():
    require_gfx950()
    x, w = make(8, 5)
    x[0] = 0.0
    x[1] = 3e38  # near the bf16 maximum
    x[2] = 1e-30  # the square underflows: the norm is eps-limited
    x[3] = 0.0
    x[3, 4100] = 5.0
    x[4, ::2] = 1e-3
    x[5] = 60000.0
    codes, scale = rmsnorm_mxfp8(x, w, EPS)
    check(codes, scale, x.clone(), w)


def test_row_stride_and_caller_buffers():
    require_gfx950()
    m = 96
    x, w = make(m, 8)
    wide = torch.zeros((m, H + 64), device="cuda", dtype=torch.bfloat16)
    wide[:, :H] = x
    codes = torch.empty((m, H), dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.empty((m, H // 32), dtype=torch.uint8, device="cuda")
    out = rmsnorm_mxfp8(wide[:, :H], w, EPS, codes=codes, scale=scale)
    assert out[0] is codes and out[1] is scale
    check(codes, scale, x, w)


def test_graph_replay_with_poisoned_outputs():
    require_gfx950()
    m = 160
    x, w = make(m, 9)
    codes = torch.empty((m, H), dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.empty((m, H // 32), dtype=torch.uint8, device="cuda")
    rmsnorm_mxfp8(x, w, EPS, codes=codes, scale=scale)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rmsnorm_mxfp8(x, w, EPS, codes=codes, scale=scale)
    for seed in (10, 11):
        x2, _ = make(m, seed, scale=2.0)
        x.copy_(x2)
        codes.view(torch.uint8).fill_(7)
        scale.fill_(7)
        g.replay()
        torch.cuda.synchronize()
        check(codes, scale, x2, w)


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
    bad = []
    for m in (160, 1, 1024, 33):
        before = len(late)
        x, w = make(m, m)
        rmsnorm_mxfp8(x, w, EPS)
        if len(late) > before:
            bad.append(m)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_the_kernel(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=900,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 1, result


if __name__ == "__main__":
    print(json.dumps(scenario()))


@pytest.mark.parametrize("m", (5, 160, 640))
def test_codes_feed_the_mxfp8_gemm(m):
    """The hand-off: this entry's codes and row-major scales go straight into the common MXFP8 GEMM (a draft layer's
    ``wkv`` shape, N 512) and give the GEMM of the reference's own codes, with no quantizer between them."""
    require_gfx950()
    from amd_kernels.gemm.mxfp8_common import gemm
    from amd_kernels.gemm.mxfp8_common.reference import compare, mxfp8_gemm_reference, quantize_mxfp8

    x, w = make(m, 20 + m)
    g = torch.Generator(device="cuda").manual_seed(3)
    wq, ws = quantize_mxfp8((torch.randn(512, H, generator=g, device="cuda") * 0.05).bfloat16())
    wp, wsp = gemm.preshuffle_mxfp8_weight(wq, ws)
    codes, scale = rmsnorm_mxfp8(x, w, EPS)
    out = gemm.mxfp8_gemm(codes, scale, wp, wsp, torch.bfloat16)
    rc, rs = rmsnorm_mxfp8_reference(x, w, EPS)
    ref = mxfp8_gemm_reference(rc, rs, wq, ws)
    torch.cuda.synchronize()
    c = compare(out, ref)
    assert c["cos"] >= 0.99999 and c["rel_l2"] <= 1e-3, c
