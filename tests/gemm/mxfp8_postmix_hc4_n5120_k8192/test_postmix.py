# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The wo_b GEMM with the mHC post-mix epilogue against ``reference.py`` and, bit for bit, against the separate chain
(the same GEMM sums, then the seam kernel's post-mix), at every dispatch row and between them; the gate-partial
variant (x1, records, aux mean) against torch; every table config including split K; bitwise repeats; graph replay with
poisoned outputs; the first call's warm-up compiles every variant and nothing compiles later."""

import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.gemm import mxfp8_postmix_hc4_n5120_k8192 as pm
from amd_kernels.gemm.mxfp8_postmix_hc4_n5120_k8192 import host
from amd_kernels.gemm.mxfp8_postmix_hc4_n5120_k8192.reference import (
    COS_MIN, K, N, collapse_reference, partial_sums_reference, postmix_gemm_reference, quantize_mxfp8,
)
from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120 import mhc_post_pre_rmsnorm

H = N
KW = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult=1.5, norm_eps=1e-6)
ROWS = sorted({1, 7, 17, 100, 250, 500, host.MAX_ROWS} | {b for b, _ in host._ROWS} | {b + 1 for b, _ in host._ROWS[:-1]}
              | {b for b, _ in host._GATE_ROWS} | {b + 1 for b, _ in host._GATE_ROWS[:-1]})


def require_gfx950():
    if not pm.available():
        pytest.skip("requires gfx950 with FlyDSL")


def cos(a, b):
    a, b = a.float().flatten(1).double(), b.float().flatten(1).double()
    return (((a * b).sum(1) + 1e-30) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)).min().item()


@pytest.fixture(scope="module")
def weight():
    require_gfx950()
    gen = torch.Generator(device="cuda").manual_seed(N + K)
    codes, exps = quantize_mxfp8((torch.randn(N, K, generator=gen, device="cuda") * 0.05).bfloat16())
    return codes, exps, pm.preshuffle_weight(codes, exps)


@pytest.fixture(scope="module")
def fn():
    gen = torch.Generator(device="cuda").manual_seed(3)
    f = torch.randn(24, 4 * H, generator=gen, device="cuda") * 0.02
    return f, pm.pack_gate_weight(f)


def make(m, seed=0):
    g = torch.Generator(device="cuda").manual_seed(1000 * m + seed)
    x = torch.randn(m, K, generator=g, device="cuda")
    e = torch.randint(-4, 5, (m, K // 32, 1), generator=g, device="cuda")
    codes, exps = quantize_mxfp8((x * torch.exp2(e.expand(m, K // 32, 32).reshape(m, K).float())).bfloat16())
    scale = torch.exp2(torch.randint(-3, 4, (m, 1, 1), device="cuda", generator=g).float())
    return dict(
        codes=codes, scales=exps, residual=(torch.randn(m, 4, H, generator=g, device="cuda") * scale).bfloat16(),
        post=torch.rand(m, 4, 1, device="cuda", generator=g), pre=torch.rand(m, 4, device="cuda", generator=g),
        comb=torch.softmax(torch.randn(m, 4, 4, generator=g, device="cuda"), -1),
    )


def run(d, w, **kw):
    return pm.postmix_gemm(d["codes"], d["scales"], w[2][0], w[2][1], d["residual"], d["post"], d["comb"], **kw)


def y_of(d, w, config=None):
    """The GEMM's own bf16 y: post = (1, 0, 0, 0), comb = 0, R = 0 gives R'_0 = bf16(y) exactly."""
    m = d["codes"].shape[0]
    post = torch.zeros(m, 4, 1, device="cuda")
    post[:, 0] = 1
    out = pm.postmix_gemm(d["codes"], d["scales"], w[2][0], w[2][1], torch.zeros_like(d["residual"]), post,
                          torch.zeros(m, 4, 4, device="cuda"), config=config)
    return out[:, 0].contiguous()


@pytest.mark.parametrize("m", ROWS)
def test_matches_reference_and_separate_chain(m, weight):
    d = make(m)
    out = run(d, weight)
    ref = postmix_gemm_reference(d["codes"], d["scales"], weight[0], weight[1], d["residual"], d["post"], d["comb"])
    assert torch.isfinite(out.float()).all() and cos(out, ref) >= COS_MIN
    y = y_of(d, weight)
    chain = mhc_post_pre_rmsnorm(d["residual"], torch.zeros(4, 160, 3, 4, 16, 8, dtype=torch.bfloat16, device="cuda"),
                                 torch.ones(3, device="cuda"), torch.zeros(24, device="cuda"),
                                 torch.ones(H, dtype=torch.bfloat16, device="cuda"), sublayer_out=y,
                                 post_mix=d["post"], comb_mix=d["comb"], **KW)[0]
    assert torch.equal(out, chain)
    assert torch.equal(run(d, weight), out)


@pytest.mark.parametrize("m", ROWS)
def test_gates_variant(m, weight, fn):
    d = make(m)
    out = run(d, weight, config=host.select_gates(m))
    for aux in (False, True):
        r, x1, gp, mean = pm.postmix_gates_gemm(d["codes"], d["scales"], weight[2][0], weight[2][1], d["residual"],
                                                d["post"], d["comb"], d["pre"], fn[1], aux_mean=aux)
        assert torch.equal(r, out)
        assert (mean is None) != aux
        x1ref = collapse_reference(r, d["pre"])
        assert cos(x1, x1ref) >= COS_MIN and (x1 != x1ref).float().mean() < 1e-4
        tot = gp.double().sum(1)
        ref = partial_sums_reference(r, x1, fn[0]).double()
        assert ((tot[:, :24] - ref[:, :24]).abs().max() / ref[:, :24].abs().max()).item() < 1e-5
        assert ((tot[:, 24:26] - ref[:, 24:26]).abs() / ref[:, 24:26]).max().item() < 1e-5
        if aux:
            assert cos(mean, r.float().mean(1)) >= COS_MIN
    again = pm.postmix_gates_gemm(d["codes"], d["scales"], weight[2][0], weight[2][1], d["residual"], d["post"],
                                  d["comb"], d["pre"], fn[1])
    assert torch.equal(again[2], gp) and torch.equal(again[1], x1)


def table_configs():
    seen = {}
    for rows, which in ((host._ROWS, "plain"), (host._GATE_ROWS, "gates")):
        for bound, cfg in rows:
            seen.setdefault((which, cfg), bound)
    return [(w, c, b) for (w, c), b in seen.items()]


@pytest.mark.parametrize("which,cfg,m", table_configs())
def test_every_table_config(which, cfg, m, weight, fn):
    d = make(min(m, 300), seed=5)
    base = pm.postmix_gemm(d["codes"], d["scales"], weight[2][0], weight[2][1], d["residual"], d["post"], d["comb"],
                           config=cfg)
    ref = postmix_gemm_reference(d["codes"], d["scales"], weight[0], weight[1], d["residual"], d["post"], d["comb"])
    assert cos(base, ref) >= COS_MIN
    if which == "gates" and cfg.kw <= cfg.ni:
        r, x1, gp, _ = pm.postmix_gates_gemm(d["codes"], d["scales"], weight[2][0], weight[2][1], d["residual"],
                                             d["post"], d["comb"], d["pre"], fn[1], config=cfg)
        assert cos(r, ref) >= COS_MIN
        ref_sums = partial_sums_reference(r, x1, fn[0]).double()
        assert ((gp.double().sum(1)[:, :24] - ref_sums[:, :24]).abs().max() / ref_sums[:, :24].abs().max()) < 1e-5
        again = pm.postmix_gates_gemm(d["codes"], d["scales"], weight[2][0], weight[2][1], d["residual"], d["post"],
                                      d["comb"], d["pre"], fn[1], config=cfg)
        assert torch.equal(again[2], gp)


@pytest.mark.parametrize("ks", (2, 4))
def test_split_k_counters_rearm(ks, weight, fn):
    d = make(40, seed=9)
    cfg = host.Config(2, 4, 2, ks)
    first = pm.postmix_gemm(d["codes"], d["scales"], weight[2][0], weight[2][1], d["residual"], d["post"], d["comb"],
                            config=cfg)
    for _ in range(3):
        assert torch.equal(first, pm.postmix_gemm(d["codes"], d["scales"], weight[2][0], weight[2][1], d["residual"],
                                                  d["post"], d["comb"], config=cfg))
    assert not host._WORKSPACES[first.device][1].any()


def test_rows_past_m_and_zero_streams(weight):
    d = make(20)
    d["residual"][3].zero_()
    out = run(d, weight)
    assert out.shape == (20, 4, H)
    d["post"].zero_()
    d["comb"].zero_()
    assert not run(d, weight).float().abs().max().item()


def test_rejects_bad_inputs(weight):
    d = make(16)
    with pytest.raises(AssertionError):
        pm.postmix_gemm(d["codes"].view(torch.uint8), d["scales"], weight[2][0], weight[2][1], d["residual"], d["post"],
                        d["comb"])
    with pytest.raises(AssertionError):
        run({**d, "post": d["post"].bfloat16()}, weight)
    with pytest.raises(AssertionError):
        run({**d, "residual": d["residual"][:, :3].contiguous()}, weight)
    with pytest.raises(AssertionError):
        run(d, weight, out=d["residual"])
    big = make(host.MAX_ROWS + 1)
    with pytest.raises(AssertionError):
        run(big, weight)


def test_graph_replay_new_inputs_poisoned(weight, fn):
    pm.warmup_postmix_gemm(*weight[2])
    for m in (16, 192, 384):
        d = make(m)
        keep = {k: v.clone() for k, v in d.items()}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run(keep, weight)
            pm.postmix_gates_gemm(keep["codes"], keep["scales"], weight[2][0], weight[2][1], keep["residual"],
                                  keep["post"], keep["comb"], keep["pre"], fn[1])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = run(keep, weight)
            gout = pm.postmix_gates_gemm(keep["codes"], keep["scales"], weight[2][0], weight[2][1], keep["residual"],
                                         keep["post"], keep["comb"], keep["pre"], fn[1])
        for seed in (1, 2):
            new = make(m, seed=seed)
            for k in keep:
                keep[k].copy_(new[k])
            for t in (out, *gout[:3]):
                t.view(torch.uint8).fill_(255)
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, run(new, weight))
            want = pm.postmix_gates_gemm(new["codes"], new["scales"], weight[2][0], weight[2][1], new["residual"],
                                         new["post"], new["comb"], new["pre"], fn[1])
            assert all(torch.equal(a, b) for a, b in zip(gout[:3], want[:3]))


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
    gen = torch.Generator(device="cuda").manual_seed(1)
    codes, exps = quantize_mxfp8((torch.randn(N, K, generator=gen, device="cuda") * 0.05).bfloat16())
    w = pm.preshuffle_weight(codes, exps)
    f = pm.pack_gate_weight(torch.randn(24, 4 * H, generator=gen, device="cuda") * 0.02)
    bad = []
    for m in (3, 1, 16, 17, 192, 384, 768, 40):
        d = make(m)
        for name, call in (
            ("plain", lambda: pm.postmix_gemm(d["codes"], d["scales"], w[0], w[1], d["residual"], d["post"], d["comb"])),
            ("gates", lambda: pm.postmix_gates_gemm(d["codes"], d["scales"], w[0], w[1], d["residual"], d["post"],
                                                    d["comb"], d["pre"], f)),
            ("aux", lambda: pm.postmix_gates_gemm(d["codes"], d["scales"], w[0], w[1], d["residual"], d["post"],
                                                  d["comb"], d["pre"], f, aux_mean=True)),
        ):
            before = len(late)
            call()
            if len(late) > before:
                bad.append((m, name))
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=3600,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 3, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
