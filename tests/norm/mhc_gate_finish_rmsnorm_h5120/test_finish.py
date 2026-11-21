# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The mHC seam finisher against ``reference.py`` for every record count the producer GEMMs emit and every row
count; the MXFP8 bytes exact against the quantizer applied to its own bf16 output; the full chain (producer GEMM
records -> finisher) against the seam kernel of ``norm/mhc_post_pre_rmsnorm_h5120``; bitwise repeats; graph replay
with poisoned outputs; the first call's warm-up."""

import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.norm import mhc_gate_finish_rmsnorm_h5120 as fin
from amd_kernels.norm.mhc_gate_finish_rmsnorm_h5120 import host
from amd_kernels.norm.mhc_gate_finish_rmsnorm_h5120.reference import (
    COS_MIN, dequantize_mxfp8, gate_finish_reference, quantize_mxfp8_reference,
)

H = 5120
KW = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult=1.5, norm_eps=1e-6)
ROWS = (1, 16, 17, 100, 192, 384, 768, 1024)
NBS = (32, 40, 64, 80, 128, 160, 320, 640)


def cos(a, b):
    a, b = a.float().flatten(1).double(), b.float().flatten(1).double()
    return (((a * b).sum(1) + 1e-30) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)).min().item()


def make(m, nb, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed + 31 * m + nb)
    scale = torch.exp2(torch.randint(-3, 4, (m, 1), device="cuda", generator=g).float())
    x1 = (torch.randn(m, H, generator=g, device="cuda") * scale).bfloat16()
    gp = torch.zeros(m, nb, 32, device="cuda")
    gp[:, :, :24] = torch.randn(m, nb, 24, generator=g, device="cuda") * 0.5
    gp[:, :, 24] = torch.rand(m, nb, generator=g, device="cuda") * (4 * H / nb) * 4
    gp[:, :, 25] = x1.float().square().sum(1, keepdim=True) / nb
    return dict(partials=gp, x1=x1, scale=torch.tensor([0.5, 0.7, 0.9], device="cuda"),
                base=torch.randn(24, generator=g, device="cuda") * 0.3,
                weight=(torch.randn(H, generator=g, device="cuda") * 0.5 + 1).bfloat16())


def run(d, **kw):
    return fin.mhc_gate_finish(d["partials"], d["x1"], d["scale"], d["base"], d["weight"], **KW, **kw)


def ref(d):
    return gate_finish_reference(d["partials"], d["x1"], d["scale"], d["base"], d["weight"], 1e-6, 1e-6, 1e-6, 1.5, 1e-6)


def require():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


@pytest.mark.parametrize("nb", NBS)
@pytest.mark.parametrize("m", ROWS)
def test_matches_reference_and_repeats(m, nb):
    require()
    d = make(m, nb)
    out = run(d, quant=True)
    torch.cuda.synchronize()
    post, comb, y, npre, codes, qs = out
    rpost, rcomb, ry, rpre = ref(d)
    assert torch.isfinite(y.float()).all()
    assert cos(post, rpost) >= COS_MIN and cos(npre, rpre) >= COS_MIN and cos(comb, rcomb) >= COS_MIN
    assert (comb - rcomb).abs().max() < 5e-4 and cos(y, ry) >= COS_MIN
    qc, qq = quantize_mxfp8_reference(y)
    assert torch.equal(codes.view(torch.uint8), qc.view(torch.uint8)) and torch.equal(qs, qq)
    assert cos(dequantize_mxfp8(codes, qs), y) >= 0.999
    again = run(d, quant=True)
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, again))
    assert all(torch.equal(a, b) for a, b in zip(run(d), out[:4]))


def test_zero_row_and_partial_block():
    require()
    d = make(20, 64)
    d["x1"][3].zero_()
    d["partials"][3, :, 25] = 0
    post, comb, y, npre, codes, qs = run(d, quant=True)
    assert (y[3] == 0).all() and (qs[3] == 0).all()
    assert torch.isfinite(comb).all() and torch.isfinite(post).all()


def test_rejects_bad_inputs():
    require()
    d = make(16, 64)
    with pytest.raises(AssertionError):
        fin.mhc_gate_finish(d["partials"][:, :, :31].contiguous(), d["x1"], d["scale"], d["base"], d["weight"], **KW)
    with pytest.raises(AssertionError):
        fin.mhc_gate_finish(d["partials"], d["x1"].float(), d["scale"], d["base"], d["weight"], **KW)


def test_graph_replay_new_inputs_poisoned():
    require()
    m, nb = 192, 80
    d = make(m, nb)
    run(d, quant=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run(d, quant=True)
    for seed in (1, 2):
        new = make(m, nb, seed=seed)
        for k in d:
            d[k].copy_(new[k])
        for t in out:
            t.view(torch.uint8).fill_(255)
        g.replay()
        torch.cuda.synchronize()
        want = run(new, quant=True)
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, want))


def test_chain_matches_seam_kernel():
    """The producer GEMM's records through the finisher against the seam kernel on the same new residual."""
    require()
    from amd_kernels.gemm import mxfp8_postmix_hc4_n5120_k8192 as pm
    from amd_kernels.gemm.mxfp8_postmix_hc4_n5120_k8192.reference import quantize_mxfp8
    from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120 import mhc_post_pre_rmsnorm, pack_gate_weight

    g = torch.Generator(device="cuda").manual_seed(11)
    codes, exps = quantize_mxfp8((torch.randn(5120, 8192, generator=g, device="cuda") * 0.05).bfloat16())
    w = pm.preshuffle_weight(codes, exps)
    fn = torch.randn(24, 4 * H, generator=g, device="cuda") * 0.02
    d = make(1, 1)
    for m in (16, 192, 384):
        ac, asc = quantize_mxfp8(torch.randn(m, 8192, generator=g, device="cuda").bfloat16())
        res = torch.randn(m, 4, H, generator=g, device="cuda").bfloat16()
        post = torch.rand(m, 4, 1, generator=g, device="cuda")
        comb = torch.softmax(torch.randn(m, 4, 4, generator=g, device="cuda"), -1)
        pre = torch.rand(m, 4, generator=g, device="cuda")
        r, x1, gp, _ = pm.postmix_gates_gemm(ac, asc, w[0], w[1], res, post, comb, pre, pm.pack_gate_weight(fn))
        got = fin.mhc_gate_finish(gp, x1, d["scale"], d["base"], d["weight"], quant=True, **KW)
        want = mhc_post_pre_rmsnorm(r, pack_gate_weight(fn), d["scale"], d["base"], d["weight"], pre_mix=pre, quant=True,
                                    **KW)
        assert cos(got[0], want[1]) >= COS_MIN and cos(got[1], want[2]) >= COS_MIN and cos(got[3], want[4]) >= COS_MIN
        assert cos(got[2], want[3]) >= COS_MIN
        assert (got[4].view(torch.uint8) != want[5].view(torch.uint8)).float().mean() < 1e-3


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
    for nb, quant in ((64, True), (160, False), (64, False)):
        d = make(32, nb)
        run(d, quant=quant)
        for m in (1, 16, 700):
            before = len(late)
            run(make(m, nb), quant=quant)
            if len(late) > before:
                bad.append((nb, quant, m))
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_size(tmp_path):
    require()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 3, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
