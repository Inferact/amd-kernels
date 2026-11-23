# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch
from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120 import mhc_post_pre_rmsnorm, pack_gate_weight, warmup_mhc_post_pre_rmsnorm
from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120.host import _config
from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120.reference import (
    dequantize_mxfp8,
    mhc_post_pre_rmsnorm_reference,
    quantize_mxfp8_reference,
)

H = 5120
KW = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult=1.5, norm_eps=1e-6)
ROWS = (1, 16, 20, 32, 64, 100, 128, 192, 256, 320, 384, 448, 512, 768, 1024, 2048, 4096)


def cos(a, b):
    """Minimum per-row cosine; a row that is zero in both is 1."""
    a, b = a.float().flatten(1).double(), b.float().flatten(1).double()
    return (((a * b).sum(1) + 1e-30) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)).min()


def make(m, seed=0, post=True, pre=True):
    g = torch.Generator(device="cuda").manual_seed(seed + m)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)  # noqa: E731
    scale = torch.exp2(torch.randint(-3, 4, (m, 1, 1), device="cuda", generator=g).float())
    d = dict(
        residual=(r(m, 4, H) * scale).bfloat16(),
        fn=r(24, 4 * H) * 0.02,
        hc_scale=torch.tensor([0.5, 0.7, 0.9], device="cuda"),
        hc_base=r(24) * 0.3,
        norm_weight=(r(H) * 0.5 + 1).bfloat16(),
    )
    if post:
        d.update(sublayer_out=(r(m, H) * scale[:, 0]).bfloat16(), post_mix=torch.rand(m, 4, 1, device="cuda", generator=g),
                 comb_mix=torch.softmax(r(m, 4, 4), -1))
    if pre:
        d["pre_mix"] = torch.rand(m, 4, device="cuda", generator=g)
    return d


def run(d, **kw):
    return mhc_post_pre_rmsnorm(d["residual"], pack_gate_weight(d["fn"]), d["hc_scale"], d["hc_base"], d["norm_weight"],
                                **{k: d[k] for k in ("sublayer_out", "post_mix", "comb_mix", "pre_mix") if k in d},
                                **KW, **kw)


def ref(d):
    return mhc_post_pre_rmsnorm_reference(
        d["residual"], d["fn"], d["hc_scale"], d["hc_base"], 1e-6, 1e-6, 1e-6, 1.5, 20, d.get("pre_mix"),
        d.get("sublayer_out"), d.get("post_mix"), d.get("comb_mix"), d["norm_weight"], 1e-6,
    )


def check(d, out, quant):
    rr, rpost, rcomb, rout, rpre = ref(d)
    ro, post, comb, y, npre = out[:5]
    assert torch.isfinite(y.float()).all()
    assert cos(ro, rr) >= 0.99999 and cos(post, rpost) >= 0.99999 and cos(npre, rpre) >= 0.99999
    assert cos(comb, rcomb) >= 0.99999 and (comb - rcomb).abs().max() < 5e-4
    assert cos(y, rout) >= 0.99999
    if quant:
        codes, scale = out[5], out[6]
        qc, qs = quantize_mxfp8_reference(y)
        assert torch.equal(codes.view(torch.uint8), qc.view(torch.uint8)) and torch.equal(scale, qs)
        assert cos(dequantize_mxfp8(codes, scale), y) >= 0.999


def same(a, b):
    return all(torch.equal(x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x,
                           z.view(torch.uint8) if z.dtype == torch.float8_e4m3fn else z) for x, z in zip(a, b))


@pytest.mark.parametrize("m", ROWS)
def test_contract_and_repeat(m):
    d = make(m)
    out = run(d, quant=True)
    torch.cuda.synchronize()
    check(d, out, True)
    for _ in range(3):
        assert same(run(d, quant=True), out)


@pytest.mark.parametrize("m", (16, 256))
@pytest.mark.parametrize("post,pre", ((False, False), (False, True), (True, False)))
def test_variants(m, post, pre):
    d = make(m, post=post, pre=pre)
    out = run(d)
    check(d, out, False)
    if post:
        assert not torch.equal(out[0], d["residual"])
    else:
        assert out[0] is d["residual"]


@pytest.mark.parametrize("m", (16, 256))
def test_aux_mean(m):
    d = make(m)
    out = run(d, aux_mean=True)
    rr = ref(d)[0]
    assert torch.equal(out[-1], rr.float().mean(1).bfloat16()) or cos(out[-1], rr.float().mean(1)) >= 0.99999
    assert same(out[:5], run(d))


def test_zero_rows_and_partial_block():
    d = make(20)
    d["residual"][3].zero_()
    d["sublayer_out"][3].zero_()
    out = run(d, quant=True)
    assert (out[3][3] == 0).all() and (out[6][3] == 0).all()
    check(d, out, True)


def test_graph_poison_change_and_repeat():
    m = 128
    d = make(m)
    run(d, quant=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run(d, quant=True)
    for i in range(3):
        d["residual"].copy_(make(m, seed=10 + i)["residual"])
        d["sublayer_out"].copy_(make(m, seed=20 + i)["sublayer_out"])
        eager = run(d, quant=True)
        for t in out:
            t.view(torch.uint8).fill_(255) if t.dtype in (torch.uint8, torch.float8_e4m3fn) else t.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert same(out, eager)
        check(d, out, True)


def test_dispatch_grid_within_cus():
    for m in range(16, 4097, 16):
        rt, ks, waves = _config(m)
        assert (m // 16 + (m % 16 > 0)) * ks <= 256 or ks == 1
        assert 160 % ks == 0 and (160 // ks) % waves == 0


def test_warmup_covers_every_size():
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels import warm_up

    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    orig = MlirCompiler.compile
    try:
        warmup_mhc_post_pre_rmsnorm()

        def unexpected(*a, **kw):
            raise AssertionError("compile after warm-up")

        MlirCompiler.compile = unexpected
        for m in (16, 64, 128, 256, 384, 512, 768):
            d = make(m)
            run(d, quant=True)
            run(d)
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig
        warm_up.set_auto(False)
