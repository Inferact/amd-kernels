# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch
from amd_kernels.norm.mhc_post_collapse_rmsnorm_h5120 import mhc_post_collapse_rmsnorm, warmup_mhc_post_collapse_rmsnorm
from amd_kernels.norm.mhc_post_collapse_rmsnorm_h5120.reference import mhc_post_collapse_rmsnorm_reference

H = 5120
ROWS = (1, 7, 16, 20, 64, 128, 256, 384, 512, 768, 1024)


def cos(a, b):
    a, b = a.float().flatten(1).double(), b.float().flatten(1).double()
    return (((a * b).sum(1) + 1e-30) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)).min()


def make(m, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed + m)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)  # noqa: E731
    scale = torch.exp2(torch.randint(-3, 4, (m, 1, 1), device="cuda", generator=g).float())
    return dict(
        residual=(r(m, 4, H) * scale).bfloat16(),
        pre_mix=torch.rand(m, 4, device="cuda", generator=g),
        norm_weight=(r(H) * 0.5 + 1).bfloat16(),
        sublayer_out=(r(m, H) * scale[:, 0]).bfloat16(),
        post_mix=torch.rand(m, 4, 1, device="cuda", generator=g),
        comb_mix=torch.softmax(r(m, 4, 4), -1),
    )


def same(a, b):
    return all((x is None and z is None) or torch.equal(x, z) for x, z in zip(a, b))


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("post", (True, False))
def test_contract_and_repeat(m, post):
    d = make(m)
    kw = {} if post else dict(sublayer_out=None, post_mix=None, comb_mix=None)
    args = dict(d, **kw)
    out = mhc_post_collapse_rmsnorm(d["residual"], d["pre_mix"], d["norm_weight"], eps=1e-6, keep_residual=post,
                                    aux_mean=True, **{k: args[k] for k in ("sublayer_out", "post_mix", "comb_mix")})
    torch.cuda.synchronize()
    y, r, mean = out
    ry, rr, rmean = mhc_post_collapse_rmsnorm_reference(d["residual"], d["pre_mix"], d["norm_weight"], 1e-6, **{
        k: args[k] for k in ("sublayer_out", "post_mix", "comb_mix")})
    assert torch.isfinite(y.float()).all() and cos(y, ry) >= 0.99999 and cos(mean, rmean) >= 0.99999
    if post:
        assert cos(r, rr) >= 0.99999
    else:
        assert r is None
    again = mhc_post_collapse_rmsnorm(d["residual"], d["pre_mix"], d["norm_weight"], eps=1e-6, keep_residual=post,
                                      aux_mean=True, **{k: args[k] for k in ("sublayer_out", "post_mix", "comb_mix")})
    assert same(out, again)


def test_without_optional_outputs():
    d = make(64)
    y, r, mean = mhc_post_collapse_rmsnorm(d["residual"], d["pre_mix"], d["norm_weight"], eps=1e-6)
    assert r is None and mean is None
    y2, r2, mean2 = mhc_post_collapse_rmsnorm(d["residual"], d["pre_mix"], d["norm_weight"], eps=1e-6, aux_mean=True,
                                              **{k: d[k] for k in ("sublayer_out", "post_mix", "comb_mix")})
    assert r2 is None and mean2 is not None


def test_graph_poison_change_and_repeat():
    m = 128
    d = make(m)
    kw = dict(eps=1e-6, keep_residual=True, aux_mean=True,
              **{k: d[k] for k in ("sublayer_out", "post_mix", "comb_mix")})
    mhc_post_collapse_rmsnorm(d["residual"], d["pre_mix"], d["norm_weight"], **kw)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mhc_post_collapse_rmsnorm(d["residual"], d["pre_mix"], d["norm_weight"], **kw)
    for i in range(3):
        d["residual"].copy_(make(m, seed=10 + i)["residual"])
        d["sublayer_out"].copy_(make(m, seed=20 + i)["sublayer_out"])
        eager = mhc_post_collapse_rmsnorm(d["residual"], d["pre_mix"], d["norm_weight"], **kw)
        for t in out:
            t.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert same(out, eager)


def test_warmup_covers_every_variant():
    from flydsl.compiler.jit_function import MlirCompiler

    warmup_mhc_post_collapse_rmsnorm()
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        d = make(48)
        for post in (False, True):
            for keep in (False, True) if post else (False,):
                for mean in (False, True):
                    kw = {k: d[k] for k in ("sublayer_out", "post_mix", "comb_mix")} if post else {}
                    mhc_post_collapse_rmsnorm(d["residual"], d["pre_mix"], d["norm_weight"], eps=1e-6,
                                              keep_residual=keep, aux_mean=mean, **kw)
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig
