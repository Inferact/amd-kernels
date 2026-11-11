# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One-GPU contract of the N-shard concat linear: fp64 reference, every row tier, odd and zero rows, strided taps,
exponents +-5 binades, determinism, graph replay with poisoned outputs."""
import pytest
import torch
from amd_kernels.gemm.bf16_concat_linear_n1536_k30720 import bf16_concat_linear, pack_weight, warmup_concat_linear
from amd_kernels.gemm.bf16_concat_linear_n1536_k30720.host import config_for
from amd_kernels.gemm.bf16_concat_linear_n1536_k30720.reference import concat_linear_reference

ROWS = (1, 7, 8, 16, 17, 33, 63, 64, 65, 80, 81, 127, 128, 129, 255, 256, 300, 512, 1000, 1024, 2048)


@pytest.fixture(scope='module')
def weights():
    g = torch.Generator(device='cuda').manual_seed(5)
    w = (torch.randn((1536, 30720), device='cuda', generator=g) * 0.02).bfloat16()
    return w, pack_weight(w)


def taps_for(m, seed, stride=6152):
    g = torch.Generator(device='cuda').manual_seed(seed)
    out = []
    for _ in range(5):
        base = torch.randn((m, stride), device='cuda', generator=g)
        ex = torch.randint(-5, 6, (m, 192, 1), device='cuda', generator=g).expand(m, 192, 32).reshape(m, 6144)
        x = base[:, :6144] * torch.exp2(ex.float())
        base[:, :6144] = x
        t = base.bfloat16()[:, :6144]
        t[0].zero_()
        out.append(t)
    return out


@pytest.mark.parametrize('m', ROWS)
def test_contract(weights, m):
    w, packed = weights
    taps = taps_for(m, m)
    ref = concat_linear_reference(taps, w)
    got = bf16_concat_linear(taps, packed)
    torch.cuda.synchronize()
    assert not got.isnan().any() and not got.isinf().any()
    assert torch.equal(got[0], torch.zeros_like(got[0]))  # zero row
    live = slice(1, m)
    if m > 1:
        cos = torch.nn.functional.cosine_similarity(got[live].double(), ref[live].double(), dim=1)
        assert float(cos.min()) >= 0.99999, (m, config_for(m), float(cos.min()))
        rel = float((got[live].double() - ref[live].double()).norm() / ref[live].double().norm())
        assert rel <= 1e-3, (m, rel)
        per_row_rel = (got[live].double() - ref[live].double()).norm(dim=1) / ref[live].double().norm(dim=1)
        assert float(per_row_rel.max()) <= 1e-3, (m, float(per_row_rel.max()))
    again = bf16_concat_linear(taps, packed)
    torch.cuda.synchronize()
    assert torch.equal(got, again)


@pytest.mark.parametrize('m', (8, 80, 128, 256, 1024))
def test_graph_replay(weights, m):
    w, packed = weights
    taps = taps_for(m, 100 + m)
    warmup_concat_linear(taps, packed, max_rows=m)
    eager = bf16_concat_linear(taps, packed)
    out = torch.empty_like(eager)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        bf16_concat_linear(taps, packed, out=out)
    for _ in range(3):
        out.fill_(float('nan'))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
    # changed inputs (in place) must be consumed by the replay: equal to a fresh eager call, not the old result
    for t in taps:
        t.mul_(0.5).add_(0.25)
    fresh = bf16_concat_linear(taps, packed)
    g.replay()
    torch.cuda.synchronize()
    assert not torch.equal(fresh, eager)
    assert torch.equal(out, fresh)


@pytest.mark.parametrize('m',(80,256))
def test_distinct_tap_strides(weights,m):
    w,packed=weights
    gen=torch.Generator(device='cuda').manual_seed(901+m)
    taps=[torch.randn((m,6144+8*j),device='cuda',generator=gen,dtype=torch.bfloat16)[:,:6144] for j in range(5)]
    got=bf16_concat_linear(taps,packed)
    ref=concat_linear_reference(taps,w)
    cos=torch.nn.functional.cosine_similarity(got.double(),ref.double(),dim=1)
    assert cos.min()>=.99999
    assert (got.double()-ref.double()).norm()/ref.double().norm()<=1e-3


@pytest.mark.parametrize('m', (8, 17, 80, 256, 512, 1024, 2048))
def test_mfma_co_residency(weights, m):
    from tests.mfma_spinner import mfma_spinner
    _, packed = weights
    taps = taps_for(m, 2000 + m)
    ref = bf16_concat_linear(taps, packed)
    torch.cuda.synchronize()
    spinner = mfma_spinner(256)
    side = torch.cuda.Stream(priority=-1)
    for _ in range(3):
        spinner(100_000, side)
        got = bf16_concat_linear(taps, packed)
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
