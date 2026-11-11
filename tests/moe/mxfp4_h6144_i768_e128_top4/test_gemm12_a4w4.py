# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""W4A4 GEMM1 + GEMM2 in one launch (``gemm12_a4w4``, 128- and 256-row tiles): the weighted slots equal the two-launch
chain bit for bit on every routing, the m-block counters are zero after every call, a graph replays changed inputs."""

import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import mxfp4_moe_experts
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4 as a2
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4_bm128 as aw
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant

from .workload import H, make_activation, make_experts, make_routing, require_gfx950, two_launch_slots

LO, HI = aw.GEMM12_TOKENS
LO2, HI2 = a2.GEMM12_TOKENS
MS = sorted({LO, LO + 1, LO + 63, aw.GEMM12_ADAPTIVE_MAX, aw.GEMM12_ADAPTIVE_MAX + 1, 1024, 1407, 1408,
             1536, 1537, 2047, 2048, 2049, 2311, 3583, 3584, 3585, 3841, 3968, HI - 1, HI})
MS2 = sorted({LO2, LO2 + 1, 5000, 7424, 7425, 7680, 8191, HI2})


@pytest.fixture(scope="module")
def weights():
    require_gfx950()
    raw, _ = make_experts()
    gen = torch.Generator(device="cuda").manual_seed(77)
    gate = torch.randn((128, H), generator=gen, device="cuda") / H**0.5
    bias = torch.randn((128,), generator=gen, device="cuda") * 0.01
    return MoeWeights.from_row_major(gate, bias, *raw)


def _routing(m, routing, seed):
    wt, ids = make_routing(m, seed=seed, skewed=routing == "hot")
    if routing == "four":
        ids = torch.arange(4, device="cuda", dtype=torch.int32).expand(m, 4).contiguous()
    if routing == "one-hot":
        ids = ids % 127 + 1
        ids[:, 0] = 0
    return wt, ids


def _two_launch(x, wt, ids, w):
    return two_launch_slots(mxfp4_quant(x), w, wt, ids)


def _counters_zero():
    torch.cuda.synchronize()
    return all(int(t.count_nonzero()) == 0 for t in aw._ready.values())


def test_range_is_served(weights, monkeypatch):
    """The entry takes the one-launch path inside the range and only there."""
    calls = []
    for mod in (aw, a2):
        real = mod._gemm12
        monkeypatch.setattr(mod, "_gemm12", lambda *a, _r=real, **k: calls.append(a) or _r(*a, **k))
    for m in (LO, HI, HI + 1, LO2, HI2, HI2 + 1):
        x = make_activation(m, seed=m)
        wt, ids = _routing(m, "uniform", m)
        mxfp4_moe_experts(mxfp4_quant(x), wt, ids, weights)
    assert len(calls) == 4 + (HI + 1 == LO2)


@pytest.mark.parametrize("m", MS + MS2)
@pytest.mark.parametrize("routing", ["uniform", "hot", "four", "one-hot"])
def test_gemm12_a4w4_equals_two_launches(weights, m, routing):
    w = weights
    for seed in (m, m + 1000):  # a changed input reuses the reset counters
        x = make_activation(m, seed=seed)
        wt, ids = _routing(m, routing, seed)
        expect = _two_launch(x, wt, ids, w)
        assert torch.isfinite(expect.float()).all()
        for _ in range(3):
            got = mxfp4_moe_experts(mxfp4_quant(x), wt, ids, w)
            assert torch.equal(got, expect)
            assert _counters_zero()


@pytest.mark.parametrize("m", [LO, aw.GEMM12_ADAPTIVE_MAX + 1, 1024, 1536, 2048, 2049, 3584, HI, LO2, HI2])
def test_gemm12_a4w4_graph_replay_changed_inputs(weights, m):
    """Capture once, replay on inputs written into the captured buffers, the output NaN-poisoned before each replay."""
    w = weights
    x = make_activation(m, seed=1)
    wt, ids = _routing(m, "uniform", 1)
    mxfp4_moe_experts(mxfp4_quant(x), wt, ids, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mxfp4_moe_experts(mxfp4_quant(x), wt, ids, w)
    for seed, routing in [(2, "uniform"), (3, "hot"), (4, "one-hot"), (5, "four"), (6, "uniform")]:
        x.copy_(make_activation(m, seed=seed))
        nwt, nids = _routing(m, routing, seed)
        wt.copy_(nwt)
        ids.copy_(nids)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expect = _two_launch(x, wt, ids, w)
        assert torch.equal(out, expect)
        assert _counters_zero()


@pytest.mark.parametrize("m", [513, 1024, 1536, 2048, 3584, 4095, 4096, 4097, 8192, 16384, 32768])
def test_gemm12_next_to_mfma(weights, m):
    """The state handoff and the epilogues give the two-launch bits next to another MFMA stream."""
    from tests.mfma_spinner import mfma_spinner
    x = make_activation(m, seed=m)
    wt, ids = _routing(m, "uniform", m)
    expect = _two_launch(x, wt, ids, weights)
    mxfp4_moe_experts(mxfp4_quant(x), wt, ids, weights)
    spin = mfma_spinner(64)
    foreign = torch.cuda.Stream(priority=-1)
    foreign.wait_stream(torch.cuda.current_stream())
    spin(2_000_000, foreign)
    for _ in range(3):
        got = mxfp4_moe_experts(mxfp4_quant(x), wt, ids, weights)
        assert torch.equal(got, expect)
        assert _counters_zero()
    foreign.synchronize()
