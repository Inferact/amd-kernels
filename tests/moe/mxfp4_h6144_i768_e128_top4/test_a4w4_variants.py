# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The BM128 variants equal BM256 slot by slot on odd and skewed routing."""
import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.experts_a4w4 import a4w4_experts
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.experts_a4w4_bm128 import a4w4_experts as candidate
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from .test_entry import weights
from .workload import make_routing, make_activation, two_launch_slots


@pytest.mark.parametrize('m',[513,1024,1537,1792,2048,2560,3072,3583,4096])
@pytest.mark.parametrize('skewed',[False,True])
def test_bm128(weights,m,skewed):
    _,w=weights
    wt,ids=make_routing(m,seed=m,skewed=skewed)
    xq=mxfp4_quant(make_activation(m,seed=m))
    expect=a4w4_experts(xq,w.w13,w.w2,w.a4w4_scales,wt,ids)
    out=candidate(xq,w.w13,w.w2,w.a4w4_scales,wt,ids)
    assert torch.equal(out,expect)
    assert torch.equal(out,candidate(xq,w.w13,w.w2,w.a4w4_scales,wt,ids))


@pytest.mark.parametrize("m", [4097, 32768])
def test_bm256_four_experts_scale_ring(weights, m):
    """Stress repeated scale-slot reuse with long expert runs."""
    from .workload import replay_check

    _, w = weights
    wt, _ = make_routing(m, seed=m)
    ids = torch.tensor([7, 63, 81, 127], device="cuda", dtype=torch.int32).expand(m, 4).contiguous()
    xq = mxfp4_quant(make_activation(m, seed=m))
    expect = two_launch_slots(xq, w, wt, ids)

    def run():
        return a4w4_experts(xq, w.w13, w.w2, w.a4w4_scales, wt, ids)

    assert torch.equal(run(), expect)
    assert torch.equal(run(), expect)
    eager, replayed = replay_check(lambda: [run()])
    assert torch.equal(eager[0], expect)
    assert torch.equal(replayed[0], expect)
