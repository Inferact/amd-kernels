# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Wave row probes and skipped padding epilogues preserve every slot output."""

import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import mxfp4_moe_experts
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from .test_entry import weights
from .workload import make_activation, make_routing, replay_check, two_launch_slots


@pytest.mark.parametrize("m", [1536, 1537, 1600, 1792, 2047, 2048, 2049, 2304, 2560, 2816, 3072, 3328, 3583, 3584, 4095])
def test_row_group_boundaries(weights, m):
    _, w = weights
    x = make_activation(m, seed=m)
    wt, _ = make_routing(m, seed=m)
    counts = [1] + [b + d for b in range(16, 129, 16) for d in (-1, 0, 1)]
    counts += [m * 4 - sum(counts)]
    ids = torch.repeat_interleave(torch.arange(len(counts), dtype=torch.int32),
                                  torch.tensor(counts)).reshape(m, 4).cuda()
    expect = two_launch_slots(mxfp4_quant(x), w, wt, ids)

    def run():
        return mxfp4_moe_experts(mxfp4_quant(x), wt, ids, w)

    assert torch.isfinite(expect).all()
    for _ in range(3):
        assert torch.equal(run(), expect)
    eager, replayed = replay_check(lambda: [run()])
    assert torch.equal(eager[0], expect)
    assert torch.equal(replayed[0], expect)
