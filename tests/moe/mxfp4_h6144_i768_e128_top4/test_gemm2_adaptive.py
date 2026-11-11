# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The adaptive GEMM2's column splits follow the actual padded expert blocks, with exact slot ownership."""

import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import mxfp4_moe_experts
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from .test_entry import weights
from .workload import make_activation, make_routing, replay_check, two_launch_slots


def routing_with_blocks(m, blocks):
    """Expert counts with exactly ``blocks`` padded 128-row tiles; duplicate ids in a token stress slot ownership."""
    active = min(blocks, 128)
    counts = [1] * active
    counts[0] += (blocks - active) * 128
    capacity = [128] * active
    capacity[0] += (blocks - active) * 128
    left = m * 4 - sum(counts)
    assert left >= 0
    for e in range(active):
        take = min(left, capacity[e] - counts[e])
        counts[e] += take
        left -= take
    assert left == 0
    assert sum((c + 127) // 128 for c in counts) == blocks
    ids = torch.repeat_interleave(torch.arange(active, dtype=torch.int32), torch.tensor(counts))
    return ids.reshape(m, 4).cuda()


@pytest.mark.parametrize("m,blocks", [
    (513, 17), (513, 21), (513, 22), (513, 42), (513, 43), (513, 64),
    (513, 65), (513, 85), (513, 86), (513, 127), (513, 128), (513, 129),
    (3584, 170), (3584, 171), (3584, 192), (3584, 193),
])
def test_column_split_thresholds(weights, m, blocks):
    _, w = weights
    x = make_activation(m, seed=m)
    wt, _ = make_routing(m, seed=m)
    ids = routing_with_blocks(m, blocks)
    expect = two_launch_slots(mxfp4_quant(x), w, wt, ids)
    run = lambda: mxfp4_moe_experts(mxfp4_quant(x), wt, ids, w)
    got = run()
    assert torch.isfinite(got).all()
    assert torch.equal(got, expect)
    assert torch.equal(got, run())
    eager, replayed = replay_check(lambda: [run()])
    assert torch.equal(eager[0], expect)
    assert torch.equal(replayed[0], expect)


@pytest.mark.parametrize("m", [513, 640, 641, 1535, 1536, 1537, 3583, 3584, 3585, 4095, 4096])
def test_adaptive_dispatch_edges(weights, m):
    _, w = weights
    x = make_activation(m, seed=m)
    wt, ids = make_routing(m, seed=m, skewed=True)
    expect = two_launch_slots(mxfp4_quant(x), w, wt, ids)
    got = mxfp4_moe_experts(mxfp4_quant(x), wt, ids, w)
    assert torch.equal(got, expect)
