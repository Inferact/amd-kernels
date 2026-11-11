# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The small-tile host (shared tiles folded in): its bits next to an MFMA co-runner, and inline routing equal to the
sorted rows."""
import pytest
import torch
from .test_entry import weights  # noqa: F401
from .workload import make_routing, make_activation, replay_check, small_slots


@pytest.mark.parametrize("m", [1, 8, 16, 257, 385, 512])
def test_next_to_mfma(weights, m):
    from tests.mfma_spinner import mfma_spinner
    _, w = weights
    x = make_activation(m, seed=200 + m)
    wt, ids = make_routing(m, seed=200 + m)
    fn = lambda: small_slots(x, w, wt, ids)  # noqa: E731
    expected = [t.clone() for t in fn()]
    torch.cuda.synchronize()
    spinner = mfma_spinner(256)
    stream = torch.cuda.Stream(priority=-1)
    spinner(5_000_000, stream)  # 50 ms; the contraction must retain its own bits
    out = fn()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(out, expected))


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 11, 12, 13, 15, 16])
@pytest.mark.parametrize("kind", ["uniform", "hot", "four", "one"])
def test_inline_sort(weights, m, kind):
    _, w = weights
    x = make_activation(m, seed=m)
    wt, ids = make_routing(m, seed=m, skewed=kind == 'hot')
    if kind == 'four':
        ids = torch.arange(4, device='cuda', dtype=torch.int32).expand(m, 4).contiguous()
    if kind == 'one':
        ids.zero_()
    expect = small_slots(x, w, wt, ids, id_rows=False, inline=False)
    eager, replay = replay_check(lambda: list(small_slots(x, w, wt, ids, id_rows=False, inline=True)))
    assert all(torch.equal(a, b) for a, b in zip(eager, replay))
    assert all(torch.equal(a, b) for a, b in zip(eager, expect))
