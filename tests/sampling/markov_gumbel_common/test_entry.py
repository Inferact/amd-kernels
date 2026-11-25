# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``markov_gumbel_fused`` through the entry is the package's call, bit for bit (draft tokens, the draft logits
cache), at the smallest and largest served row counts, with no reason on those calls; the scratch each call left
(kernel-owned, its records are not a contract) serves a second call of new inputs the same way; an operand off the
current device gets a reason."""

import importlib

import pytest
import torch

from amd_kernels.sampling.markov_gumbel_common import (
    available,
    markov_gumbel_fused,
    markov_gumbel_fused_blocker,
    markov_max_rows,
    markov_scratch_dwords,
)
from amd_kernels.sampling.markov_gumbel_common.host import _ROUTES

from .inputs import RANK, V, clone, make


@pytest.mark.parametrize("n,k", [(1, 4), (48, 3), (48, 8)])
def test_entry_is_the_package(n, k):
    if not available():
        pytest.skip("requires gfx950 with FlyDSL")
    pkg = importlib.import_module(_ROUTES.hosts[(V, RANK)])
    d = make(n, k, 10 + n + k)
    assert markov_max_rows(d["w1"]) == 48 and markov_scratch_dwords(d["w1"], n) <= d["scratch"].numel()
    assert markov_gumbel_fused_blocker(**d) is None
    a, b = clone(d), clone(d)
    markov_gumbel_fused(**a)
    pkg.markov_gumbel_fused(**b)
    torch.cuda.synchronize()
    for key in ("draft_tokens", "cache"):
        assert torch.equal(a[key], b[key]), key
    assert not torch.equal(a["cache"], d["cache"]) and bool((a["draft_tokens"][:n] != 0).any())
    d2 = make(n, k, 20 + n + k, (d["w1"], d["w2"]))
    a2, b2 = clone(d2), clone(d2)
    a2["scratch"], b2["scratch"] = a["scratch"], b["scratch"]
    markov_gumbel_fused(**a2)
    pkg.markov_gumbel_fused(**b2)
    torch.cuda.synchronize()
    for key in ("draft_tokens", "cache"):
        assert torch.equal(a2[key], b2[key]), key


@pytest.mark.parametrize("name", ["base", "seeds", "row_mapping"])
def test_off_device_reason(name):
    if not available():
        pytest.skip("requires gfx950 with FlyDSL")
    d = make(4, 3, 5)
    d[name] = d[name].cpu()
    why = markov_gumbel_fused_blocker(**d)
    assert why == f"markov_gumbel_fused: {name} on cpu (the current CUDA device)"
