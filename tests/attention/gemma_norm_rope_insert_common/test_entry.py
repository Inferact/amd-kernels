# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry writes the same bits (q and the e4m3 cache) as a direct call of the shape package, and its reason is None
on that call (one GPU)."""

import pytest
import torch

from amd_kernels.attention import gemma_norm_rope_insert_common as E
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.qkv_norm_rope_insert import qkv_norm_rope_insert


def build(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, nq, nkv, d = "cuda", 16, 1, 128

    def r(*s):
        return torch.randn(*s, generator=g, device=dev).to(torch.bfloat16)

    a = [r(n, (nq + 2 * nkv) * d), torch.arange(n, device=dev, dtype=torch.int64) * 7, r(d), r(d), r(4096, 64), 1e-6,
         nq, nkv, torch.empty(n, nq * d, dtype=torch.bfloat16, device=dev),
         torch.zeros(4, nkv, 16, 2 * d, dtype=torch.uint8, device=dev),
         torch.randperm(64, generator=g, device=dev)[:n].to(torch.int64),
         torch.tensor(0.5, device=dev), torch.tensor(0.25, device=dev)]
    return a


@pytest.mark.parametrize("n", [1, 40])
@torch.inference_mode()
def test_entry_equals_package(n):
    if not E.available():
        pytest.skip("needs gfx950 with FlyDSL")
    a, b = build(n), build(n)
    assert E.gemma_qkv_norm_rope_insert_blocker(*a) is None
    E.gemma_qkv_norm_rope_insert(*a)
    qkv_norm_rope_insert(*b)
    torch.cuda.synchronize()
    assert torch.equal(a[8], b[8]) and torch.equal(a[9], b[9])
