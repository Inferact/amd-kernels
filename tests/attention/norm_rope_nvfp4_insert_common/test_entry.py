# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry writes the same bits (q, index-q, both NVFP4 caches) as a direct call of the shape package (one GPU)."""

import pytest
import torch

from amd_kernels.attention import norm_rope_nvfp4_insert_common as E
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import qkv_norm_rope_nvfp4_insert

KW = {"num_heads": 16, "num_kv_heads": 1, "num_index_heads": 1, "eps": 1e-6}


def build(n):
    g = torch.Generator(device="cuda").manual_seed(2)
    dev = "cuda"

    def r(*s):
        return torch.randn(*s, generator=g, device=dev).to(torch.bfloat16)

    slots = torch.arange(n, device=dev, dtype=torch.int64) + 128  # a request's consecutive slots
    return [r(n, 20 * 128), r(128), r(128), r(128), r(128), r(4096, 64),
            torch.arange(n, device=dev, dtype=torch.int64), slots, slots.clone(),
            torch.zeros(4, 2, 128, 72, dtype=torch.uint8, device=dev),
            torch.zeros(4, 128, 72, dtype=torch.uint8, device=dev),
            torch.tensor(1.0, device=dev), torch.tensor(1.0, device=dev)]


@pytest.mark.parametrize("n", [1, 96])
@torch.inference_mode()
def test_entry_equals_package(n):
    if not E.available():
        pytest.skip("needs gfx950 with FlyDSL")
    a, b = build(n), build(n)
    assert E.qkv_norm_rope_nvfp4_insert_blocker(*a, **KW) is None
    qa, ia = E.qkv_norm_rope_nvfp4_insert(*a, **KW)
    qb, ib = qkv_norm_rope_nvfp4_insert(*b, **KW)
    torch.cuda.synchronize()
    assert torch.equal(qa, qb) and torch.equal(ia.view(torch.uint8), ib.view(torch.uint8))
    assert torch.equal(a[9], b[9]) and torch.equal(a[10], b[10])
