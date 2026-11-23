# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry's shuffle and GEMM equal the shape package's bit for bit (one GPU, N 50016 K 6144)."""

import pytest
import torch

from amd_kernels.gemm import bf16_shuffled_gemm_common as E
from amd_kernels.gemm.bf16_n50016_k6144 import bf16_gemm_n50016_k6144, shuffle_weight


@pytest.mark.parametrize("m", [1, 300])
@torch.inference_mode()
def test_entry_equals_package(m):
    if not E.available():
        pytest.skip("needs gfx950 with FlyDSL")
    g = torch.Generator(device="cuda").manual_seed(3)
    w = torch.randn(50016, 6144, generator=g, device="cuda").to(torch.bfloat16)
    x = torch.randn(m, 6144, generator=g, device="cuda").to(torch.bfloat16)
    ref_w = shuffle_weight(w)
    assert E.prepare_bf16_shuffled_gemm_weight(w) is w and torch.equal(w, ref_w)
    assert E.bf16_shuffled_gemm_blocker(x, w) is None
    out = E.bf16_shuffled_gemm(x, w)
    ref = bf16_gemm_n50016_k6144(x, ref_w, shuffled=True)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
