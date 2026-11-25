# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The one-GPU entry writes the same codes, scales and residual as a direct call of the shape package (one GPU; row
counts that fill whole code and scale tiles: the padding rows of the buffers are not written)."""

import pytest
import torch

from amd_kernels.norm import gemma_add_rmsnorm_common as E
from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm_mxfp8_for


def build(m):
    g = torch.Generator(device="cuda").manual_seed(4)

    def r(*s):
        return torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)

    gemm_weight = torch.zeros(2560, 6144, dtype=torch.float8_e4m3fn, device="cuda")
    return r(m, 6144), r(m, 6144), r(6144), 1e-6, gemm_weight


@pytest.mark.parametrize("m", [32, 64, 512])
@torch.inference_mode()
def test_entry_equals_package(m):
    if not E.available():
        pytest.skip("needs gfx950 with FlyDSL")
    a, b = build(m), build(m)
    assert E.gemma_add_rmsnorm_mxfp8_for_blocker(*a) is None
    out = E.gemma_add_rmsnorm_mxfp8_for(*a)
    ref = gemma_add_rmsnorm_mxfp8_for(*b)
    torch.cuda.synchronize()
    for x, y in zip(out, ref, strict=True):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))
