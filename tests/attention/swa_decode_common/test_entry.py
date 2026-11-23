# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry's bf16 output equals a direct call of the shape package bit for bit, and its reason is None (one GPU)."""

import pytest
import torch

from amd_kernels.attention import swa_decode_common as E
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128.oproj_input import swa_decode_fp8kv_oproj_input


def build(reqs, ql):
    g = torch.Generator(device="cuda").manual_seed(1)
    dev = "cuda"
    q = torch.randn(reqs * ql, 16 * 128, generator=g, device=dev).to(torch.bfloat16)
    cache = (torch.randn(16, 1, 64, 256, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    bt = torch.arange(reqs * 4, dtype=torch.int32, device=dev).view(reqs, 4)
    sl = torch.full((reqs,), 200, dtype=torch.int32, device=dev)
    qsl = torch.arange(reqs + 1, dtype=torch.int32, device=dev) * ql
    one = torch.tensor(1.0, device=dev)
    return (q, cache, bt, sl, qsl, ql, 0.088, 128, True, one, one)


@pytest.mark.parametrize("reqs,ql", [(1, 1), (4, 3)])
@torch.inference_mode()
def test_entry_equals_package(reqs, ql):
    if not E.available():
        pytest.skip("needs gfx950 with FlyDSL")
    a = build(reqs, ql)
    assert E.swa_decode_oproj_input_blocker(*a) is None
    out = E.swa_decode_oproj_input(*a)
    ref = swa_decode_fp8kv_oproj_input(*a)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
