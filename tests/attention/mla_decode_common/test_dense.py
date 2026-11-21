# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``fp8_dense_mla_decode`` through the entry: query length 1 and 4, the smallest and the largest request count, into
fresh or given outputs, (out, lse) equal the shape package's direct call byte for byte and the reason is None."""

import pytest
import torch
from amd_kernels.attention import mla_decode_common as E
from amd_kernels.attention.fp8_dense_mla_decode_h96_kv512 import fp8_dense_mla_decode as direct

from tests.attention.fp8_dense_mla_decode_h96_kv512.test_decode import SM_SCALE, make_batch, scales

from .cases import assert_same, clone, require_gfx950

REQS = (1, 63)
RANK = 5


def case(qlen, reqs, given):
    batch = make_batch(qlen, [300 + 37 * i for i in range(reqs)], RANK, seed=reqs)
    rows = reqs * qlen
    kw = {}
    if given:
        kw["out"] = torch.zeros((rows, 96, 512), dtype=torch.bfloat16, device="cuda")
        kw["lse"] = torch.zeros((rows, 96), dtype=torch.float32, device="cuda")
    return (*batch, *scales(), SM_SCALE, qlen, RANK, 8), kw


@pytest.mark.parametrize("given", [False, True])
@pytest.mark.parametrize("reqs", REQS)
@pytest.mark.parametrize("qlen", [1, 4])
@torch.inference_mode()
def test_equals_the_package(qlen, reqs, given):
    require_gfx950()
    args, kw = case(qlen, reqs, given)
    assert E.fp8_dense_mla_decode_unsupported_reason(*args, **kw) is None
    want_a, want_kw = clone(args), clone(kw)
    want = direct(*want_a, **want_kw)
    got = E.fp8_dense_mla_decode(*args, **kw)
    torch.cuda.synchronize()
    assert_same(got, want, ("dense", qlen, reqs, given))


@torch.inference_mode()
def test_layout_reasons():
    require_gfx950()
    args, _ = case(4, 2, False)
    a = list(args)
    a[0] = a[0].to(torch.bfloat16)
    assert "operand layout (supports_fp8_dense_mla_decode)" in E.fp8_dense_mla_decode_unsupported_reason(*a)
    a = list(args)
    a[5] = a[5].cpu()
    assert "operand layout (supports_fp8_dense_mla_decode)" in E.fp8_dense_mla_decode_unsupported_reason(*a)
    lse = torch.zeros((8, 96), dtype=torch.bfloat16, device="cuda")
    assert "operand layout (lse fp32" in E.fp8_dense_mla_decode_unsupported_reason(*args, lse=lse)
