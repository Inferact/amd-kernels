# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``fp8_sparse_mla_decode`` through the entry: at the smallest and the largest served row count (the whole-row and the
tile-loop kernel), with bf16 q and with the pre-quantized q8, into a fresh or a given ``out``, (out, lse) equal the
shape package's direct call byte for byte and the reason is None."""

import pytest
import torch
from amd_kernels.attention import mla_decode_common as E
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 import fp8_sparse_mla_decode as direct
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.reference import quantize_q

from tests.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.cases import SM_SCALE, make_batch, served_counts

from .cases import assert_same, clone, require_gfx950

ROWS = (5, 256)


def case(rows, q8, given_out, seed):
    gen = torch.Generator().manual_seed(seed)
    q, cache, indptr, slots = make_batch(served_counts(rows, gen, pad=rows // 8), gen)
    kw = {}
    if q8:
        kw["q8"], q = quantize_q(q), None
    if given_out:
        kw["out"] = torch.zeros((rows, 96, 512), dtype=torch.bfloat16, device="cuda")
    return (q, cache, indptr, slots, SM_SCALE), kw


@pytest.mark.parametrize("given_out", [False, True])
@pytest.mark.parametrize("q8", [False, True])
@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_equals_the_package(rows, q8, given_out):
    require_gfx950()
    args, kw = case(rows, q8, given_out, rows)
    assert E.fp8_sparse_mla_decode_blocker(*args, **kw) is None
    want_a, want_kw = clone(args), clone(kw)
    want = direct(*want_a, **want_kw)
    got = E.fp8_sparse_mla_decode(*args, **kw)
    torch.cuda.synchronize()
    assert_same(got, want, ("sparse", rows, q8, given_out))
    if given_out:
        assert got[0] is kw["out"]


@torch.inference_mode()
def test_layout_reasons():
    require_gfx950()
    args, kw = case(8, False, False, 1)
    q, cache, indptr, slots, scale = args
    for bad, text in (
        ((q, cache, indptr.long(), slots, scale), "operand layout (supports_fp8_sparse_mla_decode)"),
        ((q, cache[:, :640], indptr, slots, scale), "no kernel for heads=96, d_qk=576, record=640"),
        ((q.float(), cache, indptr, slots, scale), "operand layout (supports_fp8_sparse_mla_decode)"),
    ):
        why = E.fp8_sparse_mla_decode_blocker(*bad)
        assert why is not None and text in why, why
    why = E.fp8_sparse_mla_decode_blocker(*args, out=torch.zeros((8, 96, 512), device="cuda"))
    assert "supports_fp8_sparse_mla_decode" in why, why
