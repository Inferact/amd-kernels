# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Finite chunk references preserve empty rows and fully masked wave partials."""

import dataclasses

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import reference
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.query import pack_query
from .cases import make_case, require_gfx950
from .test_sparse_prefill import run, run_quant, run_pack, check_quant_bytes


@pytest.mark.parametrize("tokens,hkv", [(513, 1), (1089, 1), (1537, 1), (4099, 1)])
@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_empty_rows_and_masked_chunks(tokens, hkv, packed):
    require_gfx950()
    case = make_case([tokens - 3, 3], [tokens + 4096, 3], hkv, scales=(0.5, 2.0), seed=tokens)
    case.q.zero_()
    case.topk.fill_(-1)
    case.topk[:, ::2, -1] = case.pos[::2] // reference.BLK
    case.topk[:, :4].fill_(-1)
    # The final request starts at position zero: most waves have no visible keys.
    case.topk[:, -3:, -1] = 0
    reference_q = case.q
    if packed:
        case = dataclasses.replace(case, q=pack_query(case.q))
    out = run(case, "fp8")
    empty = (case.topk < 0).all(dim=(0, 2))
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[empty]) == 0
    assert torch.equal(out, run(case, "fp8"))
    rows = [4, 6, tokens - 3, tokens - 2, tokens - 1]
    ref = reference.sparse_prefill_reference(*case.args(q=reference_q), case.k_scale, case.v_scale, rows=rows, fp8=True)
    assert reference.rel_l2(out[rows], ref) < reference.REL_L2_TOL["fp8"]
    codes, scales = run_quant(case, "fp8")
    check_quant_bytes(out, codes, scales, tokens)
    if hkv == 1:
        assert torch.equal(run_pack(case, "fp8")[:tokens], reference.sp_pack_rows(out.flatten(1)))
