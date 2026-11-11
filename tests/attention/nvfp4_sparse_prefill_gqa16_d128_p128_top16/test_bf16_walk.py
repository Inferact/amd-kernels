# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The fused page-union walk handles empty tiles and a late first valid slot."""
import dataclasses

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import reference
from .cases import make_case, require_gfx950
from .test_sparse_prefill import check_reference, run, run_pack, run_quant, check_quant_bytes


@pytest.mark.parametrize("hkv", [1])
@pytest.mark.parametrize("variant", ["bf16", "fp8"])
@torch.inference_mode()
def test_empty_union_and_late_first_page(hkv, variant):
    require_gfx950()
    # Above the small-tile dispatch, with an incomplete final four-query tile.
    case = make_case([1537], [1664], hkv, seed=140, picks="shared")
    topk = torch.full_like(case.topk, -1)
    # Alternate completely empty tiles and tiles whose only live slot is last.
    # Each live tile shares page zero across all four queries.
    active = torch.arange(1537, device="cuda") % 8 >= 4
    topk[:, active, -1] = 0
    case = dataclasses.replace(case, topk=topk)
    out = run(case, variant)
    assert torch.equal(out[~active], torch.zeros_like(out[~active]))
    rows = [4, 5, 6, 7, 12, 1020, 1532]
    if variant == "fp8":
        ref = reference.fp8_chunk_prefill_reference(*case.args(), rows=rows)
        assert reference.rel_l2(out[rows], ref) < 2.5e-3
        cos = torch.nn.functional.cosine_similarity(out[rows].double(), ref.double(), dim=-1)
        assert cos.min() >= 0.99999
    else:
        check_reference(case, out, variant, rows=rows)
    for _ in range(2):
        assert torch.equal(out, run(case, variant))
    target = torch.full_like(out, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(case, variant, out=target)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, target)
    codes, scales = run_quant(case, variant)
    check_quant_bytes(out, codes, scales, 1537)
    assert torch.equal(run_pack(case, variant)[:1537], reference.sp_pack_rows(out.flatten(1)))
