# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch
from amd_kernels.collectives.dcp8_topk_gather_merge_top2048.reference import reference
from .workload import PAGE, SERVED_ROWS, STRIDE, make_block_table, make_plane

WORLD = 8


@pytest.mark.parametrize("ties", [0, 1, 3])
def test_reference_equals_vllm_merge_dcp_scores(ties):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    merge = pytest.importorskip("vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse").merge_dcp_scores
    for rows in (1, 12, 65, 192):
        planes = [make_plane(rows, r, rows, ties) for r in range(WORLD)]
        bt = make_block_table(rows, rows)
        scores = torch.stack([p[0] for p in planes])
        gathered = torch.cat(list(scores), dim=1).cuda()
        for rank in (0, 3, 7):
            idx = planes[rank][1].cuda()
            slots, indptr = merge(gathered, idx, bt.cuda(), rank, WORLD, PAGE, STRIDE)
            want_slots, want_indptr = reference(scores, planes[rank][1], bt, rank, PAGE, STRIDE)
            assert torch.equal(indptr.cpu(), want_indptr), (rows, rank)
            assert torch.equal(slots[: want_slots.numel()].cpu(), want_slots), (rows, rank)
