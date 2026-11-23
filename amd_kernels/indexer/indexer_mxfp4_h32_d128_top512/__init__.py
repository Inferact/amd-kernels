# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Indexer over the MXFP4 index cache (32 heads x 128 dims, 64- or 128-entry pages): the top-512 keys of decode rows,
from the dense context or from a candidate list, without materializing the logits."""


def available() -> bool:
    """gfx950 with FlyDSL importable."""
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


from .host import MAX_ENTRIES, Ragged, blocker, indexer_block_scores, indexer_candidate_topk, indexer_topk
from .reference import (
    block_scores_reference,
    candidate_topk_reference,
    candidates_reference,
    logits_reference,
    ragged_reference,
    random_inputs,
    select_reference,
    topk_matches,
    topk_reference,
)

__all__ = [
    "MAX_ENTRIES",
    "Ragged",
    "available",
    "block_scores_reference",
    "blocker",
    "candidate_topk_reference",
    "candidates_reference",
    "indexer_block_scores",
    "indexer_candidate_topk",
    "indexer_topk",
    "logits_reference",
    "ragged_reference",
    "random_inputs",
    "select_reference",
    "topk_matches",
    "topk_reference",
]
