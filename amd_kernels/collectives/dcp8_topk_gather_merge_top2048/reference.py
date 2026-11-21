# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Global top-2048 of eight ranks' local top-2048 scores, as this rank's physical KV slots.

Candidates are the rank-major concatenation of the eight [rows, 2048] fp32 score planes. A score maps to an int32
key that orders as the float does (NaN below every score, -0.0 below +0.0). The threshold is the 2048th largest key
of a row. Keys above it are selected; keys equal to it fill the remaining room in rank-major position order, counting
every lower rank's equal candidates whether or not they are live. A candidate of this rank is live when its local index
is >= 0 and may only be selected when live. The rank keeps its selected candidates in position order and maps local
index j to block_table[row, j // page_size] * block_stride + j % page_size. Output: int32 CSR slots, packed over rows,
and the int32 row pointer. Integer-exact against vLLM's merge_dcp_scores; every other output element is unspecified.
Padding candidates of other ranks must score -inf.
"""
import torch

K = 2048


def score_key(x: torch.Tensor) -> torch.Tensor:
    bits = x.contiguous().view(torch.int32)
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return torch.where((bits & 0x7FFFFFFF) > 0x7F800000, torch.iinfo(torch.int32).min, key)


def reference(scores, local_idx, block_table, rank, page_size, block_stride):
    """scores: [ranks, rows, 2048] fp32; local_idx: [rows, 2048] int32; block_table: [rows, blocks] int32."""
    world, rows, k = scores.shape
    assert k == K
    keys = score_key(scores).permute(1, 0, 2).reshape(rows, world * K).to(torch.int64)
    thr = keys.topk(K, dim=1).values[:, -1:]
    room = K - (keys > thr).sum(1, keepdim=True)
    own = keys.view(rows, world, K)[:, rank]
    live = local_idx >= 0
    prior = (keys.view(rows, world, K)[:, :rank] == thr.unsqueeze(1)).sum((1, 2)).unsqueeze(1)
    tied = (live & (own == thr)).to(torch.int64)
    keep = (live & (own > thr)) | ((tied.cumsum(1) - tied < room - prior) & (tied != 0))
    j = local_idx.clamp(min=0).to(torch.int64)
    page = block_table.to(torch.int64).gather(1, j // page_size)
    slot = (page * block_stride + j % page_size).to(torch.int32)
    counts = keep.sum(1, dtype=torch.int32)
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device=scores.device)
    indptr[1:] = counts.cumsum(0)
    return slot[keep], indptr
