# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Byte-exact references for head/token ownership exchanges."""

import torch


def shard_query_reference(head_shards: list[torch.Tensor], local_indices: torch.Tensor):
    """Concatenate heads then select tokens, preserving every stored bit."""
    dtype = head_shards[0].dtype
    return torch.cat([x.view(torch.uint8) for x in head_shards], dim=1).index_select(
        0, local_indices
    ).view(dtype)


def unshard_output_reference(token_shards: list[torch.Tensor], token_indices: list[torch.Tensor],
                             rank: int, heads: int):
    """Restore all tokens and select local heads, without arithmetic."""
    shape = (sum(x.shape[0] for x in token_shards), heads, token_shards[0].shape[-1])
    output = torch.empty(shape, dtype=token_shards[0].dtype, device=token_shards[0].device)
    for shard, indices in zip(token_shards, token_indices, strict=True):
        output.index_copy_(0, indices, shard[:, rank*heads:(rank+1)*heads])
    return output
