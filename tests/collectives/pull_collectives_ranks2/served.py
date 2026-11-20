# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The served shapes of the pull collectives: the sequence-parallel prefill on 2 ranks, hidden 6144, the o_proj input
4096 columns per rank (32 heads x 128), up to ``SP_COLL_MAX_TOKENS`` tokens per forward (a DBO microbatch of a
65536-token step), shard rows a multiple of 32."""

WORLD = 2
HIDDEN = 6144
Q_SIZE = 4096  # this rank's heads x head dim: the all-to-all's d
SP_COLL_MAX_TOKENS = 65536 // 2
ROUTING_INTS = 64  # the gathered routing per row (int32)


def collectives_slot_bytes(hidden: int = HIDDEN, q_size: int = Q_SIZE) -> int:
    """The inputs of the largest phase at ``SP_COLL_MAX_TOKENS`` (the reduce-scatter's bf16 partial of every row, the
    shard's gathered codes + scales + routing, two bf16 gathers of the shard, alignment slack)."""
    rows = SP_COLL_MAX_TOKENS // WORLD
    padded = rows * WORLD
    assert padded * (q_size + q_size // 32) <= padded * hidden * 2
    return (
        padded * hidden * 2
        + rows * (hidden + hidden // 32 + ROUTING_INTS * 4)
        + 2 * rows * hidden * 2
        + 8 * 256
    )
