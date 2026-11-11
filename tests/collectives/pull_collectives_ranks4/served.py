# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The served shapes of the pull collectives: the sequence-parallel prefill on 4
ranks, hidden 6144, the o_proj input 2048 columns per rank (16 heads x 128), up
to ``SP_COLL_MAX_TOKENS`` tokens per forward (a DBO microbatch of a 65536-token
step), shard rows a multiple of 32."""

WORLD = 4
HIDDEN = 6144
Q_SIZE = 2048  # this rank's heads x head dim: the all-to-all's d
SP_COLL_MAX_TOKENS = 65536 // 2
ROUTING_INTS = 64  # the gathered routing per row (int32)


def collectives_slot_bytes(hidden: int = HIDDEN, q_size: int = Q_SIZE) -> int:
    """``sp_prefill.collectives_slot_bytes``: the inputs of the largest phase at
    ``SP_COLL_MAX_TOKENS`` (the reduce-scatter's bf16 partial of every row, the
    shard's gathered codes + scales + routing, two bf16 gathers of the shard,
    alignment slack)."""
    rows = SP_COLL_MAX_TOKENS // WORLD
    padded = rows * WORLD
    assert padded * (q_size + q_size // 32) <= padded * hidden * 2
    return (
        padded * hidden * 2
        + rows * (hidden + hidden // 32 + ROUTING_INTS * 4)
        + 2 * rows * hidden * 2
        + 8 * 256
    )
