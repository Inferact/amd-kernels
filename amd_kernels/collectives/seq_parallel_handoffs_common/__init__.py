# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sequence-parallel decoder-layer hand-off entries on a TP group: the communicator reads its shape from the layer's
sizes and the group and routes to the shape package (``_ROUTES`` of ``host.py``); an uncovered call gets a reason
from ``*_blocker``."""

from amd_kernels.routes import available

from .host import (
    prepare_seq_parallel_handoffs,
    seq_parallel_handoffs,
    seq_parallel_handoffs_blocker,
    seq_parallel_handoffs_layer_blocker,
    seq_parallel_handoffs_shapes,
    seq_parallel_min_tokens,
    seq_parallel_shard,
)

__all__ = [
    "available",
    "prepare_seq_parallel_handoffs",
    "seq_parallel_handoffs",
    "seq_parallel_handoffs_blocker",
    "seq_parallel_handoffs_layer_blocker",
    "seq_parallel_handoffs_shapes",
    "seq_parallel_min_tokens",
    "seq_parallel_shard",
]
