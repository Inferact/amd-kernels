# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Batch-sharded sampling entries on a TP group: the communicators read their shape from the vocabulary and the group
and route to the shape package (``_ROUTES`` of ``host.py``); an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .host import (
    batch_sharded_markov_blocker,
    batch_sharded_sampling,
    batch_sharded_sampling_blocker,
    batch_sharded_sampling_config_blocker,
    batch_sharded_sampling_shapes,
    prepare_batch_sharded_sampling,
)

__all__ = [
    "available",
    "batch_sharded_markov_blocker",
    "batch_sharded_sampling",
    "batch_sharded_sampling_blocker",
    "batch_sharded_sampling_config_blocker",
    "batch_sharded_sampling_shapes",
    "prepare_batch_sharded_sampling",
]
