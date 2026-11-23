# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NVFP4 lightning-indexer entries: the prefill top-k reads its shape from the tensors and routes to the shape package
(``_ROUTES`` of ``prefill.py``); an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .prefill import (
    index_prefill_top16,
    index_prefill_top16_blocker,
    index_prefill_top16_layer_blocker,
    index_prefill_top16_shapes,
    warm_up_index_prefill_top16,
)

__all__ = [
    "available",
    "index_prefill_top16",
    "index_prefill_top16_blocker",
    "index_prefill_top16_layer_blocker",
    "index_prefill_top16_shapes",
    "warm_up_index_prefill_top16",
]
