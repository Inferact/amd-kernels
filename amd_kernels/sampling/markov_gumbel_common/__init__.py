# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Fused Markov draft chain for any (vocabulary, Markov rank) with a package: one entry, routed by ``w1``'s shape
(``_ROUTES`` of ``host.py``); an uncovered call gets a reason from ``markov_gumbel_fused_blocker``."""

from .host import (
    available,
    markov_gumbel_fused,
    markov_gumbel_fused_blocker,
    markov_gumbel_fused_shapes,
    markov_gumbel_fused_unsupported_reason,
    markov_max_rows,
    markov_scratch_dwords,
)

__all__ = ["available", "markov_gumbel_fused", "markov_gumbel_fused_blocker", "markov_gumbel_fused_shapes",
           "markov_max_rows", "markov_scratch_dwords"]
