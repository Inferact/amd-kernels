# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Collective op entries: logits exchange, top-k gather-merge, sequence-parallel handoffs. Callers import ops from here
only (``amd_kernels/README.md``, "Op entries"); each family's ``__all__`` is re-exported."""

from amd_kernels.routes import available

from . import logits_exchange_common, seq_parallel_handoffs_common, topk_gather_merge_common  # noqa: F401
from .logits_exchange_common import *  # noqa: F403
from .seq_parallel_handoffs_common import *  # noqa: F403
from .topk_gather_merge_common import *  # noqa: F403

__all__ = sorted({
    "available",
    *logits_exchange_common.__all__,
    *seq_parallel_handoffs_common.__all__,
    *topk_gather_merge_common.__all__,
})
