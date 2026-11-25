# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sampling-side kernels: logits checks, top-k / top-p, Gumbel draws, block verification.  Sampling op entries: Markov
draft chains, rejection sampling, top-p masks, batch-sharded sampling. Callers import ops from here only
(``amd_kernels/README.md``, "Op entries"); each family's ``__all__`` is re-exported."""

from amd_kernels.routes import available

from . import (  # noqa: F401
    batch_sharded_sampling_common,
    markov_chain_common,
    markov_gumbel_common,
    rejection_sample_common,
    top_p_mask_common,
)
from .batch_sharded_sampling_common import *  # noqa: F403
from .markov_chain_common import *  # noqa: F403
from .markov_gumbel_common import *  # noqa: F403
from .rejection_sample_common import *  # noqa: F403
from .top_p_mask_common import *  # noqa: F403

__all__ = sorted({
    "available",
    *batch_sharded_sampling_common.__all__,
    *markov_chain_common.__all__,
    *markov_gumbel_common.__all__,
    *rejection_sample_common.__all__,
    *top_p_mask_common.__all__,
})
