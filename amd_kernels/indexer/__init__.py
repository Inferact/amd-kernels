# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Indexer op entries: MXFP4 paged / prefill / top-512 indexers, NVFP4 prefill indexer, query RoPE + MXFP4 quant.
Callers import ops from here only (``amd_kernels/README.md``, "Op entries"); each family's ``__all__`` is re-
exported."""

from amd_kernels.routes import available

from . import mxfp4_indexer_common, mxfp4_indexer_topk_common, nvfp4_indexer_common  # noqa: F401
from .mxfp4_indexer_common import *  # noqa: F403
from .mxfp4_indexer_topk_common import *  # noqa: F403
from .nvfp4_indexer_common import *  # noqa: F403
from .mxfp4_indexer_topk_common import mxfp4_indexer_blocker  # noqa: F401  previous name, warns once

__all__ = sorted({
    "available",
    *mxfp4_indexer_common.__all__,
    *mxfp4_indexer_topk_common.__all__,
    *nvfp4_indexer_common.__all__,
})
