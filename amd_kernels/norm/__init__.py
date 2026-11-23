# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Norm op entries: RMSNorm + MXFP8, the mHC seams, Gemma add-RMSNorm and the all-reduce norms. Callers import ops from
here only (``amd_kernels/README.md``, "Op entries"); each family's ``__all__`` is re-exported."""

from amd_kernels.routes import available

from . import attnres_rmsnorm_common, gemma_add_rmsnorm_common, mhc_common, rmsnorm_mxfp8_common  # noqa: F401
from .attnres_rmsnorm_common import *  # noqa: F403
from .gemma_add_rmsnorm_common import *  # noqa: F403
from .mhc_common import *  # noqa: F403
from .rmsnorm_mxfp8_common import *  # noqa: F403

__all__ = sorted({
    "available",
    *attnres_rmsnorm_common.__all__,
    *gemma_add_rmsnorm_common.__all__,
    *mhc_common.__all__,
    *rmsnorm_mxfp8_common.__all__,
})
