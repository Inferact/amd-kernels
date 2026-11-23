# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""GEMM op entries: MXFP8 dense linears, gate_up + SwiGLU, mHC post-mix and grouped GEMMs, bf16 GEMMs. Callers import
ops from here only (``amd_kernels/README.md``, "Op entries"); each family's ``__all__`` is re-exported."""

from amd_kernels.routes import available

from . import bf16_common, bf16_gemm_common, mxfp8_common  # noqa: F401
from .bf16_common import *  # noqa: F403
from .bf16_gemm_common import *  # noqa: F403
from .mxfp8_common import *  # noqa: F403
from .mxfp8_common import mxfp8_postmix_blocker  # noqa: F401  previous name, warns once

__all__ = sorted({"available", *bf16_common.__all__, *bf16_gemm_common.__all__, *mxfp8_common.__all__})
