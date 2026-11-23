# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 [M, 5120] x [NA + NB, 5120]^T, the first NA columns in fp32 and the last NB in bf16, for up to 1024 rows, FlyDSL
(gfx950). The contract is ``reference.py``."""

from .host import gemm, shuffle_weight, supports

__all__ = ["gemm", "shuffle_weight", "supports"]
