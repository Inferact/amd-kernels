# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DCP8 DSA query path: q_b projection, W_UK absorption and the head all-gather (FlyDSL, gfx950)."""

from .comm import QueryGatherComm
from .host import q_b_allgather_wuk_all_heads, q_b_wuk_allgather_latent
from .kernel_qb import pack_q_b
from .kernel_wuk import pack_w_uk

__all__ = [
    "QueryGatherComm",
    "pack_q_b",
    "pack_w_uk",
    "q_b_allgather_wuk_all_heads",
    "q_b_wuk_allgather_latent",
]
