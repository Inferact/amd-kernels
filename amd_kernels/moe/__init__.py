# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MoE op entries: routers, routed experts, fused expert parallelism, latent MoE tails. Callers import ops from here
only (``amd_kernels/README.md``, "Op entries"); each family's ``__all__`` is re-exported."""

from amd_kernels.routes import available

from . import latent_moe_common, mxfp4_experts_common, mxfp4_moe_common, router_topk_common  # noqa: F401
from .latent_moe_common import *  # noqa: F403
from .mxfp4_experts_common import *  # noqa: F403
from .mxfp4_moe_common import *  # noqa: F403
from .router_topk_common import *  # noqa: F403

__all__ = sorted({
    "available",
    *latent_moe_common.__all__,
    *mxfp4_experts_common.__all__,
    *mxfp4_moe_common.__all__,
    *router_topk_common.__all__,
})
