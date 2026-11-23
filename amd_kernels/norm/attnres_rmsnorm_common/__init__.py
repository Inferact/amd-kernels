# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shape-agnostic entry of the all-reduce + AttnRes + RMSNorm: the communicator of a (hidden, world) key from the shape
package that serves it (``_ROUTES`` of ``host.py``), imported on first use, and the reason a call has no kernel
(``*_blocker``)."""
from .host import (
    allreduce_attnres_rmsnorm_blocker,
    allreduce_attnres_rmsnorm_shapes,
    allreduce_attnres_rmsnorm_unsupported_reason,
    attnres_comm,
    attnres_comm_blocker,
    attnres_comm_unsupported_reason,
    available,
    prepare_attnres_comm,
)

__all__ = ["allreduce_attnres_rmsnorm_blocker", "allreduce_attnres_rmsnorm_shapes", "attnres_comm",
           "attnres_comm_blocker", "available", "prepare_attnres_comm"]
