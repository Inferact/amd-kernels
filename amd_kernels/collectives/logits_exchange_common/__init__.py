# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""lm_head shard GEMM + logits exchange entry: the factory reads its shape from keyword dims and the group and builds
the communicator of the exchange and GEMM packages that serve it (``host._ROUTES``, imported on first use); the caller
runs its methods, and ``logits_gemm_exchange_blocker`` says whether a ``gemm_exchange`` call is served."""
from amd_kernels.routes import available

from .host import (LogitsExchange, logits_exchange_comm, logits_exchange_comm_blocker,
                   logits_exchange_comm_unsupported_reason, logits_gemm_exchange_blocker, logits_gemm_exchange_shapes,
                   logits_gemm_exchange_unsupported_reason, prepare_logits_exchange_comm)

__all__ = ["LogitsExchange", "available", "logits_exchange_comm", "logits_exchange_comm_blocker",
           "logits_gemm_exchange_blocker", "logits_gemm_exchange_shapes", "prepare_logits_exchange_comm"]
