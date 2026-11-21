# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""lm_head shard GEMM + logits exchange entry: the factory reads its shape from keyword dims and the group and builds
the shape package's communicator (imported on first use); the caller runs its methods, and
``logits_gemm_exchange_unsupported_reason`` says whether a ``gemm_exchange`` call is served."""
import functools


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable."""
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


from .host import (logits_exchange_comm, logits_exchange_comm_unsupported_reason,  # noqa: E402
                   logits_gemm_exchange_shapes, logits_gemm_exchange_unsupported_reason)

__all__ = ["available", "logits_exchange_comm", "logits_exchange_comm_unsupported_reason",
           "logits_gemm_exchange_shapes", "logits_gemm_exchange_unsupported_reason"]
