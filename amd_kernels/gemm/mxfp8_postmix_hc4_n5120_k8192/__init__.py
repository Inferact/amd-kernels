# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""wo_b MXFP8 GEMM (N 5120 x K 8192) with the mHC post-mix as its epilogue: pre-quantized A in, new residual out."""


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


from .host import (
    MAX_ROWS, pack_gate_weight, postmix_gates_gemm, postmix_gemm, preshuffle_weight, select, select_gates,
    warmup_postmix_gemm,
)
from .reference import postmix_gemm_reference

__all__ = ["MAX_ROWS", "available", "pack_gate_weight", "postmix_gates_gemm", "postmix_gemm", "postmix_gemm_reference",
           "preshuffle_weight", "select", "select_gates", "warmup_postmix_gemm"]
