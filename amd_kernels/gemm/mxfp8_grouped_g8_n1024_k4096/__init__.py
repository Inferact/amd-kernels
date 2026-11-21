# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Grouped MXFP8 GEMM, 8 groups of N 1024 x K 4096, bf16 out or MXFP8 out for the next GEMM."""


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


from .host import MAX_M, grouped_gemm, grouped_gemm_mxfp8, select, warmup_grouped_gemm
from .reference import grouped_gemm_mxfp8_reference, grouped_gemm_reference

__all__ = ["MAX_M", "available", "grouped_gemm", "grouped_gemm_mxfp8", "grouped_gemm_mxfp8_reference",
           "grouped_gemm_reference", "select", "warmup_grouped_gemm"]
