# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""RMSNorm of a 5120-column row to MXFP8 codes and scales in one launch, FlyDSL (gfx950)."""



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


from .host import rmsnorm_mxfp8, warmup_rmsnorm_mxfp8
from .reference import dequantize_mxfp8, rmsnorm_mxfp8_reference, rmsnorm_reference

__all__ = ["available", "dequantize_mxfp8", "rmsnorm_mxfp8", "rmsnorm_mxfp8_reference", "rmsnorm_reference",
           "warmup_rmsnorm_mxfp8"]
