# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 GEMM of the wq_b (N 32768 x K 1280) for decode row counts: bf16 activation in, bf16 out."""


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


from .host import mxfp8_linear, preshuffle_weight, serves, warmup_mxfp8_linear
from .reference import K, N, mxfp8_linear_reference, quantize_mxfp8

__all__ = [
    "K",
    "N",
    "available",
    "mxfp8_linear",
    "mxfp8_linear_reference",
    "preshuffle_weight",
    "quantize_mxfp8",
    "serves",
    "warmup_mxfp8_linear",
]
