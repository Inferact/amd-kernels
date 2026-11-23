# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 GEMM of the shared-expert gate_up (N 4608 x K 5120) for decode row counts: bf16 activation in, bf16 out."""


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


from .host import gate_up_swiglu_mxfp8, gate_up_swiglu_mxfp8_prequant, mxfp8_linear, preshuffle_weight, serves, warmup_mxfp8_linear
from .reference import K, N, gate_up_swiglu_reference, mxfp8_linear_reference, quantize_mxfp8

__all__ = [
    "K",
    "N",
    "available",
    "gate_up_swiglu_mxfp8",
    "gate_up_swiglu_mxfp8_prequant",
    "gate_up_swiglu_reference",
    "mxfp8_linear",
    "mxfp8_linear_reference",
    "preshuffle_weight",
    "quantize_mxfp8",
    "serves",
    "warmup_mxfp8_linear",
]
