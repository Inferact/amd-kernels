# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Activation op entries (``amd_kernels.routes``): clamped SiLU-mul + MXFP8."""

from amd_kernels.routes import Routes, check, first

_CLAMP_SILU = Routes("clamp_silu_mul_mxfp8", {2304: "amd_kernels.activation.clamp_silu_mul_mxfp8_i2304"})


def clamp_silu_mul_mxfp8_shapes() -> frozenset:
    """Intermediate sizes (half the gate_up width)."""
    return _CLAMP_SILU.shapes()


def clamp_silu_mul_mxfp8_blocker(gate_up) -> str | None:
    """None when a package serves ``silu(min(g, limit)) * clamp(u, -limit, limit)`` + MXFP8 of ``gate_up``
    [rows, 2 * inter] bf16."""
    if gate_up.dim() != 2 or gate_up.shape[1] % 2:
        return "clamp_silu_mul_mxfp8: gate_up must be [rows, 2 * inter]"
    return first(
        _CLAMP_SILU.blocker(gate_up.shape[1] // 2),
        None if str(gate_up.dtype).endswith("bfloat16") else "clamp_silu_mul_mxfp8: needs bf16 gate_up",
    )


def clamp_silu_mul_mxfp8(gate_up, limit: float = 10.0, **kwargs):
    """``(codes, scales)``; ``codes=`` / ``scale=`` as the shape package's ``clamp_silu_mul_mxfp8``."""
    check(clamp_silu_mul_mxfp8_blocker(gate_up))
    return _CLAMP_SILU.package(gate_up.shape[1] // 2).clamp_silu_mul_mxfp8(gate_up, limit, **kwargs)


def warmup_clamp_silu_mul_mxfp8(inter: int, device=None) -> None:
    _CLAMP_SILU.package(inter).warmup_clamp_silu_mul_mxfp8(device)
