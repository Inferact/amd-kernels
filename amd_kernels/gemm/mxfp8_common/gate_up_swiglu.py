# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gate_up GEMM + clamped SwiGLU + MXFP8 out: routes a call by (N, K) of the gate_up weight to the shape package
(``_ROUTES``)."""

from amd_kernels.routes import Prepared, Routes, check, previous_form

_ROUTES = Routes("mxfp8_gate_up_swiglu_mxfp8", {(4608, 5120): "amd_kernels.gemm.mxfp8_n4608_k5120"}, ("n", "k"))


def mxfp8_gate_up_swiglu_mxfp8_shapes() -> frozenset:
    """(N, K) of the gate_up weight."""
    return _ROUTES.shapes()


def _rows(m: int, n: int, k: int, prequant: bool = False) -> str | None:
    """None when a package's tuned row for ``m`` rows (bf16 ``x``, or MXFP8 codes with ``prequant``) beat the served
    gate_up GEMM + activation + quant chain."""
    reason = _ROUTES.blocker((n, k))
    if reason is not None:
        return reason
    pkg = _ROUTES.package((n, k))
    table = pkg.kernel.SWIGLU_PREQUANT if prequant else pkg.kernel.SWIGLU
    if not pkg.serves(m, table=table):
        return f"mxfp8_gate_up_swiglu_mxfp8: at {m} rows the served chain is faster"
    return None


@previous_form(_rows)
def mxfp8_gate_up_swiglu_mxfp8_blocker(x, weight, weight_scale) -> str | None:
    """None when a package serves the gate_up + SwiGLU + MXFP8 of bf16 ``x`` and beats the served chain."""
    return _rows(x.shape[0], *weight.shape)


def mxfp8_gate_up_swiglu_mxfp8_prequant_blocker(codes, scale, weight, weight_scale) -> str | None:
    """The same for row-major MXFP8 codes."""
    return _rows(codes.shape[0], *weight.shape, prequant=True)


def mxfp8_gate_up_swiglu_mxfp8(x, weight, weight_scale):
    """``(codes, scales)`` of ``silu(min(g, 10)) * clamp(u, -10, 10)`` over ``x @ W^T`` (gate | up halves) in MXFP8;
    ``weight`` from ``preshuffle_mxfp8_weight``."""
    check(_ROUTES.blocker(tuple(weight.shape)))
    return _ROUTES.package(tuple(weight.shape)).gate_up_swiglu_mxfp8(x, weight, weight_scale)


def mxfp8_gate_up_swiglu_mxfp8_prequant(codes, scale, weight, weight_scale):
    """The same on row-major MXFP8 codes [M, K] and e8m0 scales [M, K / 32]."""
    check(_ROUTES.blocker(tuple(weight.shape)))
    return _ROUTES.package(tuple(weight.shape)).gate_up_swiglu_mxfp8_prequant(codes, scale, weight, weight_scale)


class Mxfp8GateUpSwiglu(Prepared):
    """gate_up + SwiGLU + MXFP8 bound to one weight: ``h(x)`` on bf16, ``h(codes, scales)`` on MXFP8 codes."""

    op = "mxfp8_gate_up_swiglu_mxfp8"

    def __init__(self, weight, weight_scale):
        self.weight, self.weight_scale = weight, weight_scale

    def blocker(self, x, x_scale=None) -> str | None:
        return _rows(x.shape[0], *self.weight.shape, prequant=x_scale is not None)

    def __call__(self, x, x_scale=None):
        if x_scale is None:
            return mxfp8_gate_up_swiglu_mxfp8(x, self.weight, self.weight_scale)
        return mxfp8_gate_up_swiglu_mxfp8_prequant(x, x_scale, self.weight, self.weight_scale)


def prepare_mxfp8_gate_up_swiglu_mxfp8(weight, weight_scale, *, weight_preshuffled=None,
                                       max_num_tokens: int = 1024) -> Mxfp8GateUpSwiglu | str:  # fmt: skip
    """The bound op, compiled, or the reason no package serves the weight; inputs as ``prepare_mxfp8_linear``."""
    from .prepared import prepare_mxfp8_linear

    reason = _ROUTES.blocker(tuple(weight.shape))
    if reason is not None:
        return reason
    linear = prepare_mxfp8_linear(weight, weight_scale, weight_preshuffled=weight_preshuffled,
                                  max_num_tokens=max_num_tokens)
    if isinstance(linear, str):
        return linear
    return Mxfp8GateUpSwiglu(linear.weight, linear.weight_scale)
