# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Route post-convolution KDA verification by checkpoint head shape."""

from amd_kernels.routes import Routes

_VERIFY = Routes("kda_verify_rmsnorm", {(12, 128): "amd_kernels.attention.kda_verify_rmsnorm_h12_d128"})
kda_verify_rmsnorm_shapes = _VERIFY.shapes


def kda_verify_rmsnorm_unsupported_reason(
    qkv,
    gate,
    beta,
    out_gate,
    alog,
    bias,
    state,
    offsets,
    slots,
    weight,
    correction,
    key_gate,
    out,
    lower_bound,
    eps,
):
    key = tuple(state.shape[1:3]) if state.ndim == 4 else None
    why = _VERIFY.blocker(key)
    if why is not None:
        return why
    return _VERIFY.package(key).unsupported_reason(
        qkv,
        gate,
        beta,
        out_gate,
        alog,
        bias,
        state,
        offsets,
        slots,
        weight,
        correction,
        key_gate,
        out,
        lower_bound,
        eps,
    )


def kda_verify_rmsnorm(
    qkv,
    gate,
    beta,
    out_gate,
    alog,
    bias,
    state,
    offsets,
    slots,
    weight,
    correction,
    key_gate,
    out,
    lower_bound,
    eps,
):
    """Fill the output and compact recovery records for each live token of a packed batch."""
    key = tuple(state.shape[1:3]) if state.ndim == 4 else None
    return _VERIFY.package(key).kda_verify_rmsnorm(
        qkv,
        gate,
        beta,
        out_gate,
        alog,
        bias,
        state,
        offsets,
        slots,
        weight,
        correction,
        key_gate,
        out,
        lower_bound,
        eps,
    )
