# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Residual add + Gemma RMSNorm + MXFP8 quant on one GPU, written in the code layout the MXFP8 GEMM of the next weight
reads: routes by the hidden size of the norm weight to the shape package (``_ROUTES``, imported on the first call of
its key)."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes

_ROUTES = Routes("gemma_add_rmsnorm_mxfp8_for", {6144: "amd_kernels.norm.gemma_add_rmsnorm_h6144"}, ("hidden",))
_OP = _ROUTES.op


def _module(key):
    return _ROUTES.module(key, "add_rmsnorm")


def gemma_add_rmsnorm_mxfp8_for_shapes() -> frozenset[int]:
    """The hidden sizes with a kernel."""
    return _ROUTES.shapes()


def gemma_add_rmsnorm_mxfp8_for(x, residual, weight, eps, gemm_weight):
    """(codes, scales, residual) of residual += x, Gemma RMSNorm, MXFP8 quant, in the layout the MXFP8 GEMM of
    ``gemm_weight`` [N, hidden] reads at this row count (the routed package's ``gemma_add_rmsnorm_mxfp8_for``)."""
    return _module(weight.numel()).gemma_add_rmsnorm_mxfp8_for(x, residual, weight, eps, gemm_weight)


def gemma_add_rmsnorm_mxfp8_for_blocker(x, residual, weight, eps, gemm_weight) -> str | None:
    """None when ``gemma_add_rmsnorm_mxfp8_for`` serves this call, else why not (host checks only)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    h = weight.numel()
    reason = _ROUTES.blocker(h)
    if reason is not None:
        return reason
    if not _module(h).add_rmsnorm_ok(x, residual, weight):
        return (f"{_OP}: x {tuple(x.shape)} {x.dtype}, residual {tuple(residual.shape)} {residual.dtype}, weight "
                f"{weight.dtype} (bf16 [rows, {h}] with unit column stride, contiguous bf16 weight, not compiling)")
    if gemm_weight.dim() != 2 or gemm_weight.shape[1] != h:
        return f"{_OP}: GEMM weight {tuple(gemm_weight.shape)} ([N, {h}])"
    return None


def warm_up_gemma_add_rmsnorm_mxfp8_for(hidden: int, device) -> None:
    """Compile the variants every MXFP8 GEMM layout of the hidden size reads (before graph capture)."""
    _module(hidden).warmup_for_gemm(device)
