# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""RMSNorm + MXFP8 (routed by hidden, ``_ROUTES``) and the dual RMSNorm, q to MXFP8 and kv to bf16 (routed by
(q dim, kv dim), ``_DUAL``)."""

from amd_kernels.routes import Routes, check, first

_ROUTES = Routes("rmsnorm_mxfp8", {5120: "amd_kernels.norm.rmsnorm_mxfp8_h5120"})
_DUAL = Routes("rmsnorm_dual_mxfp8", {(1280, 512): "amd_kernels.norm.rmsnorm_dual_mxfp8_q1280_kv512"},
               ("q_dim", "kv_dim"))


def bf16(op: str, *tensors) -> str | None:
    return None if all(str(t.dtype).endswith("bfloat16") for t in tensors) else f"{op}: needs bf16 inputs"


def rmsnorm_mxfp8_shapes() -> frozenset:
    """Hidden sizes."""
    return _ROUTES.shapes()


def rmsnorm_mxfp8_blocker(x, weight, eps=None, **kwargs) -> str | None:
    """None when a package serves RMSNorm + MXFP8 quant of ``x`` [rows, hidden] bf16."""
    if x.dim() != 2:
        return "rmsnorm_mxfp8: x must be [rows, hidden]"
    return first(_ROUTES.blocker(x.shape[1]), bf16("rmsnorm_mxfp8", x, weight))


def rmsnorm_mxfp8(x, weight, eps, **kwargs):
    """``(codes, scales)`` of RMSNorm(x) * weight quantized to MXFP8; ``codes=`` / ``scale=`` as the shape
    package's ``rmsnorm_mxfp8``."""
    check(rmsnorm_mxfp8_blocker(x, weight))
    return _ROUTES.package(x.shape[1]).rmsnorm_mxfp8(x, weight, eps, **kwargs)


def warmup_rmsnorm_mxfp8(hidden: int, device=None) -> None:
    _ROUTES.package(hidden).warmup_rmsnorm_mxfp8(device)


def rmsnorm_dual_mxfp8_shapes() -> frozenset:
    """(q dim, kv dim) keys."""
    return _DUAL.shapes()


def rmsnorm_dual_mxfp8_blocker(q, kv, q_weight=None, kv_weight=None, eps=None, **kwargs) -> str | None:
    """None when a package serves the dual RMSNorm (q to MXFP8, kv to bf16) of ``q`` / ``kv`` [rows, dim] bf16."""
    if q.dim() != 2 or kv.dim() != 2:
        return "rmsnorm_dual_mxfp8: q and kv must be [rows, dim]"
    return first(_DUAL.blocker((q.shape[1], kv.shape[1])), bf16("rmsnorm_dual_mxfp8", q, kv))


def rmsnorm_dual_mxfp8(q, kv, q_weight, kv_weight, eps, **kwargs):
    """``(codes, scales, kv_out)``; keyword buffers as the shape package's ``rmsnorm_dual_mxfp8``."""
    check(rmsnorm_dual_mxfp8_blocker(q, kv))
    return _DUAL.package((q.shape[1], kv.shape[1])).rmsnorm_dual_mxfp8(q, kv, q_weight, kv_weight, eps, **kwargs)


def warmup_rmsnorm_dual_mxfp8(q_dim: int, kv_dim: int, device=None) -> None:
    _DUAL.package((q_dim, kv_dim)).warmup_rmsnorm_dual_mxfp8(device)
