# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Norm op entries (``amd_kernels.routes``): RMSNorm + MXFP8, dual RMSNorm + MXFP8, the mHC seam, its finisher and
the step-end tail."""

from amd_kernels.routes import Routes, check, first

_RMSNORM = Routes("rmsnorm_mxfp8", {5120: "amd_kernels.norm.rmsnorm_mxfp8_h5120"})
_DUAL = Routes("rmsnorm_dual_mxfp8", {(1280, 512): "amd_kernels.norm.rmsnorm_dual_mxfp8_q1280_kv512"})
_SEAM = Routes("mhc_post_pre_rmsnorm", {(4, 5120): "amd_kernels.norm.mhc_post_pre_rmsnorm_h5120"})
_TAIL = Routes("mhc_post_collapse_rmsnorm", {(4, 5120): "amd_kernels.norm.mhc_post_collapse_rmsnorm_h5120"})
_FINISH = Routes("mhc_gate_finish", {(4, 5120): "amd_kernels.norm.mhc_gate_finish_rmsnorm_h5120"})
MHC_MAX_ROWS = 4096  # rows per call of the mHC entries
_BF16 = "bfloat16"


def _bf16(op: str, *tensors) -> str | None:
    return None if all(str(t.dtype).endswith(_BF16) for t in tensors) else f"{op}: needs bf16 inputs"


def rmsnorm_mxfp8_shapes() -> frozenset:
    """Hidden sizes."""
    return _RMSNORM.shapes()


def rmsnorm_mxfp8_blocker(x, weight) -> str | None:
    """None when a package serves RMSNorm + MXFP8 quant of ``x`` [rows, hidden] bf16."""
    if x.dim() != 2:
        return "rmsnorm_mxfp8: x must be [rows, hidden]"
    return first(_RMSNORM.blocker(x.shape[1]), _bf16("rmsnorm_mxfp8", x, weight))


def rmsnorm_mxfp8(x, weight, eps, **kwargs):
    """``(codes, scales)`` of RMSNorm(x) * weight quantized to MXFP8; ``codes=`` / ``scale=`` as the shape
    package's ``rmsnorm_mxfp8``."""
    check(rmsnorm_mxfp8_blocker(x, weight))
    return _RMSNORM.package(x.shape[1]).rmsnorm_mxfp8(x, weight, eps, **kwargs)


def warmup_rmsnorm_mxfp8(hidden: int, device=None) -> None:
    _RMSNORM.package(hidden).warmup_rmsnorm_mxfp8(device)


def rmsnorm_dual_mxfp8_shapes() -> frozenset:
    """(q dim, kv dim) keys."""
    return _DUAL.shapes()


def rmsnorm_dual_mxfp8_blocker(q, kv) -> str | None:
    """None when a package serves the dual RMSNorm (q to MXFP8, kv to bf16) of ``q`` / ``kv`` [rows, dim] bf16."""
    if q.dim() != 2 or kv.dim() != 2:
        return "rmsnorm_dual_mxfp8: q and kv must be [rows, dim]"
    return first(_DUAL.blocker((q.shape[1], kv.shape[1])), _bf16("rmsnorm_dual_mxfp8", q, kv))


def rmsnorm_dual_mxfp8(q, kv, q_weight, kv_weight, eps, **kwargs):
    """``(codes, scales, kv_out)``; keyword buffers as the shape package's ``rmsnorm_dual_mxfp8``."""
    check(rmsnorm_dual_mxfp8_blocker(q, kv))
    return _DUAL.package((q.shape[1], kv.shape[1])).rmsnorm_dual_mxfp8(q, kv, q_weight, kv_weight, eps, **kwargs)


def warmup_rmsnorm_dual_mxfp8(q_dim: int, kv_dim: int, device=None) -> None:
    _DUAL.package((q_dim, kv_dim)).warmup_rmsnorm_dual_mxfp8(device)


def _residual_blocker(op: str, routes: Routes, residual) -> str | None:
    if residual.dim() != 3:
        return f"{op}: residual must be [rows, streams, hidden]"
    return first(
        routes.blocker(tuple(residual.shape[1:])),
        None if residual.shape[0] <= MHC_MAX_ROWS else f"{op}: {residual.shape[0]} rows above {MHC_MAX_ROWS}",
        _bf16(op, residual),
    )


def mhc_post_pre_rmsnorm_shapes() -> frozenset:
    """(residual streams, hidden) keys."""
    return _SEAM.shapes()


def mhc_post_pre_rmsnorm_blocker(residual) -> str | None:
    """None when a package serves the mHC seam (post-mix, gates, collapse, RMSNorm) of ``residual``
    [rows, streams, hidden] bf16."""
    return _residual_blocker("mhc_post_pre_rmsnorm", _SEAM, residual)


def mhc_post_pre_rmsnorm(residual, fn_packed, hc_scale, hc_base, norm_weight, **kwargs):
    """The mHC seam in one launch; keyword arguments (eps values, the previous step's mixes, ``quant``,
    ``aux_mean``) and outputs as the shape package's ``mhc_post_pre_rmsnorm``."""
    check(mhc_post_pre_rmsnorm_blocker(residual))
    return _SEAM.package(tuple(residual.shape[1:])).mhc_post_pre_rmsnorm(residual, fn_packed, hc_scale, hc_base,
                                                                          norm_weight, **kwargs)


def mhc_pack_gate_weight(fn, streams: int = 4, hidden: int = 5120):
    """The fp32 gate weight ``fn`` [mixes, streams * hidden] in the seam's packed layout (load time)."""
    return _SEAM.package((streams, hidden)).pack_gate_weight(fn)


def warmup_mhc_post_pre_rmsnorm(streams: int, hidden: int, device=None) -> None:
    _SEAM.package((streams, hidden)).warmup_mhc_post_pre_rmsnorm(device)


def mhc_post_collapse_rmsnorm_shapes() -> frozenset:
    """(residual streams, hidden) keys."""
    return _TAIL.shapes()


def mhc_post_collapse_rmsnorm_blocker(residual) -> str | None:
    """None when a package serves the step-end post-mix + collapse + RMSNorm of ``residual`` [rows, streams, hidden]."""
    return _residual_blocker("mhc_post_collapse_rmsnorm", _TAIL, residual)


def mhc_post_collapse_rmsnorm(residual, pre_mix, norm_weight, **kwargs):
    """The mHC step-end tail; keyword arguments and outputs as the shape package's ``mhc_post_collapse_rmsnorm``."""
    check(mhc_post_collapse_rmsnorm_blocker(residual))
    return _TAIL.package(tuple(residual.shape[1:])).mhc_post_collapse_rmsnorm(residual, pre_mix, norm_weight,
                                                                               **kwargs)


def warmup_mhc_post_collapse_rmsnorm(streams: int, hidden: int, device=None) -> None:
    _TAIL.package((streams, hidden)).warmup_mhc_post_collapse_rmsnorm(device)


def mhc_gate_finish_shapes() -> frozenset:
    """(residual streams, hidden) keys."""
    return _FINISH.shapes()


def mhc_gate_finish_blocker(partials, x1, streams: int = 4) -> str | None:
    """None when a package finishes the seam from the post-mix GEMM's tile records ``partials`` [rows, tiles, slot]
    and the new residual stream ``x1`` [rows, hidden]."""
    if partials.dim() != 3 or x1.dim() != 2:
        return "mhc_gate_finish: partials must be [rows, tiles, slot], x1 [rows, hidden]"
    return first(
        _FINISH.blocker((streams, x1.shape[1])),
        None if x1.shape[0] <= MHC_MAX_ROWS else f"mhc_gate_finish: {x1.shape[0]} rows above {MHC_MAX_ROWS}",
        None if 1 <= partials.shape[1] <= 640 else f"mhc_gate_finish: {partials.shape[1]} tiles outside 1..640",
        _bf16("mhc_gate_finish", x1),
    )


def mhc_gate_finish(partials, x1, hc_scale, hc_base, norm_weight, **kwargs):
    """Gates, collapse and RMSNorm (+ MXFP8) from the tile records; as the shape package's ``mhc_gate_finish``."""
    check(mhc_gate_finish_blocker(partials, x1))
    return _FINISH.package((4, x1.shape[1])).mhc_gate_finish(partials, x1, hc_scale, hc_base, norm_weight, **kwargs)
