# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The mHC seam (post-mix, gates, collapse, RMSNorm), its finisher from the post-mix GEMM's tile records and the
step-end tail: each routed by (residual streams, hidden) (``_SEAM``, ``_FINISH``, ``_TAIL``)."""

from amd_kernels.routes import Routes, Rows, arity, check, first, previous_form

from ..rmsnorm_mxfp8_common.host import bf16

_NAMES = ("streams", "hidden")
_SEAM = Routes("mhc_post_pre_rmsnorm", {(4, 5120): "amd_kernels.norm.mhc_post_pre_rmsnorm_h5120"}, _NAMES)
_TAIL = Routes("mhc_post_collapse_rmsnorm", {(4, 5120): "amd_kernels.norm.mhc_post_collapse_rmsnorm_h5120"}, _NAMES)
_FINISH = Routes("mhc_gate_finish", {(4, 5120): "amd_kernels.norm.mhc_gate_finish_rmsnorm_h5120"}, _NAMES)
MHC_MAX_ROWS = 4096  # rows per call of the mHC entries
# rows at which the seam (with its MXFP8 output and the next GEMM) beat AITER's MXFP8 mega-mHC seam on MI355X
# (vLLM DeepSeek-V4.1 DEP8 routes: ties at 289-384 and 721-736, AITER ahead at 385-512 and from about 744 rows)
_SEAM_ROWS = Rows((1, 288), (513, 720))


def _residual(op: str, routes: Routes, residual) -> str | None:
    if residual.dim() != 3:
        return f"{op}: residual must be [rows, streams, hidden]"
    return first(
        routes.blocker(tuple(residual.shape[1:])),
        None if residual.shape[0] <= MHC_MAX_ROWS else f"{op}: {residual.shape[0]} rows above {MHC_MAX_ROWS}",
        bf16(op, residual),
    )


def mhc_post_pre_rmsnorm_shapes() -> frozenset:
    """(residual streams, hidden) keys."""
    return _SEAM.shapes()


def _seam(residual) -> str | None:
    return _residual("mhc_post_pre_rmsnorm", _SEAM, residual)


@previous_form(_seam, arity(1))
def mhc_post_pre_rmsnorm_blocker(residual, fn_packed=None, hc_scale=None, hc_base=None, norm_weight=None, *,
                                 aux_mean: bool = False, **kwargs) -> str | None:  # fmt: skip
    """None when a package serves the mHC seam of ``residual`` [rows, streams, hidden] bf16 and beats AITER's seam at
    its row count (the variant that also emits the streams' mean, ``aux_mean``, wins at any row count)."""
    return first(_seam(residual), None if aux_mean else _SEAM_ROWS.blocker("mhc_post_pre_rmsnorm", residual.shape[0]))


def mhc_post_pre_rmsnorm(residual, fn_packed, hc_scale, hc_base, norm_weight, **kwargs):
    """The mHC seam in one launch; keyword arguments (eps values, the previous step's mixes, ``quant``,
    ``aux_mean``) and outputs as the shape package's ``mhc_post_pre_rmsnorm``."""
    check(_seam(residual))
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


def mhc_post_collapse_rmsnorm_blocker(residual, pre_mix=None, norm_weight=None, **kwargs) -> str | None:
    """None when a package serves the step-end post-mix + collapse + RMSNorm of ``residual`` [rows, streams, hidden]."""
    return _residual("mhc_post_collapse_rmsnorm", _TAIL, residual)


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


def mhc_gate_finish_blocker(partials, x1, hc_scale=None, hc_base=None, norm_weight=None, **kwargs) -> str | None:
    """None when a package finishes the seam from the post-mix GEMM's tile records ``partials`` [rows, tiles, slot]
    and the new residual stream ``x1`` [rows, hidden] (four streams)."""
    if partials.dim() != 3 or x1.dim() != 2:
        return "mhc_gate_finish: partials must be [rows, tiles, slot], x1 [rows, hidden]"
    return first(
        _FINISH.blocker((4, x1.shape[1])),
        None if x1.shape[0] <= MHC_MAX_ROWS else f"mhc_gate_finish: {x1.shape[0]} rows above {MHC_MAX_ROWS}",
        None if 1 <= partials.shape[1] <= 640 else f"mhc_gate_finish: {partials.shape[1]} tiles outside 1..640",
        bf16("mhc_gate_finish", x1),
    )


def mhc_gate_finish(partials, x1, hc_scale, hc_base, norm_weight, **kwargs):
    """Gates, collapse and RMSNorm (+ MXFP8) from the tile records; as the shape package's ``mhc_gate_finish``."""
    check(mhc_gate_finish_blocker(partials, x1))
    return _FINISH.package((4, x1.shape[1])).mhc_gate_finish(partials, x1, hc_scale, hc_base, norm_weight, **kwargs)
