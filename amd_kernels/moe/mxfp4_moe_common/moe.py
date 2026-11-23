# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP4 routed MoE block with its MXFP8 shared expert (router, top-k, experts, shared expert): routes the weight
build by (hidden, intermediate, experts, top_k), read from the checkpoint-layout expert tensors w13 [E, 2I, H / 2] /
w2 [E, H, I / 2] and the caller's top_k, to the shape package (``_ROUTES``, imported on the first build of its key);
the op routes by the package that built its weights."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare

_NAMES = ("hidden", "intermediate", "experts", "top_k")
_ROUTES = Routes("mxfp4_moe", {
    (6144, 768, 128, 4): "amd_kernels.moe.mxfp4_h6144_i768_e128_top4",
    (6144, 1536, 128, 4): "amd_kernels.moe.mxfp4_h6144_i1536_e128_top4",
}, _NAMES)
_WEIGHTS = Routes("mxfp4_moe_weights", _ROUTES.hosts, _NAMES)
_OP = _ROUTES.op


def _key(w13, w2, top_k):
    if w13.dim() != 3 or w2.dim() != 3:
        return None
    return (w2.shape[1], 2 * w2.shape[2], w13.shape[0], top_k)


def _u8(t):
    return t.view(torch.uint8)


def mxfp4_moe_shapes() -> frozenset[tuple[int, int, int, int]]:
    """The (hidden, intermediate per GPU, experts, top_k) with a kernel."""
    return _ROUTES.shapes()


def mxfp4_moe_weights(router_weight, correction_bias, w13, w13_scale, w2, w2_scale, shared_w13, shared_w13_scale,
                      shared_w2, shared_w2_scale, *, top_k: int, routed_scaling_factor: float, swiglu=None):
    """The routed package's ``MoeWeights`` (with its ``SharedWeights``) from the checkpoint layouts: MXFP4 routed
    experts + e8m0 scales, MXFP8 shared expert + scales, the fp32 router [E, H] and correction bias [E]. ``swiglu``:
    the model's SwiGLU-OAI (alpha, limit, beta), checked by the blocker. Not warmed up (``warm_up_mxfp4_moe``)."""
    pkg = _WEIGHTS.package(_key(w13, w2, top_k))
    shared = pkg.SharedWeights.from_row_major(_u8(shared_w13), _u8(shared_w13_scale), _u8(shared_w2),
                                              _u8(shared_w2_scale))
    return pkg.MoeWeights.from_row_major(router_weight, correction_bias, _u8(w13), _u8(w13_scale), _u8(w2),
                                         _u8(w2_scale), shared=shared, routed_scaling_factor=routed_scaling_factor)


def mxfp4_moe_weights_blocker(router_weight, correction_bias, w13, w13_scale, w2, w2_scale, shared_w13,
                              shared_w13_scale, shared_w2, shared_w2_scale, *, top_k: int,
                              routed_scaling_factor: float, swiglu=None) -> str | None:
    """None when ``mxfp4_moe_weights`` takes these tensors, else why not (host checks only)."""
    op = _WEIGHTS.op
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = _key(w13, w2, top_k)
    if key is None:
        return f"{op}: w13 {tuple(w13.shape)}, w2 {tuple(w2.shape)} ([E, 2I, H / 2], [E, H, I / 2])"
    reason = _WEIGHTS.blocker(key)
    if reason is not None:
        return reason
    pkg = _WEIGHTS.package(key)
    reason = pkg.unsupported_reason(w13, w13_scale, w2, w2_scale, shared_w13, shared_w13_scale, shared_w2,
                                    shared_w2_scale, swiglu=pkg.SWIGLU_OAI if swiglu is None else swiglu)
    if reason is not None:
        return f"{op}: {reason}"
    h, _, e, _ = key
    if router_weight.dtype != torch.float32 or tuple(router_weight.shape) != (e, h):
        return f"{op}: router weight {tuple(router_weight.shape)} {router_weight.dtype} (fp32 [{e}, {h}])"
    b = correction_bias
    if b is None or b.dtype != torch.float32 or tuple(b.shape) != (e,):
        return f"{op}: correction bias {None if b is None else (tuple(b.shape), b.dtype)} (fp32 [{e}])"
    return None


def prepare_mxfp4_moe_weights(router_weight, correction_bias, w13, w13_scale, w2, w2_scale, shared_w13,
                              shared_w13_scale, shared_w2, shared_w2_scale, *, top_k: int,
                              routed_scaling_factor: float, swiglu=None):
    """``mxfp4_moe_weights(...)``, or the reason it cannot be built."""
    args = (router_weight, correction_bias, w13, w13_scale, w2, w2_scale, shared_w13, shared_w13_scale, shared_w2,
            shared_w2_scale)
    kw = dict(top_k=top_k, routed_scaling_factor=routed_scaling_factor, swiglu=swiglu)
    return prepare(mxfp4_moe_weights_blocker(*args, **kw), lambda: mxfp4_moe_weights(*args, **kw))


def _owner(w):
    key = _ROUTES.owner(type(w))
    if key is None:
        raise ValueError(f"{_OP}: weights {type(w).__qualname__} not built by mxfp4_moe_weights")
    return _ROUTES.package(key)


def warm_up_mxfp4_moe(weights, *, partials: bool = False) -> None:
    """Compile every variant of the reduced mode (and of the partials mode with ``partials``) before serving."""
    _owner(weights).warm_up_mxfp4_moe(weights, partials=partials)


def mxfp4_moe_set_dbo_query(query) -> None:
    """``query()`` -> whether the current call runs inside a DBO microbatch (the shared expert then stays on the
    caller's stream); set on every routed package."""
    for key in _ROUTES.hosts:
        _ROUTES.package(key).set_dbo_query(query)


def mxfp4_moe(x, weights, *, return_partials: bool = False):
    """The MoE block of normed bf16 ``x`` [M, H]: the reduced bf16 [M, H] output, or with ``return_partials`` the
    (slots [M, top_k, H], shared [M, H]) partials (the routed package's ``mxfp4_moe``)."""
    return _owner(weights).mxfp4_moe(x, weights, return_partials=return_partials)


def mxfp4_moe_blocker(x, weights, *, return_partials: bool = False) -> str | None:
    """None when ``mxfp4_moe`` serves this call, else why not (host checks only)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _ROUTES.owner(type(weights))
    if key is None:
        return f"{_OP}: weights {type(weights).__qualname__} not built by mxfp4_moe_weights"
    if getattr(weights, "shared_weights", None) is None:
        return f"{_OP}: weights without the shared expert"
    if not isinstance(x, torch.Tensor):
        return None
    h = key[0]
    if x.dim() != 2 or x.shape[1] != h or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return f"{_OP}: x {tuple(x.shape)} {x.dtype} (contiguous bf16 [M, {h}])"
    return None
