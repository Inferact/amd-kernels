# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The gate GEMM + biased sqrt-softplus top-k router: routes a call by (hidden, experts, top_k) to the shape package
(``_ROUTES``)."""

from amd_kernels.routes import Prepared, Routes, check

_ROUTES = Routes(
    "router_topk",
    {
        (5120, 384, 6): "amd_kernels.moe.router_sqrtsoftplus_top6_h5120_e384",
        (5120, 128, 3): "amd_kernels.moe.router_sqrtsoftplus_top3_h5120_e128",
    },
    ("hidden", "experts", "top_k"),
)


def router_topk_shapes() -> frozenset:
    """(hidden, experts, top_k) keys."""
    return _ROUTES.shapes()


def router_topk_blocker(x, weight=None, bias=None, top_k: int | None = None, routed_scaling_factor: float = 1.5,
                        **kwargs) -> str | None:  # fmt: skip
    """None when a package serves the gate GEMM + biased sqrt-softplus top-``top_k`` of ``x`` [rows, hidden] over
    ``bias`` [experts] experts."""
    if type(bias) is int and top_k is None:  # the previous form: (x, bias, top_k)
        weight, bias, top_k = None, weight, bias
    if x.dim() != 2 or bias.dim() != 1:
        return "router_topk: x must be [rows, hidden], bias [experts]"
    return _ROUTES.blocker((x.shape[1], bias.shape[0], top_k))


def router_topk(x, weight, bias, top_k: int, routed_scaling_factor: float = 1.5, **kwargs):
    """``(topk_weights, topk_ids)`` of each row; keyword arguments (``is_padding``, output buffers,
    ``weight_shuffled``) as the shape package's ``router_topk``."""
    check(router_topk_blocker(x, weight, bias, top_k))
    return _ROUTES.package((x.shape[1], bias.shape[0], top_k)).router_topk(x, weight, bias, routed_scaling_factor,
                                                                            **kwargs)


def router_shuffle_gate(weight, top_k: int):
    """The gate ``weight`` [experts, hidden] in the load-time layout of ``router_topk(..., weight_shuffled=True)``."""
    return _ROUTES.package((weight.shape[1], weight.shape[0], top_k)).shuffle_gate(weight)


def warmup_router_topk(hidden: int, experts: int, top_k: int, device=None) -> None:
    """Compiles every row-count bucket of the router (before graph capture)."""
    _ROUTES.package((hidden, experts, top_k)).warmup_router_topk(device)


class RouterTopk(Prepared):
    """The router bound to its shuffled gate and bias: ``h(x, routed_scaling_factor=1.5, **kwargs)``."""

    op = "router_topk"

    def __init__(self, weight, bias, top_k: int):
        self.weight, self.bias, self.top_k = weight, bias, top_k

    def blocker(self, x, routed_scaling_factor: float = 1.5, **kwargs) -> str | None:
        return router_topk_blocker(x, self.weight, self.bias, self.top_k)

    def __call__(self, x, routed_scaling_factor: float = 1.5, **kwargs):
        return router_topk(x, self.weight, self.bias, self.top_k, routed_scaling_factor, weight_shuffled=True,
                           **kwargs)


def prepare_router_topk(weight, bias, top_k: int) -> RouterTopk | str:
    """The router with its gate ``weight`` [experts, hidden] shuffled and every row bucket compiled, or the reason no
    package serves it."""
    key = (weight.shape[1], weight.shape[0], top_k)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    warmup_router_topk(*key, weight.device)
    return RouterTopk(router_shuffle_gate(weight, top_k), bias, top_k)
