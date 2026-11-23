# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 ``x @ W^T`` of a wide weight [N, K] (a vocabulary shard) kept in the package's load-time shuffled layout: routes
by the weight's (N, K) to the shape package (``_ROUTES``, imported on the first call of its key), whose
``bf16_gemm_n<N>_k<K>`` runs it."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare

_ROUTES = Routes("bf16_shuffled_gemm", {
    (50016, 6144): "amd_kernels.gemm.bf16_n50016_k6144",
    (100032, 6144): "amd_kernels.gemm.bf16_n100032_k6144",
    (129280, 5120): "amd_kernels.gemm.bf16_n129280_k5120",
}, ("n", "k"))
_WEIGHT = Routes("bf16_shuffled_gemm_weight", _ROUTES.hosts, ("n", "k"))
_OP = _ROUTES.op


def _key(weight):
    return tuple(weight.shape) if weight.dim() == 2 else None


def _gemm(pkg, key):
    return getattr(pkg, f"bf16_gemm_n{key[0]}_k{key[1]}")


def bf16_shuffled_gemm_shapes() -> frozenset[tuple[int, int]]:
    """The (N, K) with a kernel."""
    return _ROUTES.shapes()


def bf16_shuffled_gemm_weight_blocker(weight, bias=None) -> str | None:
    """None when a package takes this weight (contiguous bf16 [N, K], no bias) and its shuffle: a load-time check."""
    if not routes.available():
        return f"{_WEIGHT.op}: needs gfx950 with FlyDSL"
    key = _key(weight)
    if key is None:
        return f"{_WEIGHT.op}: weight {tuple(weight.shape)} ([N, K])"
    reason = _WEIGHT.blocker(key)
    if reason is not None:
        return reason
    if not _WEIGHT.package(key).supports(weight, bias):
        return f"{_WEIGHT.op}: weight {weight.dtype}, bias {bias is not None} (contiguous bf16, no bias)"
    return None


def bf16_shuffled_gemm_shuffle_weight_(weight):
    """``weight`` permuted in place into the package's load-time layout (bit-exact; once, before graph capture; every
    later call passes ``shuffled=True``)."""
    return _WEIGHT.package(_key(weight)).shuffle_weight_(weight)


def prepare_bf16_shuffled_gemm_weight(weight, bias=None):
    """``weight`` shuffled in place (``bf16_shuffled_gemm_shuffle_weight_``), or the reason it cannot be."""
    return prepare(bf16_shuffled_gemm_weight_blocker(weight, bias), lambda: bf16_shuffled_gemm_shuffle_weight_(weight))


def warm_up_bf16_shuffled_gemm(device, n: int, k: int, shuffled: bool = True) -> None:
    """Compile every row-count variant of the contiguous call for this layout (before graph capture)."""
    getattr(_ROUTES.package((n, k)), f"warmup_bf16_gemm_n{n}_k{k}")(device, shuffled)


def bf16_shuffled_gemm(x, weight, out=None, *, shuffled: bool = True):
    """out [M, N] bf16 = x [M, K] bf16 @ W^T, fp32 accumulate; ``shuffled``: ``weight`` is in the load-time layout."""
    key = _key(weight)
    return _gemm(_ROUTES.package(key), key)(x, weight, out, shuffled=shuffled)


def bf16_shuffled_gemm_blocker(x, weight, out=None, *, shuffled: bool = True) -> str | None:
    """None when ``bf16_shuffled_gemm`` serves this call, else why not (host checks only)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(weight)
    if key is None:
        return f"{_OP}: weight {tuple(weight.shape)} ([N, K])"
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    n, k = key
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        return f"{_OP}: weight {weight.dtype} (contiguous bf16)"
    if x.dim() != 2 or x.shape[1] != k or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return f"{_OP}: x {tuple(x.shape)} {x.dtype} (contiguous bf16 [M, {k}])"
    if x.device != weight.device:
        return f"{_OP}: x on {x.device}, weight on {weight.device}"
    if out is not None and (tuple(out.shape) != (x.shape[0], n) or out.dtype != torch.bfloat16
                            or not out.is_contiguous() or out.device != x.device):
        return f"{_OP}: out {tuple(out.shape)} {out.dtype} (contiguous bf16 [{x.shape[0]}, {n}])"
    return None
