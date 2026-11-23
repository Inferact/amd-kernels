# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 GEMMs: ``x @ W^T`` of a wide weight (lm_head, a vocabulary shard), optionally in the package's load-time
shuffled layout, routed by the weight's (N, K) (``_ROUTES``); and one GEMM writing fp32 and bf16 column blocks,
routed by (K, bf16 columns) (``_SPLIT``)."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Prepared, Routes, Rows, check, first, previous_form

_ROUTES = Routes(
    "bf16_gemm",
    {
        (50016, 6144): "amd_kernels.gemm.bf16_n50016_k6144",
        (100032, 6144): "amd_kernels.gemm.bf16_n100032_k6144",
        (129280, 5120): "amd_kernels.gemm.bf16_n129280_k5120",
    },
    ("n", "k"),
)
_SPLIT = Routes("bf16_gemm_f32_bf16", {(5120, 32): "amd_kernels.gemm.bf16_k5120_n1056_f32_n32"}, ("k", "bf16_columns"))
# rows at which the kernel beat vLLM's lm_head GEMM dispatch on MI355X (9-13% at 113-128, 3-4% at 496-512 rows)
_ROWS = {(129280, 5120): Rows((113, 128), (496, 512))}
_OP = _ROUTES.op


def _package(key):
    return _ROUTES.package(key)


def _gemm(key):
    return getattr(_package(key), f"bf16_gemm_n{key[0]}_k{key[1]}")


def bf16_gemm_shapes() -> frozenset:
    """(N, K) keys."""
    return _ROUTES.shapes()


def bf16_gemm_weight_blocker(weight, bias=None) -> str | None:
    """None when a package takes this weight (contiguous bf16 [N, K], no bias) and its shuffle: a load-time check."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    if weight.dim() != 2:
        return f"{_OP}: weight {tuple(weight.shape)} ([N, K])"
    key = tuple(weight.shape)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    if not _package(key).supports(weight, bias):
        return f"{_OP}: weight {weight.dtype}, bias {bias is not None} (contiguous bf16, no bias)"
    return None


def _call(x, weight, out) -> str | None:
    reason = bf16_gemm_weight_blocker(weight)
    if reason is not None:
        return reason
    n, k = weight.shape
    if x.dim() != 2 or x.shape[1] != k or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return f"{_OP}: x {tuple(x.shape)} {x.dtype} (contiguous bf16 [M, {k}])"
    if x.device != weight.device:
        return f"{_OP}: x on {x.device}, weight on {weight.device}"
    if out is not None and (tuple(out.shape) != (x.shape[0], n) or out.dtype != torch.bfloat16
                            or not out.is_contiguous() or out.device != x.device):
        return f"{_OP}: out {tuple(out.shape)} {out.dtype} (contiguous bf16 [{x.shape[0]}, {n}])"
    return None


def _weight_form(args, kwargs) -> bool:
    return len(args) == 1 or (len(args) == 2 and args[1] is None) or "bias" in kwargs


@previous_form(bf16_gemm_weight_blocker, _weight_form)
def bf16_gemm_blocker(x, weight, out=None, *, shuffled: bool = False) -> str | None:
    """None when a package serves ``x @ W^T`` for this call and beats the served GEMM at ``x``'s row count."""
    key = tuple(weight.shape)
    return first(_call(x, weight, out), _ROWS[key].blocker(_OP, x.shape[0]) if key in _ROWS else None)


def bf16_gemm(x, weight, out=None, *, shuffled: bool = False):
    """out [M, N] bf16 = x [M, K] bf16 @ W^T, fp32 accumulate; ``shuffled``: ``weight`` is in the load-time layout
    (``bf16_gemm_shuffle_weight``)."""
    key = tuple(weight.shape) if weight.dim() == 2 else None
    check(_ROUTES.blocker(key))
    return _gemm(key)(x, weight, out, shuffled=shuffled)


def bf16_gemm_shuffle_weight(weight):
    return _package(tuple(weight.shape)).shuffle_weight(weight)


def bf16_gemm_shuffle_weight_(weight):
    """``bf16_gemm_shuffle_weight`` in place (bit-exact permutation, no second copy of the weight)."""
    return _package(tuple(weight.shape)).shuffle_weight_(weight)


def warmup_bf16_gemm(device, n: int, k: int, shuffled: bool) -> None:
    """Compile every row-count variant of this layout (before graph capture)."""
    getattr(_package((n, k)), f"warmup_bf16_gemm_n{n}_k{k}")(device, shuffled)


class Bf16Gemm(Prepared):
    """``x @ W^T`` bound to one weight: ``h(x, out=None)``."""

    op = "bf16_gemm"

    def __init__(self, weight, shuffled: bool):
        self.weight, self.shuffled = weight, shuffled

    def blocker(self, x, out=None) -> str | None:
        return bf16_gemm_blocker(x, self.weight, out, shuffled=self.shuffled)

    def __call__(self, x, out=None):
        return bf16_gemm(x, self.weight, out, shuffled=self.shuffled)


def prepare_bf16_gemm(weight, *, shuffle: bool = True, warm_up: bool = True) -> Bf16Gemm | str:
    """The bound GEMM, compiled, or the reason no package serves ``weight``. ``shuffle``: permute the weight into the
    load-time layout in place (no second copy; the tensor then serves this GEMM only)."""
    reason = bf16_gemm_weight_blocker(weight)
    if reason is not None:
        return reason
    if shuffle:
        bf16_gemm_shuffle_weight_(weight)
    if warm_up:
        warmup_bf16_gemm(weight.device, *weight.shape, shuffle)
    return Bf16Gemm(weight, shuffle)


def bf16_gemm_f32_bf16_shapes() -> frozenset:
    """(K, bf16 columns) keys."""
    return _SPLIT.shapes()


def _split_blocker(x, weight, na: int) -> str | None:
    if x.dim() != 2 or weight.dim() != 2:
        return "bf16_gemm_f32_bf16: x and weight must be 2-D"
    key = (x.shape[1], weight.shape[0] - na)
    reason = _SPLIT.blocker(key)
    if reason is not None:
        return reason
    if not _SPLIT.package(key).supports(x, weight, na):
        return f"bf16_gemm_f32_bf16: the kernel does not take these inputs (na {na}, rows {x.shape[0]}, layout)"
    return None


def _na_form(args, kwargs) -> bool:
    return len(args) == 3 and type(args[2]) is int


@previous_form(_split_blocker, _na_form)
def bf16_gemm_f32_bf16_blocker(x, weight, out_a, out_b) -> str | None:
    """None when a package serves one bf16 GEMM over ``weight`` [na + nb, K] writing the first na columns into
    ``out_a`` (fp32) and the last nb into ``out_b`` (bf16)."""
    return _split_blocker(x, weight, weight.shape[0] - out_b.shape[1])


def bf16_gemm_f32_bf16(x, weight, out_a, out_b):
    """``out_a`` (fp32, may be None when na is 0) and ``out_b`` (bf16) columns of ``x @ W^T`` in one launch."""
    check(_split_blocker(x, weight, weight.shape[0] - out_b.shape[1]))
    return _SPLIT.package((x.shape[1], out_b.shape[1])).gemm(x, weight, out_a, out_b)


def bf16_gemm_f32_bf16_shuffle_weight(weight_a, weight_b):
    """fp32-column weight ``weight_a`` [na, K] (or None) and bf16-column weight ``weight_b`` [nb, K] -> the layout
    ``bf16_gemm_f32_bf16`` reads (load time)."""
    return _SPLIT.package((weight_b.shape[1], weight_b.shape[0])).shuffle_weight(weight_a, weight_b)
