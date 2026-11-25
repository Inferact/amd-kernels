# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Grouped MXFP8 GEMM (``groups`` x N x K, 32 x 32 weight blocks), bf16 or MXFP8 out: routes a call by (groups, N, K)
of the weight to the shape package (``_ROUTES``); the MXFP8-out op takes the rows above that package's ``MAX_M`` to
the batched package (``_LARGE_M``), which reads the same layouts."""

from amd_kernels.routes import Prepared, Routes, check, first, previous_form

_NAMES = ("groups", "n", "k")
_ROUTES = Routes("mxfp8_grouped_gemm", {(8, 1024, 4096): "amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096"}, _NAMES)
_LARGE_M = Routes("mxfp8_grouped_gemm_mxfp8", {(8, 1024, 4096): "amd_kernels.gemm.mxfp8_bmm_mxfp8_out_n1024_k4096"},
                  _NAMES)
_LARGE_MAX_M = 16384


def _key(weight, weight_scale):
    groups = weight_scale.shape[0]
    return (groups, weight.shape[0] // groups, weight.shape[-1])


def mxfp8_grouped_gemm_shapes() -> frozenset:
    """(groups, N, K) keys."""
    return _ROUTES.shapes()


def _rows(m: int, key, mxfp8_out: bool) -> str | None:
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    limit = _ROUTES.package(key).MAX_M
    if mxfp8_out and key in _LARGE_M.hosts:
        limit = _LARGE_MAX_M
    return None if 1 <= m <= limit else f"mxfp8_grouped_gemm: {m} rows outside 1..{limit}"


def _integer_form(m: int, groups: int, n: int, k: int) -> str | None:
    return _rows(m, (groups, n, k), False)


@previous_form(_integer_form)
def mxfp8_grouped_gemm_blocker(x, x_scale, weight, weight_scale, out=None) -> str | None:
    """None when a package serves this grouped GEMM with bf16 out."""
    return _rows(x.shape[0], _key(weight, weight_scale), False)


def mxfp8_grouped_gemm_mxfp8_blocker(x, x_scale, weight, weight_scale) -> str | None:
    """None when a package serves this grouped GEMM with MXFP8 out."""
    return _rows(x.shape[0], _key(weight, weight_scale), True)


def mxfp8_grouped_gemm(x, x_scale, weight, weight_scale, out=None):
    """bf16 ``out[m, g] = x[m, g] @ W[g]^T``: ``x`` e4m3 [M, G * K] with row-major e8m0 scales, ``weight`` e4m3
    [G * N, K] in AITER's (16, 16) preshuffle with 32 x 32 scales [G, N / 32, K / 32]."""
    key = _key(weight, weight_scale)
    check(_rows(x.shape[0], key, False))
    return _ROUTES.package(key).grouped_gemm(x, x_scale, weight, weight_scale, out)


def mxfp8_grouped_gemm_mxfp8(x, x_scale, weight, weight_scale):
    """The same GEMM's output quantized to MXFP8: ``(codes [M, G * N], scales [M, G * N / 32])``."""
    key = _key(weight, weight_scale)
    m = x.shape[0]
    check(_rows(m, key, True))
    pkg = _ROUTES.package(key)
    if m <= pkg.MAX_M:
        return pkg.grouped_gemm_mxfp8(x, x_scale, weight, weight_scale)
    return _LARGE_M.package(key).mxfp8_bmm_mxfp8_out(x, x_scale, weight, weight_scale)


def warmup_mxfp8_grouped_gemm(device, groups: int, n: int, k: int) -> None:
    _ROUTES.package((groups, n, k)).warmup_grouped_gemm(device)


class Mxfp8GroupedGemm(Prepared):
    """The grouped GEMM bound to one weight: ``h(x, x_scale)`` -> bf16, or ``(codes, scales)`` with ``mxfp8_out``."""

    op = "mxfp8_grouped_gemm"

    def __init__(self, weight, weight_scale, mxfp8_out: bool):
        self.weight, self.weight_scale, self.mxfp8_out = weight, weight_scale, mxfp8_out

    def blocker(self, x, x_scale) -> str | None:
        return first(
            None if x.is_contiguous() and x_scale.is_contiguous() else "mxfp8_grouped_gemm: needs contiguous x",
            _rows(x.shape[0], _key(self.weight, self.weight_scale), self.mxfp8_out),
        )

    def __call__(self, x, x_scale):
        if self.mxfp8_out:
            return mxfp8_grouped_gemm_mxfp8(x, x_scale, self.weight, self.weight_scale)
        return mxfp8_grouped_gemm(x, x_scale, self.weight, self.weight_scale)


def prepare_mxfp8_grouped_gemm(weight, weight_scale, groups: int, *, mxfp8_out: bool = True,
                               max_num_tokens: int = 1024) -> Mxfp8GroupedGemm | str:  # fmt: skip
    """The bound grouped GEMM, compiled, or the reason no package serves it.

    Args:
        weight: e4m3 [G * N, K] in AITER's (16, 16) preshuffle (used as is, no copy).
        weight_scale: e8m0 32 x 32 block scales [G * N / 32, K / 32] or [G, N / 32, K / 32] (viewed, no copy).
        max_num_tokens: compile the MXFP8-out rows above the decode package's up to this many.
    """
    import torch

    if weight.dim() != 2 or weight.dtype != torch.float8_e4m3fn or weight.shape[0] % groups:
        return f"mxfp8_grouped_gemm: weight {tuple(weight.shape)} {weight.dtype} is not e4m3 [G * N, K]"
    n, k = weight.shape[0] // groups, weight.shape[1]
    reason = _ROUTES.blocker((groups, n, k))
    if reason is not None:
        return reason
    if weight_scale.numel() != groups * (n // 32) * (k // 32):
        return f"mxfp8_grouped_gemm: weight scales {tuple(weight_scale.shape)} are not 32 x 32 blocks"
    scale = weight_scale.view(torch.uint8).reshape(groups, n // 32, k // 32)
    warmup_mxfp8_grouped_gemm(weight.device, groups, n, k)
    pkg = _ROUTES.package((groups, n, k))
    if mxfp8_out and max_num_tokens > pkg.MAX_M and (groups, n, k) in _LARGE_M.hosts:
        x = torch.zeros(max_num_tokens, groups * k, dtype=torch.float8_e4m3fn, device=weight.device)
        xs = torch.zeros(max_num_tokens, groups * k // 32, dtype=torch.uint8, device=weight.device)
        _LARGE_M.package((groups, n, k)).mxfp8_bmm_mxfp8_out(x[: pkg.MAX_M + 1], xs[: pkg.MAX_M + 1], weight, scale)
    return Mxfp8GroupedGemm(weight, scale, mxfp8_out)
