# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""GEMM op entries (``amd_kernels.routes``) besides the MXFP8 dense linears (``gemm.mxfp8_common.gemm``:
``mxfp8_linear``, ``mxfp8_gemm``, ``mxfp8_linear_blocker``, ``tuned_shapes``): gate_up + SwiGLU + MXFP8, the
mHC post-mix GEMMs, the grouped MXFP8 GEMM, the bf16 GEMMs."""

from amd_kernels.routes import Routes, check, first

_GATE_UP_SWIGLU = Routes("mxfp8_gate_up_swiglu_mxfp8", {(4608, 5120): "amd_kernels.gemm.mxfp8_n4608_k5120"})
_POSTMIX = Routes(
    "mxfp8_postmix_gemm",
    {
        (5120, 8192, False): "amd_kernels.gemm.mxfp8_postmix_hc4_n5120_k8192",
        (5120, 2304, True): "amd_kernels.gemm.mxfp8_postmix_add_hc4_n5120_k2304",
    },
)
_GROUPED = Routes("mxfp8_grouped_gemm", {(8, 1024, 4096): "amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096"})
_BF16 = Routes("bf16_gemm", {(129280, 5120): "amd_kernels.gemm.bf16_n129280_k5120"})
_BF16_SPLIT = Routes("bf16_gemm_f32_bf16", {(5120, 32): "amd_kernels.gemm.bf16_k5120_n1056_f32_n32"})


def mxfp8_gate_up_swiglu_mxfp8_shapes() -> frozenset:
    """(N, K) of the gate_up weight."""
    return _GATE_UP_SWIGLU.shapes()


def mxfp8_gate_up_swiglu_mxfp8_blocker(m: int, n: int, k: int, prequant: bool = False) -> str | None:
    """None when a package's tuned row for ``m`` rows (bf16 ``x``, or MXFP8 codes with ``prequant``) beat the served
    gate_up GEMM + activation + quant chain."""
    reason = _GATE_UP_SWIGLU.blocker((n, k))
    if reason is not None:
        return reason
    pkg = _GATE_UP_SWIGLU.package((n, k))
    table = pkg.kernel.SWIGLU_PREQUANT if prequant else pkg.kernel.SWIGLU
    if not pkg.serves(m, table=table):
        return f"mxfp8_gate_up_swiglu_mxfp8: at {m} rows the served chain is faster"
    return None


def mxfp8_gate_up_swiglu_mxfp8(x, weight, weight_scale):
    """``(codes, scales)`` of ``silu(min(g, 10)) * clamp(u, -10, 10)`` over ``x @ W^T`` (gate | up halves) in MXFP8;
    ``weight`` from ``mxfp8_common.gemm.preshuffle_mxfp8_weight``."""
    return _GATE_UP_SWIGLU.package(tuple(weight.shape)).gate_up_swiglu_mxfp8(x, weight, weight_scale)


def mxfp8_gate_up_swiglu_mxfp8_prequant(codes, scale, weight, weight_scale):
    """The same on row-major MXFP8 codes [M, K] and e8m0 scales [M, K / 32]."""
    return _GATE_UP_SWIGLU.package(tuple(weight.shape)).gate_up_swiglu_mxfp8_prequant(codes, scale, weight,
                                                                                      weight_scale)


def mxfp8_postmix_shapes() -> frozenset:
    """(N, K, routed add) keys."""
    return _POSTMIX.shapes()


def mxfp8_postmix_blocker(m: int, n: int, k: int, routed: bool) -> str | None:
    """None when a package serves the MXFP8 GEMM with the mHC post-mix epilogue (``routed``: plus the routed MoE add)
    at ``m`` rows."""
    key = (n, k, bool(routed))
    reason = _POSTMIX.blocker(key)
    if reason is not None:
        return reason
    limit = _POSTMIX.package(key).MAX_ROWS
    return None if 1 <= m <= limit else f"mxfp8_postmix_gemm: {m} rows outside 1..{limit}"


def _postmix(weight, routed):
    return _POSTMIX.package((weight.shape[0], weight.shape[1], routed is not None))


def mxfp8_postmix_gemm(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, routed=None, out=None):
    """The new residual streams ``post_mix * (codes @ W^T [+ routed]) + comb_mix @ residual``; as the shape
    package's ``postmix_gemm``."""
    check(mxfp8_postmix_blocker(codes.shape[0], weight.shape[0], weight.shape[1], routed is not None))
    pkg = _postmix(weight, routed)
    if routed is None:
        return pkg.postmix_gemm(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, out)
    return pkg.postmix_gemm(codes, scales, weight, weight_scale, residual, routed, post_mix, comb_mix, out)


def mxfp8_postmix_gates_gemm(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, pre_mix, fn_packed,
                             routed=None, *, aux_mean: bool = False):  # fmt: skip
    """The post-mix GEMM that also writes the next seam's gate tile records (``norm.mhc_gate_finish`` consumes them);
    as the shape package's ``postmix_gates_gemm``."""
    check(mxfp8_postmix_blocker(codes.shape[0], weight.shape[0], weight.shape[1], routed is not None))
    pkg = _postmix(weight, routed)
    if routed is None:
        return pkg.postmix_gates_gemm(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, pre_mix,
                                      fn_packed, aux_mean=aux_mean)
    return pkg.postmix_gates_gemm(codes, scales, weight, weight_scale, residual, routed, post_mix, comb_mix, pre_mix,
                                  fn_packed, aux_mean=aux_mean)


def mxfp8_postmix_preshuffle_weight(weight, weight_scale, routed: bool):
    """E4M3 [N, K] + e8m0 [N, K / 32] -> the layouts the post-mix GEMMs read (load time)."""
    return _POSTMIX.package((weight.shape[0], weight.shape[1], routed)).preshuffle_weight(weight, weight_scale)


def mxfp8_postmix_pack_gate_weight(fn, n: int, k: int, routed: bool):
    """The fp32 gate weight in the gates GEMM's packed layout (load time)."""
    return _POSTMIX.package((n, k, routed)).pack_gate_weight(fn)


def warmup_mxfp8_postmix_gemm(weight, weight_scale, routed: bool) -> None:
    _POSTMIX.package((weight.shape[0], weight.shape[1], routed)).warmup_postmix_gemm(weight, weight_scale)


def mxfp8_grouped_gemm_shapes() -> frozenset:
    """(groups, N, K) keys."""
    return _GROUPED.shapes()


def mxfp8_grouped_gemm_blocker(m: int, groups: int, n: int, k: int) -> str | None:
    """None when a package serves the grouped MXFP8 GEMM (``groups`` x N x K, 32 x 32 weight blocks) at ``m`` rows."""
    key = (groups, n, k)
    reason = _GROUPED.blocker(key)
    if reason is not None:
        return reason
    limit = _GROUPED.package(key).MAX_M
    return None if 1 <= m <= limit else f"mxfp8_grouped_gemm: {m} rows outside 1..{limit}"


def _grouped_key(x, weight, weight_scale):
    groups = weight_scale.shape[0]
    return x.shape[0], (groups, weight.shape[0] // groups, weight.shape[-1])


def mxfp8_grouped_gemm(x, x_scale, weight, weight_scale, out=None):
    """bf16 ``out[m, g] = x[m, g] @ W[g]^T``: ``x`` e4m3 [M, G * K] with row-major e8m0 scales, ``weight`` e4m3
    [G * N, K] in AITER's (16, 16) preshuffle with 32 x 32 scales [G, N / 32, K / 32]."""
    m, key = _grouped_key(x, weight, weight_scale)
    check(mxfp8_grouped_gemm_blocker(m, *key))
    return _GROUPED.package(key).grouped_gemm(x, x_scale, weight, weight_scale, out)


def mxfp8_grouped_gemm_mxfp8(x, x_scale, weight, weight_scale):
    """The same GEMM's output quantized to MXFP8: ``(codes [M, G * N], scales [M, G * N / 32])``."""
    m, key = _grouped_key(x, weight, weight_scale)
    check(mxfp8_grouped_gemm_blocker(m, *key))
    return _GROUPED.package(key).grouped_gemm_mxfp8(x, x_scale, weight, weight_scale)


def warmup_mxfp8_grouped_gemm(device, groups: int, n: int, k: int) -> None:
    _GROUPED.package((groups, n, k)).warmup_grouped_gemm(device)


def bf16_gemm_shapes() -> frozenset:
    """(N, K) keys."""
    return _BF16.shapes()


def bf16_gemm_blocker(weight, bias=None) -> str | None:
    """None when a package serves ``x @ W^T`` for this bf16 ``weight`` [N, K] (no bias)."""
    if weight.dim() != 2:
        return "bf16_gemm: weight must be [N, K]"
    key = tuple(weight.shape)
    reason = _BF16.blocker(key)
    if reason is not None:
        return reason
    if not _BF16.package(key).supports(weight, bias):
        return "bf16_gemm: needs a contiguous bf16 weight and no bias"
    return None


def bf16_gemm(x, weight, out=None, *, shuffled: bool = False):
    """bf16 ``x @ W^T``; ``shuffled``: ``weight`` went through ``bf16_gemm_shuffle_weight``."""
    return _BF16.package(tuple(weight.shape)).bf16_gemm_n129280_k5120(x, weight, out, shuffled=shuffled)


def bf16_gemm_shuffle_weight(weight):
    return _BF16.package(tuple(weight.shape)).shuffle_weight(weight)


def bf16_gemm_shuffle_weight_(weight):
    """``bf16_gemm_shuffle_weight`` in place (no second copy of the weight)."""
    return _BF16.package(tuple(weight.shape)).shuffle_weight_(weight)


def warmup_bf16_gemm(device, n: int, k: int, shuffled: bool) -> None:
    _BF16.package((n, k)).warmup_bf16_gemm_n129280_k5120(device, shuffled)


def bf16_gemm_f32_bf16_shapes() -> frozenset:
    """(K, bf16 columns) keys."""
    return _BF16_SPLIT.shapes()


def bf16_gemm_f32_bf16_blocker(x, weight, na: int) -> str | None:
    """None when a package serves one bf16 GEMM over ``weight`` [na + nb, K] writing the first ``na`` columns in
    fp32 and the last nb in bf16."""
    if x.dim() != 2 or weight.dim() != 2:
        return "bf16_gemm_f32_bf16: x and weight must be 2-D"
    key = (x.shape[1], weight.shape[0] - na)
    reason = _BF16_SPLIT.blocker(key)
    if reason is not None:
        return reason
    if not _BF16_SPLIT.package(key).supports(x, weight, na):
        return f"bf16_gemm_f32_bf16: the kernel does not take these inputs (na {na}, rows {x.shape[0]}, layout)"
    return None


def bf16_gemm_f32_bf16(x, weight, out_a, out_b):
    """``out_a`` (fp32, may be None when na is 0) and ``out_b`` (bf16) columns of ``x @ W^T`` in one launch."""
    na = weight.shape[0] - out_b.shape[1]
    check(bf16_gemm_f32_bf16_blocker(x, weight, na))
    return _BF16_SPLIT.package((x.shape[1], out_b.shape[1])).gemm(x, weight, out_a, out_b)


def bf16_gemm_f32_bf16_shuffle_weight(weight_a, weight_b):
    """fp32-column weight ``weight_a`` [na, K] (or None) and bf16-column weight ``weight_b`` [nb, K] -> the layout
    ``bf16_gemm_f32_bf16`` reads (load time)."""
    return _BF16_SPLIT.package((weight_b.shape[1], weight_b.shape[0])).shuffle_weight(weight_a, weight_b)
