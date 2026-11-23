# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 GEMM with the mHC post-mix epilogue (optionally plus the routed MoE add): routes a call by (N, K, routed add)
to the shape package (``_ROUTES``)."""

from amd_kernels.routes import Prepared, Routes, Rows, check, first, previous_form, renamed

_ROUTES = Routes(
    "mxfp8_postmix_gemm",
    {
        (5120, 8192, False): "amd_kernels.gemm.mxfp8_postmix_hc4_n5120_k8192",
        (5120, 2304, True): "amd_kernels.gemm.mxfp8_postmix_add_hc4_n5120_k2304",
    },
    ("n", "k", "routed"),
)
# rows at which the fused epilogue beat the served GEMM + the next mHC seam on MI355X (vLLM DeepSeek-V4.1 DEP8 routes:
# wo_b 1-64, shared-expert down + routed add 1-192)
_ROWS = {(5120, 8192, False): Rows((1, 64)), (5120, 2304, True): Rows((1, 192))}


def mxfp8_postmix_shapes() -> frozenset:
    """(N, K, routed add) keys."""
    return _ROUTES.shapes()


def _supported(m: int, key) -> str | None:
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    limit = _ROUTES.package(key).MAX_ROWS
    return None if 1 <= m <= limit else f"mxfp8_postmix_gemm: {m} rows outside 1..{limit}"


def _faster(m: int, key) -> str | None:
    return first(_supported(m, key), _ROWS[key].blocker("mxfp8_postmix_gemm", m) if key in _ROWS else None)


def _integer_form(m: int, n: int, k: int, routed: bool) -> str | None:
    return _supported(m, (n, k, bool(routed)))


def _key(weight, routed):
    return (weight.shape[0], weight.shape[1], routed is not None)


def mxfp8_postmix_gemm_blocker(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, routed=None,
                               out=None) -> str | None:  # fmt: skip
    """None when a package serves ``mxfp8_postmix_gemm`` for this call and beats the served path at its row count."""
    return _faster(codes.shape[0], _key(weight, routed))


def mxfp8_postmix_gemm(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, routed=None, out=None):
    """The new residual streams ``post_mix * (codes @ W^T [+ routed]) + comb_mix @ residual``; as the shape
    package's ``postmix_gemm``."""
    key = _key(weight, routed)
    check(_supported(codes.shape[0], key))
    pkg = _ROUTES.package(key)
    if routed is None:
        return pkg.postmix_gemm(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, out)
    return pkg.postmix_gemm(codes, scales, weight, weight_scale, residual, routed, post_mix, comb_mix, out)


def mxfp8_postmix_gates_gemm(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, pre_mix, fn_packed,
                             routed=None, *, aux_mean: bool = False):  # fmt: skip
    """The post-mix GEMM that also writes the next seam's gate tile records (``norm.mhc_gate_finish`` consumes them);
    as the shape package's ``postmix_gates_gemm``."""
    key = _key(weight, routed)
    check(_supported(codes.shape[0], key))
    pkg = _ROUTES.package(key)
    if routed is None:
        return pkg.postmix_gates_gemm(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, pre_mix,
                                      fn_packed, aux_mean=aux_mean)
    return pkg.postmix_gates_gemm(codes, scales, weight, weight_scale, residual, routed, post_mix, comb_mix, pre_mix,
                                  fn_packed, aux_mean=aux_mean)


def mxfp8_postmix_gates_gemm_blocker(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, pre_mix,
                                     fn_packed, routed=None, *, aux_mean: bool = False) -> str | None:  # fmt: skip
    """``mxfp8_postmix_gemm_blocker`` for ``mxfp8_postmix_gates_gemm``."""
    return _faster(codes.shape[0], _key(weight, routed))


def mxfp8_postmix_preshuffle_weight(weight, weight_scale, routed: bool):
    """E4M3 [N, K] + e8m0 [N, K / 32] -> the layouts the post-mix GEMMs read (load time)."""
    return _ROUTES.package((weight.shape[0], weight.shape[1], routed)).preshuffle_weight(weight, weight_scale)


def mxfp8_postmix_pack_gate_weight(fn, n: int, k: int, routed: bool):
    """The fp32 gate weight in the gates GEMM's packed layout (load time)."""
    return _ROUTES.package((n, k, routed)).pack_gate_weight(fn)


def warmup_mxfp8_postmix_gemm(weight, weight_scale, routed: bool) -> None:
    _ROUTES.package((weight.shape[0], weight.shape[1], routed)).warmup_postmix_gemm(weight, weight_scale)


class Mxfp8PostmixGemm(Prepared):
    """The post-mix GEMM bound to one weight (and, with ``fn``, its gate weight): ``h(codes, scales, residual,
    post_mix, comb_mix, routed=None, out=None)``; ``h.gates(...)`` the gates variant."""

    op = "mxfp8_postmix_gemm"

    def __init__(self, weight, weight_scale, routed: bool, fn_packed):
        self.weight, self.weight_scale, self.routed, self.fn_packed = weight, weight_scale, routed, fn_packed
        self.key = (weight.shape[0], weight.shape[1], routed)

    def blocker(self, codes, scales, residual=None, post_mix=None, comb_mix=None, routed=None, out=None):
        return _faster(codes.shape[0], self.key)

    def __call__(self, codes, scales, residual, post_mix, comb_mix, routed=None, out=None):
        return mxfp8_postmix_gemm(codes, scales, self.weight, self.weight_scale, residual, post_mix, comb_mix, routed,
                                  out)

    def gates(self, codes, scales, residual, post_mix, comb_mix, pre_mix, routed=None, *, aux_mean: bool = False):
        return mxfp8_postmix_gates_gemm(codes, scales, self.weight, self.weight_scale, residual, post_mix, comb_mix,
                                        pre_mix, self.fn_packed, routed, aux_mean=aux_mean)


def prepare_mxfp8_postmix_gemm(weight, weight_scale, routed: bool, fn=None, *,
                               weight_preshuffled=None) -> Mxfp8PostmixGemm | str:  # fmt: skip
    """The bound post-mix GEMM, compiled, or the reason no package serves it.

    Args:
        weight: e4m3 [N, K] in the checkpoint's row-major layout; weight_scale: e8m0 per 1 x 32 or 32 x 32 block.
        fn: the next seam's fp32 gate weight [mixes, streams * hidden], for ``gates``.
        weight_preshuffled: the weight already in AITER's (16, 16) preshuffle (used as is, no copy).
    """
    from .prepared import preshuffled

    key = (weight.shape[0], weight.shape[1], bool(routed))
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    w, s = preshuffled(weight, weight_scale, weight_preshuffled)
    warmup_mxfp8_postmix_gemm(w, s, bool(routed))
    fn_packed = None if fn is None else mxfp8_postmix_pack_gate_weight(fn.contiguous(), *key)
    return Mxfp8PostmixGemm(w, s, bool(routed), fn_packed)


mxfp8_postmix_blocker = renamed(previous_form(_integer_form)(mxfp8_postmix_gemm_blocker), "mxfp8_postmix_blocker")
