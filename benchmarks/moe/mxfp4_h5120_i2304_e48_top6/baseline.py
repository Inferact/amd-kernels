# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""AITER's FlyDSL a8w4 ``fused_moe`` as the baseline: sort, 2 FlyDSL GEMMs (fused MXFP8 requant in stage 1, atomic or
reduce stage 2), the weights in AITER's preshuffled layout (the bytes of ``ExpertWeights``). AITER ships no tuned row
for this shape at decode counts: ``BASELINE`` holds the kernels the sweep (``benchmark_baseline_sweep.py``) found
fastest per padded token count."""
import torch

H, I, E, TOPK = 5120, 2304, 48, 6
# routing -> tokens -> (sort block_m, stage 1 kernel, stage 2 kernel); filled by benchmark_baseline_sweep.py
BASELINE: dict[str, dict[int, tuple[int, str, str]]] = {
    "uniform": {
        16: (
            16, 'flydsl_moe1_afp8_wfp4_bf16_t16x128x256_w4_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t16x128x128_atomic_bnt2'),
        32: (
            32, 'flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t32x256x256_atomic_bnt2'),
        64: (
            32, 'flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t32x256x256_atomic'),
        128: (
            32, 'flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t32x256x256_atomic_bnt2'),
        192: (
            32, 'flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic_bnt2'),
        384: (
            64, 'flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w2_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t64x256x256_atomic_bnt2'),
        768: (
            128, 'flydsl_moe1_afp8_wfp4_bf16_t128x256x256_w4_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t128x128x256_atomic_bnt2'),
    },
    "skew": {
        16: (
            16, 'flydsl_moe1_afp8_wfp4_bf16_t16x128x256_w3_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t16x256x128_atomic_bnt2'),
        32: (
            32, 'flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw2_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic'),
        64: (
            32, 'flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic'),
        128: (
            32, 'flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic_bnt2'),
        192: (
            64, 'flydsl_moe1_afp8_wfp4_bf16_t64x256x256_w4_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic_bnt2'),
        384: (
            64, 'flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_bnt0_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic'),
        768: (
            64, 'flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w4_bnt0_gui_fp8',
            'flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic'),
    },
}
DEFAULT = (32, "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_bnt0_gui_fp8",
           "flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic")


def install_row(n_tokens: int, block_m: int, kernel1: str, kernel2: str) -> None:
    import aiter.fused_moe as fm

    primary, _ = fm.cfg_2stages
    key = ("gfx950", 256, fm.get_padded_M(n_tokens), H, I, E, TOPK, "ActivationType.Silu", "torch.bfloat16",
           "torch.float8_e4m3fn", "torch.float4_e2m1fn_x2", "QuantType.per_1x32", 1, 0)
    row = dict(next(iter(primary.values())))
    row.update(block_m=block_m, ksplit=0, kernelName1=kernel1, kernelName2=kernel2, run_1stage=0)
    primary[key] = row
    fm.get_2stage_cfgs.cache_clear()


def aiter_experts(w, n_tokens: int, dist: str = "uniform", config=None):
    """``call(x_codes, x_scales, topk_weights, topk_ids)`` -> bf16 ``[n, 5120]`` through ``fused_moe``."""
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe
    import aiter.fused_moe as fm

    n = w.w13.shape[0]
    w1 = w.w13.view(n, 2 * I, H // 2).view(dtypes.fp4x2)
    w2 = w.w2.view(n, H, I // 2).view(dtypes.fp4x2)
    w1.is_shuffled = w2.is_shuffled = True
    w1.gate_up_interleave, w2.gate_up_interleave = True, False
    s1 = w.w13_scale.view(-1, H // 32)
    s2 = w.w2_scale.view(-1, I // 32)
    if fm.cfg_2stages is None:
        try:  # loads the tuned tables
            fm.get_2stage_cfgs(1, H, I, E, TOPK, torch.bfloat16, dtypes.fp8, dtypes.fp4x2, QuantType.per_1x32, True,
                               ActivationType.Silu, False, 0, 0, True)
        except Exception:
            pass
    bm, k1, k2 = config or BASELINE[dist].get(n_tokens) or DEFAULT
    install_row(n_tokens, bm, k1, k2)

    def call(x_codes, x_scales, topk_weights, topk_ids):
        return fused_moe(x_codes, w1, w2, topk_weights, topk_ids, w1_scale=s1.view(dtypes.fp8_e8m0),
                         w2_scale=s2.view(dtypes.fp8_e8m0), a1_scale=x_scales.view(dtypes.fp8_e8m0),
                         quant_type=QuantType.per_1x32, activation=ActivationType.Silu, swiglu_limit=10.0,
                         quant_dtype_a=dtypes.fp8, dtype=torch.bfloat16)

    return call
