# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The attention between its producer and its consumer, with no quantize or cast launch in between: the query and the
sliding-window cache come from ``rope_fp8_kv_insert`` (e4m3 query, e4m3 rows inserted by slot, the query scale
``1 / q_scale_inv`` and the KV scale given as that kernel takes them, int64 positions), and the output goes into the
wo_a grouped GEMM as it is (e4m3 codes [rows, 8 * 4096], row-major e8m0 scales [rows, 8 * 128])."""

import torch
from aiter.ops.shuffle import shuffle_weight
from amd_kernels.attention.rope_fp8_kv_insert_h64_d512_rot64 import rope_fp8_kv_insert
from amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64 import reference as R
from amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64 import sparse_mla_decode
from amd_kernels.gemm import mxfp8_grouped_g8_n1024_k4096 as wo_a

from .cases import D, H, SOFTMAX_SCALE, e4m3, outputs, require_gfx950

N_POS, ROWS, WINDOW, TOPK = 300, 6, 128, 512
KV_SCALE, Q_SCALE_INV = 0.5, 2.0


def weight(gen):
    """A Gaussian wo_a weight: row-major codes, 32 x 32 exponents."""
    w = torch.randn(wo_a.reference.G, wo_a.reference.N // 32, 32, wo_a.reference.K // 32, 32, generator=gen) * 0.05
    exponent = (w.abs().amax(dim=(2, 4)).clamp_min(1e-30) / 448.0).log2().ceil().add(127).clamp(0, 254)
    codes = (w * torch.exp2(127 - exponent)[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return codes.view(-1, wo_a.reference.K).cuda(), exponent.to(torch.uint8).cuda()


@torch.inference_mode()
def test_producer_attention_consumer():
    require_gfx950()
    gen = torch.Generator(device="cuda").manual_seed(11)
    dev = "cuda"
    q = torch.randn((N_POS, H, D), generator=gen, device=dev).bfloat16()
    kv = torch.randn((N_POS, D), generator=gen, device=dev).bfloat16()
    cos_sin = torch.cat([torch.rand((N_POS, 32), generator=gen, device=dev).cos(),
                         torch.rand((N_POS, 32), generator=gen, device=dev).sin()], dim=1).float()  # fmt: skip
    pages, block = 4, 128
    cache = torch.zeros((pages, block, D), dtype=torch.float8_e4m3fn, device=dev)
    q_fp8 = torch.empty((N_POS, H, D), dtype=torch.float8_e4m3fn, device=dev)
    pos = torch.arange(N_POS, dtype=torch.int64, device=dev)
    kv_scale = torch.full((), KV_SCALE, device=dev)
    q_scale_inv = torch.full((), Q_SCALE_INV, device=dev)
    rope_fp8_kv_insert(q, kv, q_fp8, cache, pos, pos, cos_sin, kv_scale, q_scale_inv)  # slot = position
    # the last ROWS positions attend the WINDOW slots up to their own and TOPK compressed rows
    rows = torch.arange(N_POS - ROWS, N_POS, device=dev)
    main = torch.stack([torch.arange(int(p) - WINDOW + 1, int(p) + 1) for p in rows]).int().to(dev)
    comp = e4m3((1 << 12, D), 1.0, gen).view(8, 512, D).view(torch.float8_e4m3fn)
    extra = torch.stack([torch.randperm(1 << 12, generator=torch.Generator().manual_seed(int(p)))[:TOPK]
                         for p in rows]).int().to(dev)  # fmt: skip
    case = dict(
        q=q_fp8[N_POS - ROWS :].contiguous(), main_cache=cache, main_indptr=torch.arange(ROWS + 1, dtype=torch.int32, device=dev) * WINDOW,
        main_indices=main.flatten(), extra_cache=comp, extra_indptr=torch.arange(ROWS + 1, dtype=torch.int32, device=dev) * TOPK,
        extra_indices=extra.flatten(), attn_sink=torch.randn(H, generator=gen, device=dev), softmax_scale=SOFTMAX_SCALE,
        q_scale=torch.full((1,), 1.0 / Q_SCALE_INV, device=dev), kv_scale=kv_scale.reshape(1),
        positions=pos[N_POS - ROWS :].contiguous(), cos_sin=cos_sin,
    )  # fmt: skip
    codes, scales = outputs(ROWS)
    sparse_mla_decode(*case.values(), codes, scales)
    att = R.attention_reference(case["q"], 1.0 / Q_SCALE_INV, case["main_cache"], case["main_indptr"], case["main_indices"],
                                case["extra_cache"], case["extra_indptr"], case["extra_indices"], KV_SCALE, case["attn_sink"],
                                SOFTMAX_SCALE)  # fmt: skip
    ref_codes, ref_scales = R.epilogue_reference(att, case["positions"], cos_sin)
    cos = torch.nn.functional.cosine_similarity(R.dequant(codes, scales).double(),
                                                R.inverse_rope(att, case["positions"], cos_sin).flatten(1).double(), dim=1)
    assert cos.min() >= 0.9995, cos  # the MXFP8 output's own rounding is ~0.9996
    # the consumer takes the buffers as they are
    wcodes, wexp = weight(torch.Generator().manual_seed(3))
    shuffled = shuffle_weight(wcodes, (16, 16)).view_as(wcodes)
    got = wo_a.grouped_gemm(codes, scales, shuffled, wexp)
    again = wo_a.grouped_gemm(ref_codes.view(torch.float8_e4m3fn), ref_scales, shuffled, wexp)
    ref = wo_a.reference.grouped_gemm_reference(codes, scales, wcodes, wexp)
    assert torch.isfinite(got.float()).all() and got.shape == (ROWS, 8 * 1024)
    assert wo_a.reference.compare(got, ref)["cos"] >= wo_a.reference.COS_MIN  # the GEMM on these buffers
    assert wo_a.reference.compare(got, again)["cos"] >= 0.999  # and on the reference's quantization of the output
