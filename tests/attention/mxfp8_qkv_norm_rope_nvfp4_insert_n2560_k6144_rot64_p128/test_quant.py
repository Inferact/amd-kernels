# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Check exact reciprocal staging on all 65,280 finite bf16 encodings and several global scales."""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128.kernel import _e4m3_value, _quant4
from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128.quant import _quant4_table


@fx.struct
class QuantShared:
    reciprocal: fx.Array[fx.Int32, 260, 16]


@flyc.kernel(name="test_exact_v_reciprocal_all_finite_bf16", known_block_size=[256, 1, 1])
def quantize_finite_bf16(x: fx.Tensor, scales: fx.Tensor, out: fx.Tensor):
    tx, bx, by = fx.thread_idx.x, fx.block_idx.x, fx.block_idx.y
    smem = fx.SharedAllocator().allocate(QuantShared).peek()
    gs = fx.Float32(scales[by])
    sv = _e4m3_value(tx)
    quant = (sv != 0.0).select(fx.Float32(1.0) / (gs * sv), fx.Float32(0.0))
    smem.reciprocal.ptr[tx] = quant.bitcast(fx.Int32)
    if tx == 0:
        smem.reciprocal.ptr[256] = (fx.Float32(1.0) / gs).bitcast(fx.Int32)
    fx.gpu.barrier()
    idx = bx * 256 + tx
    # A permutation mixes signs/exponents within each lane and its 16-value scale group.
    xs = [fx.Float32(x[(idx + i * 16381) % 65536]) for i in range_constexpr(4)]
    codes0, scale0 = _quant4(xs, gs, fx.Float32(1.0) / gs)
    codes1, scale1 = _quant4_table(xs, smem.reciprocal.ptr)
    off = (by * 65536 + idx) * 2
    out[off] = codes0 | (scale0 << 16)
    out[off + 1] = codes1 | (scale1 << 16)


@flyc.jit
def launch_exact_v_reciprocal_test(x: fx.Tensor, scales: fx.Tensor, out: fx.Tensor, n: fx.Int32,
                                  stream: fx.Stream):
    quantize_finite_bf16(x, scales, out).launch(grid=(256, n, 1), block=(256, 1, 1), stream=stream)


def test_all_finite_bf16_encodings():
    from .test_qkv_insert import require_gfx950

    require_gfx950()
    encodings = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    finite = torch.isfinite(encodings)
    assert int(finite.sum()) == 65280
    values = torch.where(finite, encodings.float(), 0).contiguous()
    scales = torch.tensor([
        0.37, 0.61, 1.0, -0.61, 0.0, -0.0, 2.0**-100, 2.0**-20, 2.0**-5, 2.0**5,
        2.0**20, 2.0**100, 1.0000001192092896, 0.9999999403953552,
    ], device="cuda")
    out = torch.empty(scales.numel() * 65536 * 2, dtype=torch.int32, device="cuda")
    compiled = flyc.compile(launch_exact_v_reciprocal_test, values, scales, out, scales.numel(),
                            torch.cuda.current_stream())
    compiled(values, scales, out, scales.numel(), torch.cuda.current_stream())
    torch.cuda.synchronize()
    pairs = out.view(scales.numel(), 65536, 2)
    counts = (pairs[:, :, 0] != pairs[:, :, 1]).sum(dim=1).cpu().tolist()
    assert not any(counts), dict(zip(scales.cpu().tolist(), counts))
