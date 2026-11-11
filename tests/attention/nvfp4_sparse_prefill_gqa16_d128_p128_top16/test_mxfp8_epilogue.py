# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Fused MXFP8 bytes at every finite bf16 group maximum, including subnormals."""

import pytest
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from amd_kernels.common import buffer_ops
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import reference
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.flydsl_utils import _store_out


def _compile_epilogue(pack):
    @flyc.kernel(name=f"bf16_mxfp8_d128_epilogue_{pack}", known_block_size=[64, 1, 1])
    def kernel(q: fx.Int64, out: fx.Int64, scales: fx.Int64, nt: fx.Int32):
        lane, tok = fx.thread_idx.x, fx.block_idx.x
        n, g = lane % 16, lane // 16
        qr = buffer_ops.create_buffer_resource_from_addr(q, num_records_bytes=fx.Int64(nt) * 4096)
        base = (tok * 16 + n) * 64 + g * 16
        words = [[fx.Int32(buffer_ops.buffer_load(qr, base + e * 4 + j, vec_width=1, dtype=fx.Int32))
                  for j in range_constexpr(4)] for e in range_constexpr(4)]
        _store_out(words, tok, fx.Int32(0), n, g, nt, fx.Int32(16), out, scales, pack, not pack)

    @flyc.jit
    def launch(q: fx.Int64, out: fx.Int64, scales: fx.Int64, nt: fx.Int32, stream: fx.Stream):
        kernel(q, out, scales, nt).launch(grid=(nt, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("pack", [False, True])
def test_every_finite_bf16_maximum(pack):
    mags = torch.arange(0x7F80, device="cuda", dtype=torch.int32)
    nt = (len(mags) + 63) // 64
    padded = torch.zeros(nt * 64, device="cuda", dtype=torch.int32)
    padded[:len(mags)] = mags
    cols = torch.arange(32, device="cuda", dtype=torch.int32)
    bits = padded[:, None] * ((cols * 173 + 19) % 997)[None, :] // 997
    bits[:, 0] = padded
    bits |= (cols[None, :] % 2) * 0x8000
    q = bits.to(torch.int16).view(torch.bfloat16).reshape(nt, 16, 128).contiguous()
    codes, scales = reference.mxfp8_quantize(q.reshape(nt, -1))
    c = torch.full((nt, 2048), 91, dtype=torch.uint8, device="cuda")
    s = torch.full(((nt + 31) // 32 * 32, 64), 91, dtype=torch.uint8, device="cuda")
    p = torch.full((nt, 16 * 132), 91, dtype=torch.uint8, device="cuda")
    exe = _compile_epilogue(pack)
    args = (q.data_ptr(), c.data_ptr(), p.data_ptr() if pack else s.data_ptr(), nt)
    exe(*args, torch.cuda.current_stream())
    torch.cuda.synchronize()
    if pack:
        assert torch.equal(p, reference.sp_pack_rows(q.reshape(nt, -1)))
        before = p.clone()
    else:
        assert torch.equal(c, codes)
        assert torch.equal(reference.aiter_unshuffle_scales(s)[:nt], scales)
        before = (c.clone(), s.clone())
    exe(*args, torch.cuda.current_stream())
    torch.cuda.synchronize()
    if pack:
        assert torch.equal(p, before)
    else:
        assert torch.equal(c, before[0]) and torch.equal(s, before[1])
