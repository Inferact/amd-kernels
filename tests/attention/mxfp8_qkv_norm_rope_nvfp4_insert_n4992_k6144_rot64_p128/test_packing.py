# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``packing._v_expanded_words``: every input basis bit plus random four/eight-row packets equal the row-by-row
nibble insert."""

import functools

import pytest
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128.packing import (
    _v_expand,
    _v_expanded_words,
)


@functools.cache
def packing_launcher(rows):
    name = f"test_n4992_v_nibble_transpose_rows{rows}"

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def pack(x: fx.Tensor, out: fx.Tensor):
        idx = fx.block_idx.x * 64 + fx.thread_idx.x
        values = [fx.Int32(x[idx * rows + r]) for r in range_constexpr(rows)]
        words = _v_expanded_words([_v_expand(v) for v in values])
        for i in range_constexpr(4):
            out[idx * 4 + i] = words[i]

    @flyc.jit
    def launch_n4992_v_nibble_transpose_test(x: fx.Tensor, out: fx.Tensor, blocks: fx.Int32, stream: fx.Stream):
        pack(x, out).launch(grid=(blocks, 1, 1), block=(64, 1, 1), stream=stream)

    return launch_n4992_v_nibble_transpose_test


def expected(x, rows):
    """Row r's nibble of dimension d goes to bits 4r of word d (the run-by-row insert)."""
    out = torch.zeros(x.shape[0], 4, dtype=torch.int64)
    for r in range(rows):
        for d in range(4):
            out[:, d] |= ((x[:, r] >> (4 * d)) & 15) << (4 * r)
    return out.to(torch.int32)


@pytest.mark.parametrize("rows", [4, 8])
def test_expanded_words(rows):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    basis = torch.tensor([[1 << b if r == rr else 0 for r in range(rows)] for rr in range(rows) for b in range(16)])
    rand = torch.randint(0, 1 << 16, (4096 - basis.shape[0], rows))
    x = torch.cat([basis, rand]).to(torch.int32)
    out = torch.empty(x.shape[0] * 4, dtype=torch.int32, device="cuda")
    packing_launcher(rows)(x.flatten().cuda(), out, x.shape[0] // 64, torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(out.view(-1, 4).cpu(), expected(x.to(torch.int64), rows))
