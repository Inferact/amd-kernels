# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Exact integer transpose: every input basis bit plus random four/eight-row packets."""

import functools
import pytest
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128.packing import _v_words, _v_expand, _v_expanded_words


@functools.cache
def packing_launcher(rows, expanded):
    name = f"test_v_nibble_transpose_rows{rows}_expanded{expanded}"

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def pack(x: fx.Tensor, out: fx.Tensor):
        idx = fx.block_idx.x * 64 + fx.thread_idx.x
        values = [fx.Int32(x[idx * rows + r]) for r in range_constexpr(rows)]
        words = _v_expanded_words([_v_expand(v) for v in values]) if expanded else _v_words(values)
        for i in range_constexpr(4):
            out[idx * 4 + i] = words[i]

    @flyc.jit
    def launch_v_nibble_transpose_test(x: fx.Tensor, out: fx.Tensor, blocks: fx.Int32, stream: fx.Stream):
        pack(x, out).launch(grid=(blocks, 1, 1), block=(64, 1, 1), stream=stream)

    return launch_v_nibble_transpose_test


@pytest.mark.parametrize("rows", [4, 8])
@pytest.mark.parametrize("expanded", [False, True])
def test_each_bit_and_random_packets(rows, expanded):
    from .test_qkv_insert import require_gfx950

    require_gfx950()
    bits = torch.arange(rows * 16, device="cuda", dtype=torch.int64)
    basis = torch.zeros((rows * 16, rows), device="cuda", dtype=torch.int64)
    basis[bits, bits // 16] = 1 << (bits % 16)
    generator = torch.Generator(device="cuda").manual_seed(rows)
    random = torch.randint(0, 65536, (1024, rows), device="cuda", generator=generator)
    packets = torch.cat((basis, random)).to(torch.int32)
    expected = torch.zeros((packets.shape[0], 4), dtype=torch.int64, device="cuda")
    for r in range(rows):
        for i in range(4):
            expected[:, i] |= ((packets[:, r].to(torch.int64) >> (4 * i)) & 15) << (4 * r)
    got = torch.empty_like(expected, dtype=torch.int32)
    launch = packing_launcher(rows, expanded)
    compiled = flyc.compile(launch, packets.flatten(), got.flatten(), packets.shape[0] // 64,
                            torch.cuda.current_stream())
    compiled(packets.flatten(), got.flatten(), packets.shape[0] // 64, torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(got, expected.to(torch.int32))
