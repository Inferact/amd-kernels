# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``qkv.mxfp8_gemm_qkv_norm_rope_nvfp4_insert`` bit for bit the explicit calls (the one-launch entry, or
the GEMM + the split-K or bf16 insert) at the TP4 shape over 1-4096 rows, shuffled-code and row-major; and
``qkv_norm_rope_nvfp4_insert`` the insert with allocated outputs."""
import pytest
import torch
from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128 import (
    mxfp8_qkv_norm_rope_nvfp4_insert,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
    norm_rope_nvfp4_insert,
    splitk_reduce_norm_rope_nvfp4_insert,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import qkv as chain
from amd_kernels.gemm.mxfp8_common import gemm as g

from ..mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128.test_qkv_insert import (
    KW, N, K, make, require_gfx950, weights,
)

E4M3 = torch.float8_e4m3fn


def codes_for(m, seed):
    """The producer's codes for ``m`` rows: the GEMM row's tile, or row-major codes
    with shuffled scales where the tile is 0."""
    tile = g.prequant_code_tile_rows(m, N, K, torch.bfloat16)
    real = g.prequant_code_tile_rows
    g.prequant_code_tile_rows = lambda *a, **k: real(*a, **k) or 16  # make() quantizes in a tile
    try:
        sa, ra, ins = make(m, seed)
    finally:
        g.prequant_code_tile_rows = real
    if not tile:
        rc = ra[0].view(E4M3) if ra[0].dtype != E4M3 else ra[0]
        _, scales = g.mxfp8_e4m3_quantize_shuffled(torch.zeros(m, K, dtype=torch.bfloat16, device="cuda"), 16)
        scales.copy_(torch.randint(118, 136, scales.shape, dtype=torch.uint8, device="cuda"))
        sa = (rc.contiguous(), scales)
    return sa, ins


def explicit(sa, w, ws, m, ins, one_launch):
    """The calls by hand."""
    args = (*ins[:8], ins[8].clone(), ins[9].clone(), *ins[10:])
    tile = g.prequant_code_tile_rows(m, N, K, torch.bfloat16)
    if one_launch and 1 <= m <= 1536 and tile:
        q = torch.empty(m, 2048, dtype=torch.bfloat16, device="cuda")
        iq = torch.empty(m, 128, dtype=E4M3, device="cuda")
        mxfp8_qkv_norm_rope_nvfp4_insert(*sa, w, ws, m, *args, q, iq, **KW)
        return q, iq, args[8], args[9]
    if tile:
        out = g.mxfp8_gemm_shuffled(*sa, w, ws, torch.bfloat16, m, defer_reduce=True)
    else:
        out = g.mxfp8_gemm(*sa, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    if isinstance(out, g.SkPartials):
        rows = out.tmp.shape[1]
        q = torch.empty(rows, 2048, dtype=torch.bfloat16, device="cuda")
        iq = torch.empty(rows, 128, dtype=E4M3, device="cuda")
        splitk_reduce_norm_rope_nvfp4_insert(out.tmp, out.split_k, *args, q, iq, **KW)
    else:
        q = torch.empty(m, 2048, dtype=torch.bfloat16, device="cuda")
        iq = torch.empty(m, 128, dtype=E4M3, device="cuda")
        norm_rope_nvfp4_insert(out.reshape(m, N), *args, q, iq, **KW)
    return q, iq, args[8], args[9]


def new(sa, w, ws, m, ins, one_launch):
    args = (*ins[:8], ins[8].clone(), ins[9].clone(), *ins[10:])
    q, iq = chain.mxfp8_gemm_qkv_norm_rope_nvfp4_insert(*sa, w, ws, m, torch.bfloat16, *args, **KW,
                                                        one_launch=one_launch)
    return q, iq, args[8], args[9]


ROWS = [1, 8, 16, 17, 32, 64, 65, 128, 129, 192, 193, 256, 288, 289, 512, 1024, 1536, 1537, 2048, 4096]


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("one_launch", [True, False])
def test_chain_bitwise_explicit_calls(m, one_launch):
    require_gfx950()
    _, (w, ws), _, _ = weights()
    sa, ins = codes_for(m, 7 * m + 1)
    a = explicit(sa, w, ws, m, ins, one_launch)
    b = new(sa, w, ws, m, ins, one_launch)
    torch.cuda.synchronize()
    for x, y in zip(a, b):
        assert x.shape == y.shape and torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def test_bf16_qkv_entry():
    require_gfx950()
    m = 37
    _, ins = codes_for(m, 3)
    qkv = (torch.randn(m, N, device="cuda") * 2).to(torch.bfloat16)
    args = (*ins[:8], ins[8].clone(), ins[9].clone(), *ins[10:])
    q = torch.empty(m, 2048, dtype=torch.bfloat16, device="cuda")
    iq = torch.empty(m, 128, dtype=E4M3, device="cuda")
    norm_rope_nvfp4_insert(qkv, *args, q, iq, **KW)
    args2 = (*ins[:8], ins[8].clone(), ins[9].clone(), *ins[10:])
    q2, iq2 = chain.qkv_norm_rope_nvfp4_insert(qkv, *args2, **KW)
    torch.cuda.synchronize()
    assert torch.equal(q.view(torch.int16), q2.view(torch.int16)) and torch.equal(iq.view(torch.uint8),
                                                                                  iq2.view(torch.uint8))
    assert torch.equal(args[8], args2[8]) and torch.equal(args[9], args2[9])


def test_zero_rows():
    require_gfx950()
    _, (w, ws), _, _ = weights()
    sa, ins = codes_for(8, 5)
    q, iq = chain.mxfp8_gemm_qkv_norm_rope_nvfp4_insert(sa[0][:0], sa[1], w, ws, 0, torch.bfloat16, *ins, **KW)
    assert q.shape == (0, 2048) and iq.shape == (0, 128)
