# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 pack of the attention output vs reference.pack_reference, bit for bit: row counts on and off the 32-row
tiles (the served 4096-32768-token range, tails), an all-zero row, tiny and large blocks, a block below 2^-119 (the
documented conversion rule), into a pool-slot view (rows beyond M untouched), eager vs graph replay; and the codes
equal the library quantizer's on the same rows."""

import pytest
import torch
from amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4 import mxfp8_pack_q2048
from amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4.reference import pack_reference

D, ROW_BYTES = 2048, 2112


def _require_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip(f"gfx950 only, found {arch}")


def _rows(m: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, D, device="cuda", generator=g)
    x = x * torch.rand(m, 1, device="cuda", generator=g) * 40
    if m > 5:
        x[1] = 0  # an all-zero row: the scale floor
        x[2, :32] = 2.0**-36  # a tiny block
        x[3, :64] = 3e4  # a large block
        x[4, 64:96] = -(2.0**-124)  # below 2^-119: byte 0, converted with 2^-126
        x[5, 96:128] = 448.0 * 2.0**3  # exactly 448 * 2^k: the scale boundary
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("m", [1, 6, 31, 32, 33, 1000, 1024, 4097, 8192, 8885, 32768])
def test_pack_reference(m):
    _require_gfx950()
    x = _rows(m, m)
    got = mxfp8_pack_q2048(x)
    torch.cuda.synchronize()
    ref = pack_reference(x)
    bad = (got != ref).any(dim=1).nonzero().flatten()[:8].tolist()
    assert torch.equal(got, ref), f"rows {bad} differ"


def test_pack_codes_equal_library_quantizer():
    """Codes as gemm/mxfp8_common's quantizer gives them (scales: that one shuffles them)."""
    _require_gfx950()
    from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_e4m3_quantize_aiter, mxfp8_e4m3_quantize_flydsl

    x = _rows(4096, 7)
    packed = mxfp8_pack_q2048(x)
    for quant in (mxfp8_e4m3_quantize_aiter, mxfp8_e4m3_quantize_flydsl):
        q, _ = quant(x)
        same = packed[:, :D] == q.view(torch.uint8)
        same[4, 64:96] = True  # the byte-0 block: the Triton quantizer multiplies by 2^127 there
        assert bool(same.all()), quant.__name__


def test_pack_into_slot_view():
    _require_gfx950()
    m, total = 1000, 4128
    x = _rows(m, 3)
    slot = torch.full((total + 7, ROW_BYTES), 0xA5, dtype=torch.uint8, device="cuda")
    view = slot[5 : 5 + total]
    mxfp8_pack_q2048(x, out=view[:m])
    torch.cuda.synchronize()
    assert torch.equal(view[:m], pack_reference(x))
    assert bool((slot[:5] == 0xA5).all()) and bool((view[m:] == 0xA5).all()) and bool((slot[5 + total :] == 0xA5).all())


def test_pack_graph_replay():
    _require_gfx950()
    x = _rows(4097, 11)
    out = torch.empty((4097, ROW_BYTES), dtype=torch.uint8, device="cuda")
    mxfp8_pack_q2048(x, out=out)  # loads the kernel before capture
    torch.cuda.synchronize()
    eager = out.clone()
    out.fill_(0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mxfp8_pack_q2048(x, out=out)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    x.copy_(_rows(4097, 12))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, pack_reference(x))
