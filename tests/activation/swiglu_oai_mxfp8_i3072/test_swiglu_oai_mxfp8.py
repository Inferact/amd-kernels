# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``swiglu_oai_mxfp8`` against its contract: the aten chain on the same bf16 gate_up, then the
MXFP8 quantizer (``amd_kernels.gemm.mxfp8_common``'s row-major and (16, 16) quantizers), bit for bit; the codes also
against ``reference.quantize_mxfp8``. Rows: the draft's 7R and 8R (R = 1, 10, 16, 32, 64, 128, 256) and odd
counts; rows scaled over +-5 binades, values at and beyond the clamp limit; graph replay; determinism."""

import pytest
import torch

from amd_kernels.activation.swiglu_oai_mxfp8_i3072 import swiglu_oai_mxfp8
from amd_kernels.activation.swiglu_oai_mxfp8_i3072.reference import quantize_mxfp8, swiglu_bf16_chain, swiglu_fp32

I = 3072
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
ROWS = [1, 7, 8, 13, 16, 17, 63, 65, 127, 129, 70, 80, 112, 128, 224, 256, 448, 512, 896, 1024, 1792, 2048]
TILES = [16, 32, 64, 96, 128]


def _gate_up(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, 2 * I), generator=g, device="cuda") * 3.0
    x *= torch.exp2(torch.randint(-5, 6, (m, 1), generator=g, device="cuda").float())
    x[:, :8] = torch.tensor([7.0, 7.5, -7.0, -9.0, 0.0, -0.0, 1e-30, 30.0], device="cuda")
    return x.bfloat16()


def _today(x, tile, form="bf16_chain"):
    from amd_kernels.gemm.mxfp8_common import gemm as G

    act = (swiglu_bf16_chain if form == "bf16_chain" else swiglu_fp32)(x, ALPHA, BETA, LIMIT)
    if tile:
        return G.mxfp8_e4m3_quantize_shuffled(act, tile)
    return G.mxfp8_e4m3_quantize_aiter(act)


@pytest.mark.parametrize("m", ROWS)
def test_rowmajor_bitwise(m):
    x = _gate_up(m, m)
    got = swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, 0)
    ref = _today(x, 0)
    assert torch.equal(got[0][:m].view(torch.uint8), ref[0][:m].view(torch.uint8)), "codes"
    _scales_eq(got[1], ref[1].view(torch.uint8), m)
    q_ref, s_ref = quantize_mxfp8(swiglu_bf16_chain(x, ALPHA, BETA, LIMIT))
    assert torch.equal(got[0][:m].view(torch.uint8), q_ref.view(torch.uint8))


@pytest.mark.parametrize("m", ROWS)
def test_fp32_form_bitwise(m):
    """The target's form: swiglu_oai_split's fp32 math + one rounding, then the quantizer."""
    x = _gate_up(m, 11 * m)
    for tile in (0, 16, 64):
        got = swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, tile, "fp32")
        ref = _today(x, tile, "fp32")
        _codes_eq(got[0], ref[0], m, tile)
        _scales_eq(got[1], ref[1].view(torch.uint8), m)


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("tile", TILES)
def test_tiled_bitwise(m, tile):
    x = _gate_up(m, 7 * m + tile)
    got = swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, tile)
    ref = _today(x, tile)
    _codes_eq(got[0], ref[0], m, tile)
    _scales_eq(got[1], ref[1].view(torch.uint8), m)


def _codes_eq(got, ref, m, tile):
    gq, rq = got.view(torch.uint8), ref.view(torch.uint8)
    if not tile:
        assert torch.equal(gq[:m], rq[:m])
        return
    full = m // 16 * 16
    assert torch.equal(gq[:full], rq[:full]), "codes (full 16-row groups)"
    if m % 16:
        blk = gq[full:full + 16].reshape(I // 64, 4, 16, 16)[:, :, : m % 16]
        rblk = rq[full:full + 16].reshape(I // 64, 4, 16, 16)[:, :, : m % 16]
        assert torch.equal(blk, rblk), "codes (partial group)"


def _scales_eq(gs, rs, m):
    """Shuffled scales: compare the bytes that belong to real rows (unshuffle by permuting an index table)."""
    from aiter.ops.shuffle import shuffle_scale_a16w4

    rows = gs.shape[0]
    idx = torch.arange(rows, device=gs.device, dtype=torch.int32)[:, None].expand(rows, I // 32).contiguous()
    row_of = shuffle_scale_a16w4(idx, 1, False).reshape(-1)
    real = row_of < m
    assert torch.equal(gs.reshape(-1)[real], rs[:rows].reshape(-1)[real]), "scales"


def test_rowmajor_scales_partial():
    for m in (1, 7, 13, 70):
        x = _gate_up(m, 99 + m)
        got = swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, 0)
        ref = _today(x, 0)
        _scales_eq(got[1], ref[1].view(torch.uint8), m)


@pytest.mark.parametrize("m", (7,17,70,112,224))
def test_graph_and_determinism(m):
    tile = 32
    xs = [_gate_up(m, 500 + i) for i in range(4)]
    eager = [swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, tile) for x in xs]
    q = torch.empty_like(eager[0][0])
    s = torch.empty_like(eager[0][1])
    graph = torch.cuda.CUDAGraph()
    src = xs[0].clone()
    with torch.cuda.graph(graph):
        swiglu_oai_mxfp8(src, ALPHA, BETA, LIMIT, tile, q_out=q, scale_out=s)
    for x, (eq, es) in zip(xs, eager):
        src.copy_(x)
        q.view(torch.uint8).fill_(0x7f)
        s.fill_(0xff)
        graph.replay()
        torch.cuda.synchronize()
        _codes_eq(q, eq, m, tile)
        _scales_eq(s, es, m)
    again = swiglu_oai_mxfp8(xs[1], ALPHA, BETA, LIMIT, tile)
    _codes_eq(again[0], eager[1][0], m, tile)


def test_no_compile_after_warmup():
    """README rule 4: after the warm-up, every row count of the draft batch (7R, 8R) and both code layouts
    run on the compiled variants (the launcher's cached CompiledFunction; no new compile)."""
    from amd_kernels.activation.swiglu_oai_mxfp8_i3072 import host, warmup_swiglu_oai_mxfp8

    host._get_kernel.cache_clear()
    warmup_swiglu_oai_mxfp8()
    before = host._get_kernel.cache_info().currsize
    warmup_swiglu_oai_mxfp8(forms=("bf16_chain", "fp32"))
    keys = [(f, t, p) for f in ("bf16_chain", "fp32") for t in (False, True) for p in (False, True)]
    compiled = {k: getattr(host._get_kernel(*k), "_cf", None) for k in keys}
    assert all(v is not None for v in compiled.values())
    for m in ROWS:
        x = _gate_up(m, 3 * m)
        for tile in (0, 16, 32, 64, 96, 128):
            for form in ("bf16_chain", "fp32"):
                swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, tile, form)
    assert host._get_kernel.cache_info().currsize == before
    assert all(getattr(host._get_kernel(*k), "_cf", None) is v for k, v in compiled.items())


@pytest.mark.parametrize('form', ('bf16_chain', 'fp32'))
@pytest.mark.parametrize('tile', (0, 64))
def test_partitioned_next_to_mfma(form, tile):
    from tests.mfma_spinner import mfma_spinner
    m = 70
    x = _gate_up(m, 987)
    ref = _today(x, tile, form)
    spinner = mfma_spinner(256)
    side = torch.cuda.Stream(priority=-1)
    for _ in range(3):
        spinner(100_000, side)
        got = swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, tile, form)
        torch.cuda.synchronize()
        _codes_eq(got[0], ref[0], m, tile)
        _scales_eq(got[1], ref[1].view(torch.uint8), m)


def test_zero_rows():
    x = torch.empty((0, 6144), device='cuda', dtype=torch.bfloat16)
    for tile in (0, 16):
        q, scales = swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, tile)
        assert q.numel() == scales.numel() == 0
