# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``mxfp8_bmm_mxfp8_out`` against its contract: ``reference.quantize_mxfp8`` of AITER's FlyDSL batched GEMM at
the same tile, bit for bit (codes and scales), and of AITER's tuned dispatch (FlyDSL or OPUS) on rows without
fp32-subnormal outputs; the fp32 reference by dequantized cosine. Rows: CUDA graph batch sizes and their neighbours
up to a 16k prefill, every tile config at a partial last tile, 8 / 4 / 2 / 1 groups (attention TP 1 / 2 / 4 / 8).
Rows scaled over a wide E8M0 range with zero, subnormal and large rows; graph replay; determinism."""

import pytest
import torch

from amd_kernels.gemm.mxfp8_bmm_mxfp8_out_n1024_k4096 import host, mxfp8_bmm_mxfp8_out
from amd_kernels.gemm.mxfp8_bmm_mxfp8_out_n1024_k4096.reference import bmm_bf16, dequant, quantize_mxfp8

N, K = 1024, 4096
ROWS = [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 48, 64, 65, 96, 127, 128, 129, 192, 255, 256, 257, 384, 511, 512,
        513, 768, 1000, 1024, 1536, 2048, 3000, 4096, 8192, 16372, 16384]


@pytest.fixture(scope="module")
def weights():
    """(unshuffled, shuffled) weight and its 32x32 scales, for 8 groups."""
    from aiter.ops.shuffle import shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randint(0, 256, (8, N, K), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    w = w.masked_fill(w & 0x7F == 0x7F, 0).view(torch.float8_e4m3fn)  # no NaN codes
    ws = torch.randint(116, 128, (8, N // 32, K // 32), generator=g, device="cuda", dtype=torch.uint8)
    return w, shuffle_weight(w.view(8 * N, K), layout=(16, 16)).view(8, N, K), ws


def _x(m, groups, seed, subnormal=True):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((m, groups, K), generator=gen, device="cuda") * 2).to(torch.float8_e4m3fn)
    xs = torch.randint(110, 135, (m, groups, K // 32), generator=gen, device="cuda", dtype=torch.uint8)
    xs[:, :, ::7] = torch.randint(80, 160, xs[:, :, ::7].shape, generator=gen, device="cuda", dtype=torch.uint8)
    if m > 3:
        x[1] = 0  # all-zero output row: scale byte 0
        xs[2] = 1 if subnormal else 100  # outputs down to fp32 subnormals
        xs[3] = 200  # large row
    return x, xs


def _aiter_flydsl(x, xs, w_sh, ws, cfg):
    from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8

    m, g, _ = x.shape
    out = torch.empty((m, g, N), dtype=torch.bfloat16, device=x.device)
    name = "flydsl_bmm_mxfp8_mfma_" + host._kernel.config_name(cfg)
    return run_bmm_a8w8_mxfp8(x, w_sh, xs, ws, out, kernel_name=name).view(m, g * N)


def _aiter_dispatch(x, xs, w_sh, ws):
    from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle

    m, g, _ = x.shape
    return batched_gemm_a8w8_mxscale_bpreshuffle(x, w_sh, xs, ws, torch.bfloat16).reshape(m, g * N)


def _equal(got, y):
    rq, rs = quantize_mxfp8(y)
    assert torch.equal(got[1], rs), "scales"
    assert torch.equal(got[0].view(torch.uint8), rq.view(torch.uint8)), "codes"


def _check(x, xs, w, w_sh, ws, cfg=None, cos=True):
    m, g, _ = x.shape
    q, s = mxfp8_bmm_mxfp8_out(x.view(m, g * K), xs.view(m, -1), w_sh, ws, cfg=cfg)
    _equal((q, s), _aiter_flydsl(x, xs, w_sh, ws, cfg or host.select(m)))
    if cos:
        fq, fs = quantize_mxfp8(bmm_bf16(x, xs, w, ws))
        got = dequant(q.view(m, 1, -1), s.view(m, 1, -1), 1, 32).double()
        ref = dequant(fq.view(m, 1, -1), fs.view(m, 1, -1), 1, 32).double()
        assert torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0) >= 0.9999
    return q, s


@pytest.mark.parametrize("m", ROWS)
def test_bitwise(weights, m):
    w, w_sh, ws = weights
    x, xs = _x(m, 8, m)
    _check(x, xs, w, w_sh, ws, cos=m <= 4096)


@pytest.mark.parametrize("m", [17, 32, 2048, 8192])
def test_matches_aiter_dispatch(weights, m):
    """Where AITER's table runs OPUS (subnormal outputs aside, which OPUS flushes)."""
    _, w_sh, ws = weights
    x, xs = _x(m, 8, m, subnormal=False)
    _equal(mxfp8_bmm_mxfp8_out(x.view(m, -1), xs.view(m, -1), w_sh, ws), _aiter_dispatch(x, xs, w_sh, ws))


@pytest.mark.parametrize("name", sorted({name for _, name in host._TABLE}))
def test_every_config(weights, name):
    """Each tuned config at a partial last M tile, wherever the table puts it."""
    w, w_sh, ws = weights
    cfg = host.parse_config(name)
    m = 3 * cfg.tile_m + 5
    x, xs = _x(m, 8, cfg.tile_m)
    _check(x, xs, w, w_sh, ws, cfg=cfg, cos=False)


@pytest.mark.parametrize("groups", [4, 2, 1])
@pytest.mark.parametrize("m", [5, 300, 2048])
def test_fewer_groups(weights, groups, m):
    w, w_sh, ws = weights
    x, xs = _x(m, groups, groups * m)
    _check(x, xs, w[:groups], w_sh[:groups].contiguous(), ws[:groups].contiguous(), cos=False)


@pytest.mark.parametrize("m", [16, 1000])
def test_graph_replay_deterministic(weights, m):
    _, w_sh, ws = weights
    x, xs = _x(m, 8, 3)
    x, xs = x.view(m, -1), xs.view(m, -1)
    q, s = torch.empty((m, 8 * N), dtype=torch.float8_e4m3fn, device="cuda"), torch.empty(
        (m, 8 * N // 32), dtype=torch.uint8, device="cuda")
    q0, s0 = mxfp8_bmm_mxfp8_out(x, xs, w_sh, ws)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        mxfp8_bmm_mxfp8_out(x, xs, w_sh, ws, q, s)
    for _ in range(2):
        q.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(q.view(torch.uint8), q0.view(torch.uint8)) and torch.equal(s, s0)
