# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The row-major and bf16 forms of the decode GEMM and its swiglu + MXFP8 epilogue against ``reference.py``: odd and
even row tiles, K-waves, split-K, rows beyond M, zero rows, bitwise repeats and graph replay."""

import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as common
from amd_kernels.gemm.mxfp8_decode_prequant import Config, gemm, gemm_swiglu_mxfp8
from amd_kernels.gemm.mxfp8_decode_prequant.reference import (
    COS_MIN,
    REL_L2_MAX,
    SWIGLU_COS_MIN,
    SWIGLU_REL_L2_MAX,
    compare,
    dequantize_mxfp8,
    mxfp8_gemm_reference,
    mxfp8_gemm_swiglu_reference,
    quantize_mxfp8,
    swiglu_clamp_reference,
)

_W = {}


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


def weights(n, k):
    if (n, k) not in _W:
        gen = torch.Generator(device="cuda").manual_seed(n + k)
        q, s = quantize_mxfp8((torch.randn(n, k, generator=gen, device="cuda") * 0.05).bfloat16())
        _W[(n, k)] = (q, s, *common.preshuffle_mxfp8_weight(q, s))
    return _W[(n, k)]


def activation(m, k, seed=0, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(1000 * m + seed)
    x = torch.randn(m, k, generator=gen, device="cuda") * scale
    e = torch.randint(-4, 5, (m, k // 32, 1), generator=gen, device="cuda")
    return (x * torch.exp2(e.expand(m, k // 32, 32).reshape(m, k).float())).bfloat16()


GEMM_CASES = [
    (4096, 1280, 1, Config(1, 4, 5)), (4096, 1280, 16, Config(1, 4, 5)), (4096, 1280, 17, Config(2, 4, 1)),
    (4096, 1280, 50, Config(4, 4, 1)), (4096, 1280, 97, Config(7, 4, 1)), (4096, 1280, 100, Config(4, 4, 5)),
    (4096, 1280, 200, Config(6, 8, 1)), (4096, 1280, 33, Config(3, 4, 1, 1)), (4096, 1280, 192, Config(3, 4, 5, 1, 1, 0)),
    (5120, 2304, 16, Config(1, 5, 3)), (5120, 2304, 64, Config(4, 4, 3, 1)), (5120, 2304, 128, Config(8, 4, 1)),
    (4608, 5120, 32, Config(2, 4, 4)), (4608, 5120, 192, Config(4, 4, 2, 2)), (4608, 5120, 100, Config(4, 8, 1, 1, 1, 2)),
]


@pytest.mark.parametrize("n,k,m,cfg", GEMM_CASES, ids=[f"n{c[0]}_k{c[1]}_m{c[2]}_{c[3].rt}x{c[3].ni}k{c[3].kw}s{c[3].ks}" for c in GEMM_CASES])
@torch.inference_mode()
def test_rowmajor_and_bf16_forms(n, k, m, cfg):
    require_gfx950()
    wq, ws, w, wsh = weights(n, k)
    x = activation(m, k)
    if m > 1:
        x[0].zero_()
    q, s = quantize_mxfp8(x)
    ref = mxfp8_gemm_reference(q, s, wq, ws)
    out = gemm(q, s, w, wsh, cfg)
    r = compare(out, ref)
    assert out.shape == (m, n) and r["cos"] >= COS_MIN and r["rel_l2"] <= REL_L2_MAX, r
    assert torch.equal(out, gemm(q, s, w, wsh, cfg)) and (m == 1 or not out[0].float().abs().max() > 0)
    if (cfg.rt % 2 == 0 or cfg.rt == 1) and cfg.rt * cfg.ni <= 16:  # the registers of the quantizing form
        fq = gemm(x, None, w, wsh, cfg)
        r = compare(fq, ref)
        assert r["cos"] >= COS_MIN and r["rel_l2"] <= REL_L2_MAX, r
        assert torch.equal(fq, out)  # the registers quantize as the reference does: same codes, same sums


@pytest.mark.parametrize("m,cfg", [
    (5, Config(1, 4, 5)), (16, Config(1, 4, 1)), (30, Config(2, 4, 2)), (64, Config(4, 8, 2)), (100, Config(4, 4, 1)),
    (128, Config(8, 4, 1)), (192, Config(4, 4, 4)), (192, Config(3, 4, 4, 1, 1, 2)),
], ids=lambda v: str(v) if isinstance(v, int) else f"{v.rt}x{v.ni}k{v.kw}")
@torch.inference_mode()
def test_swiglu_epilogue(m, cfg):
    require_gfx950()
    n, k = 4608, 5120
    wq, ws, w, wsh = weights(n, k)
    x = activation(m, k, scale=2.0)
    if m > 1:
        x[0].zero_()
    q, s = quantize_mxfp8(x)
    codes, scale = gemm_swiglu_mxfp8(q, s, w, wsh, cfg)
    assert codes.shape == (m, 2304) and codes.dtype == torch.float8_e4m3fn and scale.shape == (m, 72)
    got = dequantize_mxfp8(codes, scale)
    rc, rs = mxfp8_gemm_swiglu_reference(q, s, wq, ws)
    r = compare(got, dequantize_mxfp8(rc, rs))
    assert r["cos"] >= SWIGLU_COS_MIN and r["rel_l2"] <= SWIGLU_REL_L2_MAX, r
    # bit for bit the torch epilogue on this kernel's own bf16 sums
    ec, es = quantize_mxfp8(swiglu_clamp_reference(gemm(q, s, w, wsh, cfg), 10.0))
    assert torch.equal(codes.view(torch.uint8), ec.view(torch.uint8)) and torch.equal(scale, es)
    again = gemm_swiglu_mxfp8(q, s, w, wsh, cfg)
    assert torch.equal(codes.view(torch.uint8), again[0].view(torch.uint8)) and torch.equal(scale, again[1])
    if (cfg.rt % 2 == 0 or cfg.rt == 1) and cfg.rt * cfg.ni <= 16:
        fc, fs = gemm_swiglu_mxfp8(x, None, w, wsh, cfg)
        assert torch.equal(codes.view(torch.uint8), fc.view(torch.uint8)) and torch.equal(scale, fs)


@torch.inference_mode()
def test_swiglu_limit_clamps():
    require_gfx950()
    n, k, m = 4608, 5120, 16
    wq, ws, w, wsh = weights(n, k)
    x = activation(m, k, scale=40.0)  # gate / up sums far beyond the limit
    q, s = quantize_mxfp8(x)
    codes, scale = gemm_swiglu_mxfp8(q, s, w, wsh, Config(1, 4, 4), 10.0)
    rc, rs = mxfp8_gemm_swiglu_reference(q, s, wq, ws, 10.0)
    r = compare(dequantize_mxfp8(codes, scale), dequantize_mxfp8(rc, rs))
    assert r["cos"] >= SWIGLU_COS_MIN and r["rel_l2"] <= SWIGLU_REL_L2_MAX, r


@pytest.mark.parametrize("epi", [False, True])
@torch.inference_mode()
def test_graph_replay_changed_inputs(epi):
    require_gfx950()
    n, k, m = 4608, 5120, 48
    wq, ws, w, wsh = weights(n, k)
    cfg = Config(3, 4, 4)
    x = activation(m, k, 1)
    q, s = quantize_mxfp8(x)
    call = (lambda: gemm_swiglu_mxfp8(q, s, w, wsh, cfg)) if epi else (lambda: gemm(q, s, w, wsh, cfg))
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    for seed in (2, 3):
        nq, ns = quantize_mxfp8(activation(m, k, seed))
        q.copy_(nq)
        s.copy_(ns)
        for t in (out if epi else (out,)):
            t.view(torch.uint8).fill_(0xFF)
        graph.replay()
        if epi:
            ec, es = quantize_mxfp8(swiglu_clamp_reference(mxfp8_gemm_reference(nq, ns, wq, ws), 10.0))
            r = compare(dequantize_mxfp8(*out), dequantize_mxfp8(ec, es))
            assert r["cos"] >= SWIGLU_COS_MIN and r["rel_l2"] <= SWIGLU_REL_L2_MAX, r
        else:
            r = compare(out, mxfp8_gemm_reference(nq, ns, wq, ws))
            assert r["cos"] >= COS_MIN and r["rel_l2"] <= REL_L2_MAX, r
