# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Check each MXFP8 GEMM schedule against the contract in ``mxfp8/reference.py``.

The cases are rows of the MiniMax-M3.1-Flash table (the rows of ``mxfp8_gemm_tuned.py``
and the decode windows of ``_select_config``) at an M inside the row's range, run with their explicit
configuration.
"""

import pytest
import torch
from aiter.ops.shuffle import shuffle_scale_a16w4
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.gemm import _Config, mxfp8_linear, preshuffle_mxfp8_weight
from amd_kernels.gemm.mxfp8_common.reference import (
    COS_MIN,
    REL_L2_MAX,
    compare,
    mxfp8_linear_reference,
    quantize_mxfp8,
)

CASES = [
    ("aiter", 300, 2560, 6144, _Config(128, 64, 256, 2, 4, 1)),
    ("decode_fq", 3, 2560, 6144, _Config(4, 16, 128, 0, 0, 1, "decode_fq", 3, 0, 8, 2)),
    ("decode_fq48", 29, 2304, 6144, _Config(8, 48, 128, 0, 0, 1, "decode_fq", 3, 0)),
    ("decode", 41, 2560, 6144, _Config(32, 64, 128, 0, 0, 3, "decode", 1, 0)),
    ("decode48", 250, 6144, 3072, _Config(128, 48, 128, 0, 0, 1, "decode", 3, 0)),
    # the o_proj decode window: four K-waves (RT 6, and RT 7 above 384)
    ("decode_kw4", 320, 6144, 8192, _Config(96, 96, 128, 0, 0, 1, "decode", 4, 1)),
    ("decode_kw4_rt7", 416, 6144, 8192, _Config(112, 96, 128, 0, 0, 1, "decode", 4, 1)),
    ("staged", 1000, 2560, 6144, _Config(128, 256, 128, 2, 4, 3, "staged_ab", 2, 4)),
    ("fourwave", 1200, 6144, 3072, _Config(256, 128, 128, 1, 1, 1, "4wave", 2, 2)),
    ("fourwave_nt", 5000, 6144, 3072, _Config(256, 256, 128, 1, 1, 1, "4wave_nt", 2, 2)),
]
IDS = [case[0] for case in CASES]


def require_gfx950():
    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        pytest.skip("requires gfx950")


def make_inputs(m, n, k, seed=17):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.5
    codes, exponents = quantize_mxfp8(weight)
    return x, codes, exponents, preshuffle_mxfp8_weight(codes, exponents)


def check(name, got, ref, cos_min=COS_MIN, rel_l2_max=REL_L2_MAX):
    assert got.shape == ref.shape, name
    assert torch.isfinite(got).all(), name
    metrics = compare(got, ref)
    print(f"{name}: cos {metrics['cos']:.7f} rel_l2 {metrics['rel_l2']:.2e}")
    assert metrics["cos"] >= cos_min, (name, metrics)
    assert metrics["rel_l2"] <= rel_l2_max, (name, metrics)


def unshuffle_scales(shuffled, rows, cols):
    """The row-major ``[rows, cols]`` E8M0 bytes of scales in AITER's shuffled layout
    (``shuffle_scale_a16w4`` is a permutation: shuffle an index table to invert it)."""
    padded = shuffled.shape[0]
    index = torch.arange(padded * cols, dtype=torch.int32, device=shuffled.device)
    perm = shuffle_scale_a16w4(index.view(padded, cols), 1, False).reshape(-1).long()
    out = torch.empty(padded * cols, dtype=torch.uint8, device=shuffled.device)
    out[perm] = shuffled.reshape(-1).view(torch.uint8)
    return out.view(padded, cols)[:rows]


@pytest.mark.parametrize("name,m,n,k,config", CASES, ids=IDS)
@torch.inference_mode()
def test_gemm_matches_reference(name, m, n, k, config):
    require_gfx950()
    x, codes, exponents, (w, w_scale) = make_inputs(m, n, k)
    ref = mxfp8_linear_reference(x, codes, exponents)
    check(name, mxfp8_linear(x, w, w_scale, config=config), ref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mxfp8_linear(x, w, w_scale, config=config)
    captured.fill_(float("nan"))
    graph.replay()
    check(f"{name} (graph)", captured, ref)


@pytest.mark.parametrize("name,m,n,k,config", CASES, ids=IDS)
@torch.inference_mode()
def test_gemm_deterministic(name, m, n, k, config):
    require_gfx950()
    x, _, _, (w, w_scale) = make_inputs(m, n, k)
    first = mxfp8_linear(x, w, w_scale, config=config)
    for _ in range(3):
        assert torch.equal(mxfp8_linear(x, w, w_scale, config=config), first), name


@torch.inference_mode()
def test_fourwave_dbo_grid_bitwise(monkeypatch):
    """Inside a DBO microbatch the 4-wave kernels launch a plain grid (``g.DBO``): the
    same tiles in the same K order, so the same bytes as the persistent grid."""
    require_gfx950()
    runs = {}
    for dbo in (False, True):
        monkeypatch.setattr(g, "DBO", dbo)
        out = {}
        for name, m, n, k, config in CASES:
            if config.kernel in ("4wave", "4wave_nt"):
                x, _, _, (w, w_scale) = make_inputs(m, n, k)
                out[name] = mxfp8_linear(x, w, w_scale, config=config)
        runs[dbo] = out
    for name, want in runs[False].items():
        got = runs[True][name]
        assert torch.equal(got, want), name
