# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The TP2 o_proj N 6144 x K 4096: every row of ``host`` at its window edges and at the served CUDA-graph sizes, through
the three entries (bf16 ``mxfp8_linear``, producer codes ``mxfp8_gemm_prequant``, row-major codes ``mxfp8_gemm``),
against the FP64 reference; determinism, graph replay and the warm-up."""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from amd_kernels.gemm.mxfp8_n6144_k4096 import host as H

from ..mxfp8_common.test_gemm import make_inputs, require_gfx950

N, K = 6144, 4096
MAX_TOKENS = 32768
# vLLM's capture sizes up to max_cudagraph_capture_size 2048 (decode: 8 rows per request with DSpark 7)
CAPTURE = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 2049, 16))


def _configs(m):
    return (g._select_config(m, N, K, torch.bfloat16), g._select_config(m, N, K, torch.bfloat16, False),
            g._select_prequant_config(m, N, K, torch.bfloat16))


def _edges():
    """The first and last row of every window of the three entries' rows."""
    edges, previous = {1, MAX_TOKENS}, None
    for m in range(1, MAX_TOKENS + 1):
        configs = _configs(m)
        if configs != previous and previous is not None:
            edges |= {m - 1, m}
        previous = configs
    return sorted(edges)


EDGES = _edges()


def test_host_owns_every_row():
    for m in range(1, MAX_TOKENS + 1):
        linear, plain, prequant = _configs(m)
        assert H.select(m, torch.bfloat16, True) == linear and H.select(m, torch.bfloat16, False) == plain
        assert plain.kernel not in g._SHUFFLED_A_KERNELS
        if linear.kernel not in g._SHUFFLED_A_KERNELS:
            assert linear == plain
        tile = g.prequant_code_tile_rows(m, N, K)
        assert tile == (max(16, prequant.tile_m) if prequant.kernel in g._SHUFFLED_A_KERNELS else 0)


def _spread(x):
    m = x.shape[0]
    exponents = torch.randint(-5, 6, (m, K // 32), device="cuda")
    return (x.float().view(m, -1, 32) * torch.exp2(exponents.float())[..., None]).view_as(x).bfloat16()


def _check(got, expected):
    metrics = ref.compare(got, expected)
    assert metrics["cos"] >= ref.COS_MIN and metrics["rel_l2"] <= ref.REL_L2_MAX, metrics
    row_g, row_r = got.float(), expected.float()
    valid = row_r.norm(dim=1) > 0
    assert torch.nn.functional.cosine_similarity(row_g[valid], row_r[valid]).min() >= ref.COS_MIN
    assert ((row_g - row_r).norm(dim=1) / row_r.norm(dim=1).clamp_min(1e-30)).max() <= ref.REL_L2_MAX


def _entries(x, w, s, m):
    tile = g.prequant_code_tile_rows(m, N, K)
    codes = g.mxfp8_e4m3_quantize_shuffled(x, tile) if tile else g.mxfp8_e4m3_quantize_aiter(x)
    plain = g.mxfp8_e4m3_quantize_aiter(x)
    return {
        "linear": lambda: g.mxfp8_linear(x, w, s),
        "prequant": lambda: g.mxfp8_gemm_prequant(*codes, w, s, torch.bfloat16, m),
        "gemm": lambda: g.mxfp8_gemm(*plain, w, s, torch.bfloat16, activation_scale_preshuffled=True),
    }


def _run_entries(m, spread):
    x, wc, ws, (w, s) = make_inputs(m, N, K, seed=m)
    if spread:
        x = _spread(x)
    if m > 1:
        x[m // 2] = 0
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws, accumulate=torch.float64)
    entries = _entries(x, w, s, m)
    outs = {}
    for name, call in entries.items():
        got = call()
        _check(got, expected)
        assert m == 1 or torch.count_nonzero(got[m // 2]) == 0, name
        for _ in range(3):
            assert torch.equal(got, call()), name
        outs[name] = got
    linear, plain, prequant = _configs(m)
    if linear == plain:
        assert torch.equal(outs["linear"], outs["gemm"])
    if linear == prequant:
        assert torch.equal(outs["linear"], outs["prequant"])
    if prequant.kernel not in g._SHUFFLED_A_KERNELS:
        assert torch.equal(outs["prequant"], outs["gemm"])
    return entries, outs


@pytest.mark.parametrize("m", EDGES)
@pytest.mark.parametrize("spread", [False, True])
@torch.inference_mode()
def test_window_edges(m, spread):
    require_gfx950()
    _run_entries(m, spread)


@pytest.mark.parametrize("m", [m for m in CAPTURE if m not in EDGES])
@torch.inference_mode()
def test_capture_sizes(m):
    require_gfx950()
    entries, outs = _run_entries(m, False)
    for name, call in entries.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        for _ in range(2):
            captured.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(outs[name], captured), name


@torch.inference_mode()
def test_warmup_compiles_every_row(monkeypatch):
    import flydsl.compiler as flyc

    require_gfx950()
    _, _, _, (w, s) = make_inputs(16, N, K)
    g._decode_launcher.cache_clear()
    g._warmup_weight(w, s, MAX_TOKENS, torch.bfloat16)
    calls = [(m, name, call) for m in sorted(set(CAPTURE) | set(EDGES))
             for name, call in _entries(torch.zeros((m, K), dtype=torch.bfloat16, device="cuda"), w, s, m).items()]
    compiled = []
    real = flyc.compile

    def counting(*args, **kwargs):
        compiled.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    for m, name, call in calls:
        call()
        assert not compiled, (m, name, compiled)
    torch.cuda.synchronize()
