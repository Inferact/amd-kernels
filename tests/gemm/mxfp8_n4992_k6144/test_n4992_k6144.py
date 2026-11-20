# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP2 QKV (N 4992, K 6144): every row ``host.select`` / ``host.select_deferred`` dispatches, at its window edges
(window ends are multiples of 16, as vLLM's graph sizes), against the fp64 product of the same quantized operands (a
zero row, spread scales); bitwise repeats and entry agreement (``mxfp8_gemm_prequant`` from the producer's layout,
``mxfp8_linear``, ``mxfp8_gemm``); graph replay over a NaN-poisoned output; deferred partials; every persistent tile,
the row split and the DBO plain grid bitwise equal; guard rows; next to an MFMA co-runner."""

import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common import reference as ref
from amd_kernels.gemm.mxfp8_n4992_k6144 import host as H

from ..mxfp8_common.test_gemm import require_gfx950

N, K = H.N, H.K


def _edges():
    """The first and last row count of every decode and deferred window."""
    tops = sorted({top for top, _ in H.DECODE_ROWS} | {top for top, _ in H.DEFERRED_ROWS})
    ms = {1, 2, 7, 8}
    for top in tops:
        ms |= {top, top + 1}
    return sorted(ms)


def _persistent_edges():
    """Row counts where ``persistent_plan`` changes, both sides, up to 12288."""
    ms, prev = [], None
    for m in range(H.DECODE_MAX + 1, 12289):
        plan = H.persistent_plan(m)
        if plan != prev:
            ms += [m - 1, m] if prev is not None else [m]
            prev = plan
    return ms


DECODE_MS = _edges()
PERSISTENT_MS = _persistent_edges()


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(4992)
    raw = torch.randn((N, K), device="cuda") * 0.02
    raw *= torch.exp2(torch.randint(-5, 6, (N, K // 32, 1), device="cuda").float()).expand(N, K // 32, 32).reshape(N, K)
    codes, scales = ref.quantize_mxfp8(raw.bfloat16())
    return codes, scales, g.preshuffle_mxfp8_weight(codes, scales)


def _activation(m, seed):
    torch.manual_seed(seed)
    x = torch.randn((m, K), device="cuda") * torch.exp2(
        torch.randint(-5, 6, (m, K // 32, 1), device="cuda").float()).expand(m, K // 32, 32).reshape(m, K)
    x = x.bfloat16()
    if m > 1:
        x[m // 2] = 0
    return x


def _check(got, x, codes, scales):
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), codes, scales, accumulate=torch.float64)
    metrics = ref.compare(got, expected)
    assert metrics["cos"] >= ref.COS_MIN and metrics["rel_l2"] <= ref.REL_L2_MAX, metrics
    active = expected.double().norm(dim=1) > 0
    row_l2 = (got.double() - expected.double()).norm(dim=1) / expected.double().norm(dim=1).clamp_min(1e-30)
    assert row_l2[active].max() <= ref.REL_L2_MAX
    if x.shape[0] > 1:
        assert torch.count_nonzero(got[x.shape[0] // 2]) == 0


def _producer_codes(x):
    m = x.shape[0]
    tile = g.prequant_code_tile_rows(m, N, K)
    return (g.mxfp8_e4m3_quantize_shuffled(x, tile) if tile else g.mxfp8_e4m3_quantize_aiter(x)), tile


def _replays(call, want):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for _ in range(3):
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, want)


@pytest.mark.parametrize("m", DECODE_MS + PERSISTENT_MS[::3])
@torch.inference_mode()
def test_rows(weights, m):
    require_gfx950()
    codes, scales, (w, ws) = weights
    x = _activation(m, m)
    (a, asc), tile = _producer_codes(x)
    config = g._select_prequant_config(m, N, K, torch.bfloat16)
    assert (tile != 0) == (config.kernel in g._SHUFFLED_A_KERNELS)
    call = lambda: g.mxfp8_gemm_prequant(a, asc, w, ws, torch.bfloat16, m)  # noqa: E731
    got = call()
    _check(got, x, codes, scales)
    for _ in range(3):
        assert torch.equal(call(), got)
    assert torch.equal(g.mxfp8_linear(x, w, ws), got)
    if not tile:
        assert torch.equal(g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True), got)
    _replays(call, got)


@pytest.mark.parametrize("m", PERSISTENT_MS)
@torch.inference_mode()
def test_persistent_edges(weights, m):
    require_gfx950()
    codes, scales, (w, ws) = weights
    x = _activation(m, m + 1)
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    config = g._select_config(m, N, K, torch.bfloat16, allow_decode=False)
    plan = H.persistent_plan(m)
    assert (config.tile_m, config.tile_n) == plan[0][1]
    assert config.kernel == (H._PERSISTENT_SPLIT_KERNEL if len(plan) > 1 else H._PERSISTENT_KERNEL)
    got = g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    _check(got, x, codes, scales)
    assert torch.equal(got, g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True))
    if config.kernel == H._PERSISTENT_SPLIT_KERNEL:  # the split's launches sum like the one-launch tile
        assert torch.equal(got, g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True,
                                             config=H._persistent(*H.persistent_tile(m))))


@pytest.mark.parametrize("m", [1, 16, 100, H.ROW_MAJOR_FALLBACK_MAX, H.ROW_MAJOR_FALLBACK_MAX + 1, H.DECODE_MAX])
@torch.inference_mode()
def test_row_major_codes(weights, m):
    """Row-major codes below the decode rows' end (``mxfp8_gemm``): the common fallback, then persistent tiles."""
    require_gfx950()
    codes, scales, (w, ws) = weights
    x = _activation(m, 5 * m)
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    got = g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    _check(got, x, codes, scales)
    assert torch.equal(got, g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True))


@pytest.mark.parametrize("m", [m for m in DECODE_MS if m <= H.DEFERRED_ROWS[-1][0] + 1])
@torch.inference_mode()
def test_deferred(weights, m):
    """The chain's split-K partials sum (split order) to the standalone row's output within the contract, read a
    prefix of the producer's codes, and repeat bitwise."""
    require_gfx950()
    codes, scales, (w, ws) = weights
    x = _activation(m, 3 * m)
    (a, asc), tile = _producer_codes(x)
    assert tile
    out = g.mxfp8_gemm_shuffled(a, asc, w, ws, torch.bfloat16, m, defer_reduce=True)
    if not isinstance(out, g.SkPartials):
        assert torch.equal(out, g.mxfp8_gemm_prequant(a, asc, w, ws, torch.bfloat16, m))
        return
    again = g.mxfp8_gemm_shuffled(a, asc, w, ws, torch.bfloat16, m, defer_reduce=True)
    assert torch.equal(out.tmp, again.tmp)
    _check(out.materialize(), x, codes, scales)


@pytest.mark.parametrize("m", [1000, 2049])
@torch.inference_mode()
def test_persistent_tiles_bitwise(weights, m):
    """Every persistent tile sums K in the same order: the tile a row count takes never changes a byte."""
    require_gfx950()
    _, _, (w, ws) = weights
    a, asc = g.mxfp8_e4m3_quantize_aiter(_activation(m, 13))
    outs = []
    for tile in H.PERSISTENT_ROUND_US:
        out = torch.empty((m, N), dtype=torch.bfloat16, device="cuda")
        H.LAUNCHERS[H._PERSISTENT_KERNEL](H._persistent(*tile), a, asc, w, ws, out)
        outs.append(out)
    assert all(torch.equal(outs[0], out) for out in outs[1:])


@pytest.mark.parametrize("m", [4097, 8192, 12288])
@torch.inference_mode()
def test_dbo_plain_grid_bitwise(weights, m, monkeypatch):
    require_gfx950()
    _, _, (w, ws) = weights
    a, asc = g.mxfp8_e4m3_quantize_aiter(_activation(m, 7))
    run = lambda: g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)  # noqa: E731
    persistent = run()
    monkeypatch.setattr(g, "DBO", True)
    assert torch.equal(run(), persistent)


@pytest.mark.parametrize("m", [721, 1025, 2048, 4097, 5632])
@torch.inference_mode()
def test_persistent_output_guards(weights, m):
    require_gfx950()
    _, _, (w, ws) = weights
    a, asc = g.mxfp8_e4m3_quantize_aiter(_activation(m, 11))
    want = g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    arena = torch.empty((m + 2, N), dtype=torch.bfloat16, device="cuda")
    arena.view(torch.int16).fill_(0x3555)
    guard0, guard1 = arena[0].clone(), arena[-1].clone()
    out = arena[1:-1]
    config = g._select_config(m, N, K, torch.bfloat16, allow_decode=False)
    H.LAUNCHERS[config.kernel](config, a, asc, w, ws, out)
    torch.cuda.synchronize()
    assert torch.equal(out, want)
    assert torch.equal(arena[0], guard0) and torch.equal(arena[-1], guard1)


@pytest.mark.parametrize("m", [16, 256, 700, 1024, 4096, 5632])
@torch.inference_mode()
def test_next_to_mfma(weights, m):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    _, _, (w, ws) = weights
    (a, asc), _ = _producer_codes(_activation(m, 5))
    run = lambda: g.mxfp8_gemm_prequant(a, asc, w, ws, torch.bfloat16, m)  # noqa: E731
    want = run()
    spin = mfma_spinner(32)
    side = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    spin(1_000_000, side)
    results = [run() for _ in range(16)]
    torch.cuda.synchronize()
    assert all(torch.equal(want, got) for got in results)

