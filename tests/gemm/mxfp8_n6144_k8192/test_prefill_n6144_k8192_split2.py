# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Full o_proj 256x192 split-2 tile (N6144 K8192, 417-1024 rows): ragged edges, FP64 reference,
the one-launch rows' outputs, determinism, re-armed ready words, graph replay, warm-up, and the
hand-off beside resident MFMA work."""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from amd_kernels.gemm.mxfp8_n6144_k8192 import host as H_n6144_k8192
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950

SPLIT2 = "prefill_n6144_k8192_256x192_split2"
MS = [417, 418, 480, 512, 513, 640, 767, 768, 769, 800, 896, 897, 960, 1000, 1023, 1024]


def _old_config(m):
    """A one-launch row of the split-2 row's window (the 4-wave 128x256 row)."""
    return g._Config(128, 256, 128, 1, 1, 1, g._FOURWAVE_NT_KERNEL, 2, 2)


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("spread", [False, True])
@torch.inference_mode()
def test_prefill_split2_n6144_k8192(m, spread):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 6144, 8192, seed=m)
    if spread:
        exponents = torch.randint(-5, 6, (m, 8192 // 32), device="cuda")
        x = (x.float().view(m, -1, 32) * torch.exp2(exponents.float())[..., None]).view_as(x).bfloat16()
    x[m // 2] = 0
    for allow in (True, False):
        assert g._select_config(m, 6144, 8192, torch.bfloat16, allow).kernel == SPLIT2
    assert g.prequant_code_tile_rows(m, 6144, 8192) == 0
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    call = lambda: g.mxfp8_gemm(a, asc, w, s, torch.bfloat16, activation_scale_preshuffled=True)
    got = call()
    words = H_n6144_k8192._split2_ready_words(got.device)
    torch.cuda.synchronize()
    assert torch.count_nonzero(words) == 0

    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws, accumulate=torch.float64)
    metrics = ref.compare(got, expected)
    assert metrics["cos"] >= ref.COS_MIN and metrics["rel_l2"] <= ref.REL_L2_MAX, metrics
    row_g, row_r = got.float(), expected.float()
    valid = row_r.norm(dim=1) > 0
    assert torch.nn.functional.cosine_similarity(row_g[valid], row_r[valid]).min() >= ref.COS_MIN
    assert ((row_g - row_r).norm(dim=1) / row_r.norm(dim=1).clamp_min(1e-30)).max() <= ref.REL_L2_MAX
    assert torch.count_nonzero(got[m // 2]) == 0
    # the replaced one-launch row: only the K split's fp32 rounding may differ
    old = g.mxfp8_gemm(a, asc, w, s, torch.bfloat16, activation_scale_preshuffled=True, config=_old_config(m))
    assert ref.compare(got, old)["cos"] >= 0.99999
    for _ in range(5):
        assert torch.equal(got, call())
    assert torch.equal(got, g.mxfp8_linear(x, w, s))
    assert torch.equal(got, g.mxfp8_gemm(*ref.quantize_mxfp8(x), w, s, torch.bfloat16))
    graph = torch.cuda.CUDAGraph()
    n_words = len(H_n6144_k8192._SPLIT2_WORDS)
    with torch.cuda.graph(graph):
        captured = call()
    assert len(H_n6144_k8192._SPLIT2_WORDS) == n_words  # the capture records no zero-fill of new words
    for _ in range(3):
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(got, captured)
    assert all(torch.count_nonzero(w) == 0 for w in H_n6144_k8192._SPLIT2_WORDS.values())


@torch.inference_mode()
def test_prefill_split2_warmup(monkeypatch):
    import flydsl.compiler as flyc
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(1024, 6144, 8192)
    H_n6144_k8192._prefill_n6144_k8192_192_split2_launcher.cache_clear()
    g._warmup_weight(w, ws, 1024, torch.bfloat16)
    prepared = [g.mxfp8_e4m3_quantize_aiter(x[:m]) for m in MS]
    compiled = []
    real = flyc.compile
    def counting(*args, **kwargs):
        compiled.append(args[0])
        return real(*args, **kwargs)
    monkeypatch.setattr(flyc, "compile", counting)
    for m, (a, asc) in zip(MS, prepared):
        g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
        g.mxfp8_linear(x[:m], w, ws)
        assert not compiled, (m, compiled)
    # Warmup covers private-word initialization for either captured input layout too.
    for m in (417, 897, 1024):
        a, asc = ref.quantize_mxfp8(x[:m])
        for preshuffled in (False, True):
            scales = g._preshuffle_activation_scale(asc) if preshuffled else asc
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                g.mxfp8_gemm(a, scales, w, ws, torch.bfloat16,
                            activation_scale_preshuffled=preshuffled)
            graph.replay()
            assert not compiled, (m, preshuffled, compiled)
    torch.cuda.synchronize()


@pytest.mark.parametrize("m", [417, 769, 1024])
@torch.inference_mode()
def test_prefill_split2_next_to_mfma(m):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, 6144, 8192)
    a, s = g.mxfp8_e4m3_quantize_aiter(x)
    run = lambda: g.mxfp8_gemm(a, s, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    expected = run()
    spin = mfma_spinner(32)
    other = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    spin(1_000_000, other)
    results = [run() for _ in range(16)]
    torch.cuda.synchronize()
    assert all(torch.equal(expected, got) for got in results)
    assert torch.count_nonzero(H_n6144_k8192._split2_ready_words(expected.device)) == 0


@torch.inference_mode()
def test_prefill_split2_two_streams():
    """Calls on two streams overlap: each stream has its own ready words."""
    require_gfx950()
    runs = []
    for m in (1024, 897):
        x, _, _, (w, ws) = make_inputs(m, 6144, 8192, seed=m)
        a, s = g.mxfp8_e4m3_quantize_aiter(x)
        runs.append(lambda a=a, s=s, w=w, ws=ws: g.mxfp8_gemm(a, s, w, ws, torch.bfloat16,
                                                            activation_scale_preshuffled=True))
    expected = [r() for r in runs]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    outs = [[], []]
    for _ in range(8):
        for i, st in enumerate(streams):
            with torch.cuda.stream(st):
                outs[i].append(runs[i]())
    torch.cuda.synchronize()
    for i in range(2):
        assert all(torch.equal(expected[i], o) for o in outs[i])
    assert all(torch.count_nonzero(w) == 0 for w in H_n6144_k8192._SPLIT2_WORDS.values())


@pytest.mark.parametrize('preshuffled', [False, True])
@torch.inference_mode()
def test_split2_independent_graphs_and_eager(preshuffled):
    """Captured calls must not borrow an eager stream's words or another graph's words."""
    require_gfx950()
    runs, expected = [], []
    for m in (417, 1024):
        x, _, _, (w, ws) = make_inputs(m, 6144, 8192, seed=m)
        a, s = (g.mxfp8_e4m3_quantize_aiter(x) if preshuffled else ref.quantize_mxfp8(x))
        run = lambda a=a, s=s, w=w, ws=ws: g.mxfp8_gemm(
            a, s, w, ws, torch.bfloat16, activation_scale_preshuffled=preshuffled)
        runs.append(run)
        expected.append(run())
    torch.cuda.synchronize()
    graphs, outputs = [], []
    # Both captures use a stream with no eager ready-word buffer.
    capture_stream = torch.cuda.Stream()
    eager_keys = set(H_n6144_k8192._SPLIT2_WORDS)
    for run in runs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            outputs.append(run())
        graphs.append(graph)
    assert set(H_n6144_k8192._SPLIT2_WORDS) == eager_keys
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(16):
        for i, stream in enumerate(streams):
            outputs[i].fill_(float('nan'))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graphs[i].replay()
        eager = runs[1]()  # Also overlaps the two graphs, on the warmed default stream.
        torch.cuda.synchronize()
        assert all(torch.equal(x, y) for x, y in zip(outputs, expected))
        assert torch.equal(eager, expected[1])
