# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MLP block and the norm + GEMM at TP2 (I 6144, context K/V N 2560): bit for bit the per-kernel chains they
replace (the GEMM calls the MXFP8 linear integration makes for codes / bf16 input), cosine ~1 against the torch
references, the first call compiles every variant of 1-2048 rows."""
import json
import os
import subprocess
import sys
import time

import pytest
import torch

from amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2 import gemma_rmsnorm_mxfp8_linear, mxfp8_swiglu_oai_mlp
from amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2.reference import mlp_reference, norm_linear_reference
from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_gateup_swiglu_n12288_k6144_i6144 import host as GUS
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144 import gemma_rmsnorm_mxfp8

H, I, CTX_N, EPS = 6144, 6144, 2560, 1e-6
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
# graph sizes of 7-row draft blocks (7R, R = 1..64 incl. the c32-c52 band), the fused / tiled / row-major edges
ROWS = [1, 2, 7, 8, 9, 14, 16, 17, 21, 32, 33, 56, 64, 100, 112, 128, 129, 160, 161, 224, 225, 256, 257, 280, 288,
        300, 320, 336, 364, 384, 385, 416, 448, 449, 512, 513, 640, 768, 769, 896, 1024, 1792, 2048]


def _weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = (torch.randn(n, k, device="cuda", generator=g) * 0.6).to(torch.float8_e4m3fn)
    exp = torch.randint(117, 122, (n, k // 32), device="cuda", generator=g).to(torch.uint8)
    w, ws = G.preshuffle_mxfp8_weight(codes.clone(), exp.clone())
    return codes, exp, w, ws


@pytest.fixture(scope="module")
def weights():
    return _weight(2 * I, H, 1), _weight(H, I, 2), _weight(CTX_N, H, 3), torch.randn(H, device="cuda").bfloat16() * 0.1


def _codes(x, nw, m, n, k):
    """The producer's codes (the plain norm in the consumer's layout)."""
    tile = G.prequant_code_tile_rows(m, n, k, torch.bfloat16)
    return gemma_rmsnorm_mxfp8(x, nw, EPS, tile)


def _per_kernel_mlp(q, s, m, gu, gus, dw, dws, form):
    """The per-kernel chain (the MXFP8 GEMM picks of the linear integration)."""
    from amd_kernels.activation.swiglu_oai_mxfp8_i6144 import swiglu_oai_mxfp8

    def gemm(q, s, w, ws):
        n, k = w.shape
        if G.prequant_code_tile_rows(m, n, k, torch.bfloat16):
            return G.mxfp8_gemm_shuffled(q, s, w, ws, torch.bfloat16, m).reshape(m, n)
        return G.mxfp8_gemm(q, s, w, ws, torch.bfloat16, activation_scale_preshuffled=True).reshape(m, n)

    if m <= 16 and q.shape[0] == 16:
        act = GUS.mxfp8_gateup_swiglu_bf16(q, s, gu, gus, m, ALPHA, BETA, LIMIT, form=form)
        return G.mxfp8_linear(act.reshape(-1, I), dw, dws, torch.bfloat16).reshape(m, H)
    h = gemm(q, s, gu, gus)
    tile = G.prequant_code_tile_rows(m, H, I, torch.bfloat16)
    qd, sd = swiglu_oai_mxfp8(h, ALPHA, BETA, LIMIT, tile, form=form)
    return gemm(qd, sd, dw, dws)


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("form", ["bf16_chain", "fp32"])
def test_mlp(weights, m, form):
    (guc, gue, gu, gus), (dc, de, dw, dws), _, nw = weights
    x = torch.randn(m, H, device="cuda").bfloat16()
    q, s = _codes(x, nw, m, 2 * I, H)
    if m <= 16:
        assert q.shape[0] == 16  # the fused gate_up path's code block
    got = mxfp8_swiglu_oai_mlp(q, s, m, gu, gus, dw, dws, ALPHA, BETA, LIMIT, form=form)
    want = _per_kernel_mlp(q, s, m, gu, gus, dw, dws, form)
    torch.cuda.synchronize()
    assert got.shape == (m, H) and torch.equal(got.view(torch.int16), want.view(torch.int16))
    from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144.reference import gemma_rmsnorm_reference
    ref = mlp_reference(gemma_rmsnorm_reference(x, nw, EPS), guc, gue, dc, de, ALPHA, BETA, LIMIT, form)
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.999, float(cos)


@pytest.mark.parametrize("m", [0] + ROWS + [1536, 1537, 3072, 3073, 4096, 4801, 8192])
def test_norm_linear(weights, m):
    _, _, (cc, ce, cw, cws), nw = weights
    x = torch.randn(m, H, device="cuda").bfloat16()
    got = gemma_rmsnorm_mxfp8_linear(x, nw, EPS, cw, cws)
    assert got.shape == (m, CTX_N)
    if not m:
        return
    q, s = _codes(x, nw, m, CTX_N, H)
    if G.prequant_code_tile_rows(m, CTX_N, H, torch.bfloat16):
        want = G.mxfp8_gemm_shuffled(q, s, cw, cws, torch.bfloat16, m)
    else:
        want = G.mxfp8_gemm(q, s, cw, cws, torch.bfloat16, activation_scale_preshuffled=True)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.reshape(m, CTX_N).view(torch.int16))
    ref = norm_linear_reference(x, nw, EPS, cc, ce)
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.999, float(cos)


def test_mlp_graph_replay(weights):
    """Captured calls replay the eager bits (fused, tiled and row-major gate_up rows)."""
    (_, _, gu, gus), (_, _, dw, dws), _, nw = weights
    for m in (7, 16, 112, 364, 1024):
        x = torch.randn(m, H, device="cuda").bfloat16()
        q, s = _codes(x, nw, m, 2 * I, H)
        want = mxfp8_swiglu_oai_mlp(q, s, m, gu, gus, dw, dws, ALPHA, BETA, LIMIT, form="bf16_chain")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = mxfp8_swiglu_oai_mlp(q, s, m, gu, gus, dw, dws, ALPHA, BETA, LIMIT, form="bf16_chain")
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), m


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    warm_up.set_auto(True)
    (_, _, gu, gus), (_, _, dw, dws), (_, _, cw, cws), nw = _weight(2 * I, H, 1), _weight(H, I, 2), \
        _weight(CTX_N, H, 3), torch.zeros(H, device="cuda").bfloat16()
    t0 = time.perf_counter()
    for w, ws in ((gu, gus), (dw, dws), (cw, cws)):  # the linear integration's warm-up of each weight at load
        G.warmup_mxfp8_gemm_weight(w, ws, 32768, torch.bfloat16)
    t1 = time.perf_counter()
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    bad = []
    for m in ROWS + [3, 5, 11, 15, 40, 63, 72, 96, 113, 192, 200, 231, 287, 343, 350, 357, 371, 399, 640, 1000, 1280,
                     1536, 2047]:
        before = len(late)
        x = torch.zeros(m, H, device="cuda").bfloat16()
        q, s = _codes(x, nw, m, 2 * I, H)
        for form in ("bf16_chain", "fp32"):
            mxfp8_swiglu_oai_mlp(q, s, m, gu, gus, dw, dws, ALPHA, BETA, LIMIT, form=form)
        gemma_rmsnorm_mxfp8_linear(x, nw, EPS, cw, cws)
        if len(late) > before:
            bad.append(m)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed()),
            "gemm_warm_up_s": round(t1 - t0), "entries_s": round(time.perf_counter() - t1)}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=2400,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 3, result
    print(result)


def test_stack_and_context_kv_blockers():
    from amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2 import DenseLayer, context_kv_blocker, stack_blocker

    bf, f8 = torch.bfloat16, torch.float8_e4m3fn
    norm = torch.empty(H, dtype=bf)
    gu, gus = torch.empty(2 * I, H, dtype=f8), torch.empty(2 * I, H // 32, dtype=torch.uint8)
    dw, dws = torch.empty(H, I, dtype=f8), torch.empty(H, I // 32, dtype=torch.uint8)
    layer = DenseLayer(norm, (4608, H), norm, gu, gus, dw, dws)
    fc, emb = torch.empty(H, 5 * H, dtype=bf), torch.empty(64, H, dtype=bf)
    ok = dict(world=2, max_rows=2048, on_library=[True] * 20, fc_weight=fc, embedding=emb, layers=[layer] * 5)
    assert stack_blocker(**ok) is None
    tp4 = DenseLayer(norm, (2304, H), norm, gu[:6144], gus[:6144], dw[:, :3072], dws[:, :96])
    for key, bad in (("world", 4), ("world", 1), ("max_rows", 2049), ("max_rows", 0),
                     ("on_library", [True] * 19 + [False]), ("fc_weight", fc[:, :H]), ("embedding", emb.float()),
                     ("embedding", emb.T), ("layers", []), ("layers", [tp4]),
                     ("layers", [DenseLayer(norm, (4608, H), norm, gu.float(), gus, dw, dws)]),
                     ("layers", [DenseLayer(norm, (4608, 4096), norm, gu, gus, dw, dws)]),
                     ("layers", [DenseLayer(norm.float(), (4608, H), norm, gu, gus, dw, dws)])):
        assert stack_blocker(**{**ok, key: bad}) is not None, key
    caches = [torch.empty(4, 2, 128, 256, dtype=torch.uint8) for _ in range(5)]
    scales = [torch.ones(1) for _ in range(5)]
    cs = torch.empty(4096, 64, dtype=bf)
    proj = torch.empty(5 * 2 * 2 * 128, H, dtype=f8)
    ctx = dict(proj_on_library=True, proj_w=proj, rotary_dim=64, neox=True, cos_sin_cache=cs, kv_caches=caches,
               k_scales=scales, v_scales=scales)
    assert context_kv_blocker(**ctx) is None
    for key, bad in (("proj_on_library", False), ("proj_w", proj[:1280]), ("rotary_dim", 32), ("neox", False),
                     ("cos_sin_cache", cs.float()), ("kv_caches", [c.bfloat16() for c in caches]),
                     ("k_scales", [s.half() for s in scales])):
        assert context_kv_blocker(**{**ctx, key: bad}) is not None, key


if __name__ == "__main__":
    print(json.dumps(scenario()))
