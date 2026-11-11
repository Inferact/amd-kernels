# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MLP block and the norm + GEMM: bit for bit the per-kernel chains they replace (the GEMM calls the MXFP8 linear
integration makes for codes / bf16 input), cosine ~1 against the torch references, the first call compiles every
variant of 1-2048 rows."""
import json
import os
import subprocess
import sys

import pytest
import torch

from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4 import gemma_rmsnorm_mxfp8_linear, mxfp8_swiglu_oai_mlp
from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4.reference import mlp_reference, norm_linear_reference
from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_gateup_swiglu_n6144_k6144_i3072 import host as GUS
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144 import gemma_rmsnorm_mxfp8

H, I, EPS = 6144, 3072, 1e-6
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
ROWS = [1, 2, 7, 8, 9, 16, 17, 21, 32, 33, 56, 64, 100, 128, 129, 224, 256, 300, 448, 512, 513, 896, 1024, 1792, 2048]


def _weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = (torch.randn(n, k, device="cuda", generator=g) * 0.6).to(torch.float8_e4m3fn)
    exp = torch.randint(117, 122, (n, k // 32), device="cuda", generator=g).to(torch.uint8)
    w, ws = G.preshuffle_mxfp8_weight(codes.clone(), exp.clone())
    return codes, exp, w, ws


@pytest.fixture(scope="module")
def weights():
    return _weight(2 * I, H, 1), _weight(H, I, 2), _weight(1280, H, 3), torch.randn(H, device="cuda").bfloat16() * 0.1


def _codes(x, nw, m, n, k):
    """The producer's codes (the plain norm in the consumer's layout) and the bf16 rows they quantize."""
    tile = G.prequant_code_tile_rows(m, n, k, torch.bfloat16)
    q, s = gemma_rmsnorm_mxfp8(x, nw, EPS, tile)
    return q, s


def _per_kernel_mlp(q, s, m, gu, gus, dw, dws, form):
    """The per-kernel chain (the MXFP8 GEMM picks of the linear integration)."""
    from amd_kernels.activation.swiglu_oai_mxfp8_i3072 import swiglu_oai_mxfp8

    def gemm(q, s, w, ws):
        n, k = w.shape
        if G.prequant_code_tile_rows(m, n, k, torch.bfloat16):
            return G.mxfp8_gemm_shuffled(q, s, w, ws, torch.bfloat16, m).reshape(m, n)
        return G.mxfp8_gemm(q, s, w, ws, torch.bfloat16, activation_scale_preshuffled=True).reshape(m, n)

    if m <= 8 and q.shape[0] == 16:
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
    got = mxfp8_swiglu_oai_mlp(q, s, m, gu, gus, dw, dws, ALPHA, BETA, LIMIT, form=form)
    want = _per_kernel_mlp(q, s, m, gu, gus, dw, dws, form)
    torch.cuda.synchronize()
    assert got.shape == (m, H) and torch.equal(got.view(torch.int16), want.view(torch.int16))
    from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144.reference import gemma_rmsnorm_reference
    ref = mlp_reference(gemma_rmsnorm_reference(x, nw, EPS), guc, gue, dc, de, ALPHA, BETA, LIMIT, form)
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.999, float(cos)


@pytest.mark.parametrize("m", [0] + ROWS + [4096, 8192])
def test_norm_linear(weights, m):
    _, _, (cc, ce, cw, cws), nw = weights
    x = torch.randn(m, H, device="cuda").bfloat16()
    got = gemma_rmsnorm_mxfp8_linear(x, nw, EPS, cw, cws)
    assert got.shape == (m, 1280)
    if not m:
        return
    q, s = _codes(x, nw, m, 1280, H)
    if G.prequant_code_tile_rows(m, 1280, H, torch.bfloat16):
        want = G.mxfp8_gemm_shuffled(q, s, cw, cws, torch.bfloat16, m)
    else:
        want = G.mxfp8_gemm(q, s, cw, cws, torch.bfloat16, activation_scale_preshuffled=True)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.reshape(m, 1280).view(torch.int16))
    ref = norm_linear_reference(x, nw, EPS, cc, ce)
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.999, float(cos)


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    warm_up.set_auto(True)
    (_, _, gu, gus), (_, _, dw, dws), (_, _, cw, cws), nw = _weight(2 * I, H, 1), _weight(H, I, 2), \
        _weight(1280, H, 3), torch.zeros(H, device="cuda").bfloat16()
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    bad = []
    for m in ROWS + [3, 5, 11, 40, 72, 96, 160, 192, 200, 320, 384, 640, 768, 1000, 1280, 1536, 2047]:
        before = len(late)
        x = torch.zeros(m, H, device="cuda").bfloat16()
        q, s = _codes(x, nw, m, 2 * I, H)
        mxfp8_swiglu_oai_mlp(q, s, m, gu, gus, dw, dws, ALPHA, BETA, LIMIT, form="bf16_chain")
        gemma_rmsnorm_mxfp8_linear(x, nw, EPS, cw, cws)
        if len(late) > before:
            bad.append(m)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=2400,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 2, result


def test_stack_and_context_kv_blockers():
    from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4 import DenseLayer, context_kv_blocker, stack_blocker

    bf, f8 = torch.bfloat16, torch.float8_e4m3fn
    norm = torch.empty(H, dtype=bf)
    gu, gus = torch.empty(2 * I, H, dtype=f8), torch.empty(2 * I, H // 32, dtype=torch.uint8)
    dw, dws = torch.empty(H, I, dtype=f8), torch.empty(H, I // 32, dtype=torch.uint8)
    layer = DenseLayer(norm, (2304, H), norm, gu, gus, dw, dws)
    fc, emb = torch.empty(H, 5 * H, dtype=bf), torch.empty(64, H, dtype=bf)
    ok = dict(world=4, max_rows=2048, on_library=[True] * 20, fc_weight=fc, embedding=emb, layers=[layer] * 5)
    assert stack_blocker(**ok) is None
    for key, bad in (("world", 2), ("max_rows", 2049), ("max_rows", 0), ("on_library", [True] * 19 + [False]),
                     ("fc_weight", fc[:, :H]), ("embedding", emb.float()), ("embedding", emb.T), ("layers", []),
                     ("layers", [DenseLayer(norm, (2304, H), norm, gu.float(), gus, dw, dws)]),
                     ("layers", [DenseLayer(norm, (2304, 2048), norm, gu, gus, dw, dws)]),
                     ("layers", [DenseLayer(norm.float(), (2304, H), norm, gu, gus, dw, dws)])):
        assert stack_blocker(**{**ok, key: bad}) is not None, key
    caches = [torch.empty(4, 1, 128, 256, dtype=torch.uint8) for _ in range(5)]
    scales = [torch.ones(1) for _ in range(5)]
    cs = torch.empty(4096, 64, dtype=bf)
    proj = torch.empty(5 * 2 * 128, H, dtype=f8)
    ctx = dict(proj_on_library=True, proj_w=proj, rotary_dim=64, neox=True, cos_sin_cache=cs, kv_caches=caches,
               k_scales=scales, v_scales=scales)
    assert context_kv_blocker(**ctx) is None
    for key, bad in (("proj_on_library", False), ("proj_w", proj[:640]), ("rotary_dim", 32), ("neox", False),
                     ("cos_sin_cache", cs.float()), ("kv_caches", [c.bfloat16() for c in caches]),
                     ("k_scales", [s.half() for s in scales])):
        assert context_kv_blocker(**{**ctx, key: bad}) is not None, key


if __name__ == "__main__":
    print(json.dumps(scenario()))
