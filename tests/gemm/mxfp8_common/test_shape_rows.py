# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The bf16 rows of shared-expert gate_up N 3072 x K 6144 and down 6144 x 1536, draft QKV 4608 x 6144 and dense /
draft gate_up 12288 x 6144 (TP2 shards, the split-K decode rows included): the pinned row per M window over 1-65536 and the producer code tile; each
decode row against the fp64 contract of reference.py per output row (a zero row, per-32 scales over +-5 binades)
through the bf16 entry (``mxfp8_linear``) and the producer entry (``mxfp8_gemm_prequant``, same bytes as the bf16
entry for plain decode rows), repeats bitwise, graph replay with a poisoned output and changed
inputs, MFMA co-residency; each prefill row against the contract at its window edges; no compile after
``warmup_mxfp8_gemm_weight``."""
import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import COS_MIN, REL_L2_MAX, compare, mxfp8_gemm_reference, quantize_mxfp8

from .test_gemm import require_gfx950
from .test_shared_expert_rows import _key, _spread

BF16 = torch.bfloat16
# (N, K, first M, last M, (kernel, tile_m, tile_n, K-waves, split-K, w_cm))
ROWS = [
    (3072, 6144, 1, 4, ("decode_fq", 4, 16, 2, 1, None)),
    (3072, 6144, 5, 8, ("decode_fq", 8, 16, 3, 1, None)),
    (3072, 6144, 9, 16, ("decode_fq", 16, 48, 1, 8, None)),
    (3072, 6144, 17, 32, ("decode_fq", 32, 48, 2, 4, None)),
    (3072, 6144, 33, 48, ("decode_splitk", 48, 48, 3, 4, None)),
    (3072, 6144, 49, 64, ("decode_splitk", 64, 48, 3, 4, None)),
    (3072, 6144, 65, 80, ("decode_splitk", 80, 48, 3, 4, None)),
    (3072, 6144, 81, 96, ("decode_splitk", 96, 48, 3, 4, None)),
    (3072, 6144, 97, 112, ("decode_splitk", 112, 48, 3, 4, None)),
    (3072, 6144, 113, 128, ("decode_splitk", 128, 48, 3, 4, None)),
    (3072, 6144, 129, 144, ("decode_splitk", 144, 48, 3, 4, None)),
    (3072, 6144, 145, 160, ("decode_splitk", 160, 48, 3, 4, None)),
    (3072, 6144, 161, 176, ("decode_splitk", 176, 48, 3, 4, None)),
    (3072, 6144, 177, 192, ("decode_splitk", 192, 48, 3, 4, None)),
    (3072, 6144, 193, 208, ("decode_splitk", 208, 48, 3, 4, None)),
    (3072, 6144, 209, 224, ("decode_splitk", 112, 48, 3, 2, None)),
    (3072, 6144, 225, 256, ("decode_splitk", 128, 48, 3, 2, None)),
    (3072, 6144, 257, 320, ("decode", 80, 48, 3, 1, None)),
    (3072, 6144, 321, 384, ("decode", 96, 48, 3, 1, None)),
    (3072, 6144, 385, 448, ("decode", 112, 48, 3, 1, 0)),
    (3072, 6144, 449, 512, ("decode", 128, 48, 3, 1, 0)),
    (3072, 6144, 513, 640, ("decode", 160, 48, 3, 1, 0)),
    (3072, 6144, 641, 768, ("decode", 192, 48, 3, 1, 0)),
    (3072, 6144, 769, 896, ("aiter", 128, 64, 0, 1, None)),
    (3072, 6144, 897, 1024, ("staged_ab", 128, 256, 2, 2, None)),
    (3072, 6144, 1025, 1280, ("staged_ab", 128, 128, 2, 1, None)),
    (3072, 6144, 1281, 2560, ("4wave_nt", 256, 128, 2, 1, None)),
    (3072, 6144, 2561, 65536, ("4wave_nt", 256, 256, 2, 1, None)),
    (4608, 6144, 1, 4, ("decode_fq", 4, 32, 2, 1, None)),
    (4608, 6144, 5, 8, ("decode_fq", 8, 32, 2, 1, None)),
    (4608, 6144, 9, 16, ("decode_fq", 16, 64, 2, 3, None)),
    (4608, 6144, 17, 32, ("decode", 16, 32, 1, 1, 0)),
    (4608, 6144, 33, 48, ("decode_splitk", 48, 64, 4, 3, None)),
    (4608, 6144, 49, 64, ("decode_splitk", 64, 64, 4, 3, None)),
    (4608, 6144, 65, 80, ("decode_splitk", 80, 64, 4, 3, None)),
    (4608, 6144, 81, 96, ("decode_splitk", 96, 64, 4, 3, None)),
    (4608, 6144, 97, 112, ("decode_splitk", 112, 64, 4, 3, None)),
    (4608, 6144, 113, 128, ("decode_splitk", 64, 128, 4, 3, None)),
    (4608, 6144, 129, 160, ("decode_splitk", 80, 128, 4, 3, None)),
    (4608, 6144, 161, 192, ("decode_splitk", 96, 128, 4, 3, None)),
    (4608, 6144, 193, 224, ("decode_splitk", 112, 128, 4, 3, None)),
    (4608, 6144, 225, 256, ("decode_splitk", 128, 128, 4, 3, None)),
    (4608, 6144, 257, 288, ("decode", 48, 64, 1, 1, 0)),
    (4608, 6144, 289, 384, ("decode", 64, 64, 1, 1, 0)),
    (4608, 6144, 385, 448, ("decode", 80, 64, 2, 1, 0)),
    (4608, 6144, 449, 512, ("decode", 64, 96, 2, 1, 0)),
    (4608, 6144, 513, 640, ("decode", 128, 96, 2, 1, 0)),
    (4608, 6144, 641, 768, ("decode", 128, 64, 2, 1, 0)),
    (4608, 6144, 769, 896, ("decode", 112, 96, 2, 1, 0)),
    (4608, 6144, 897, 1024, ("decode", 128, 96, 2, 1, 0)),
    (4608, 6144, 1025, 1536, ("4wave", 256, 128, 2, 1, None)),
    (4608, 6144, 1537, 65536, ("4wave_nt", 256, 256, 2, 1, None)),
    (6144, 1536, 1, 8, ("decode_fq", 8, 32, 3, 1, None)),
    (6144, 1536, 9, 16, ("decode_fq", 16, 32, 3, 1, None)),
    (6144, 1536, 17, 32, ("decode_fq", 32, 32, 3, 1, None)),
    (6144, 1536, 33, 48, ("decode", 48, 32, 1, 1, None)),
    (6144, 1536, 49, 64, ("decode", 32, 48, 1, 1, None)),
    (6144, 1536, 65, 96, ("decode", 48, 48, 3, 1, None)),
    (6144, 1536, 97, 128, ("decode", 32, 64, 1, 1, 0)),
    (6144, 1536, 129, 160, ("decode", 80, 48, 3, 1, None)),
    (6144, 1536, 161, 192, ("decode", 48, 64, 1, 1, 0)),
    (6144, 1536, 193, 256, ("decode", 64, 64, 1, 1, 0)),
    (6144, 1536, 257, 288, ("decode", 144, 48, 3, 1, None)),
    (6144, 1536, 289, 320, ("decode", 80, 96, 2, 1, None)),
    (6144, 1536, 321, 384, ("decode", 96, 96, 3, 1, None)),
    (6144, 1536, 385, 448, ("decode", 64, 96, 2, 1, 0)),
    (6144, 1536, 449, 512, ("decode", 128, 96, 2, 1, None)),
    (6144, 1536, 513, 640, ("decode", 160, 96, 2, 1, None)),
    (6144, 1536, 641, 896, ("decode", 112, 96, 2, 1, 0)),
    (6144, 1536, 897, 1024, ("4wave_nt", 256, 128, 2, 1, None)),
    (6144, 1536, 1025, 1280, ("4wave_nt", 128, 256, 2, 1, None)),
    (6144, 1536, 1281, 65536, ("4wave_nt", 256, 256, 2, 1, None)),
    (12288, 6144, 1, 4, ("decode_fq", 8, 32, 1, 1, None)),
    (12288, 6144, 5, 8, ("decode_fq", 8, 48, 4, 1, None)),
    (12288, 6144, 9, 16, ("decode_fq", 16, 48, 3, 1, None)),
    (12288, 6144, 17, 32, ("decode", 32, 48, 1, 1, None)),
    (12288, 6144, 33, 48, ("decode", 48, 48, 1, 1, None)),
    (12288, 6144, 49, 64, ("decode_splitk", 64, 96, 6, 2, None)),
    (12288, 6144, 65, 80, ("decode_splitk", 80, 96, 6, 2, None)),
    (12288, 6144, 81, 96, ("decode_splitk", 96, 96, 6, 2, None)),
    (12288, 6144, 97, 112, ("decode_splitk", 112, 96, 6, 2, None)),
    (12288, 6144, 113, 128, ("decode_splitk", 128, 96, 6, 2, None)),
    (12288, 6144, 129, 144, ("decode_splitk", 144, 96, 6, 2, None)),
    (12288, 6144, 145, 160, ("decode_splitk", 160, 96, 6, 2, None)),
    (12288, 6144, 161, 176, ("decode_splitk", 176, 96, 6, 2, None)),
    (12288, 6144, 177, 192, ("decode_splitk", 192, 96, 6, 2, None)),
    (12288, 6144, 193, 208, ("decode_splitk", 208, 96, 6, 2, None)),
    (12288, 6144, 209, 224, ("decode_splitk", 224, 96, 6, 2, None)),
    (12288, 6144, 225, 256, ("decode", 128, 96, 2, 1, None)),
    (12288, 6144, 257, 288, ("decode", 96, 96, 2, 1, 0)),
    (12288, 6144, 289, 320, ("decode", 112, 96, 2, 1, 0)),
    (12288, 6144, 321, 384, ("decode", 128, 96, 2, 1, 0)),
    (12288, 6144, 385, 448, ("decode", 160, 96, 2, 1, 0)),
    (12288, 6144, 449, 512, ("4wave_nt", 256, 128, 2, 1, None)),
    (12288, 6144, 513, 640, ("decode", 160, 96, 2, 1, 0)),
    (12288, 6144, 641, 768, ("decode", 128, 96, 1, 1, 0)),
    (12288, 6144, 769, 65536, ("4wave_nt", 256, 256, 2, 1, None)),
]
SHAPES = sorted({(r[0], r[1]) for r in ROWS})
DECODE_KINDS = ("decode", "decode_fq", "decode_splitk")
DECODE = [r for r in ROWS if r[4][0] in DECODE_KINDS]
PREFILL = [r for r in ROWS if r[4][0] not in DECODE_KINDS]


def _edges(lo, hi):
    return sorted({lo, lo + 1, (lo + hi) // 2, hi - 1, hi} & set(range(lo, hi + 1)))


def test_tuned_shapes():
    assert set(SHAPES) <= G.tuned_shapes(BF16)


def test_dispatch_edges():
    for n, k, lo, hi, row in ROWS:
        for m in _edges(lo, hi):
            cfg = G._select_config(m, n, k)
            assert _key(cfg) == row, (n, k, m, cfg)
            decode = row[0] in DECODE_KINDS
            assert G.prequant_code_tile_rows(m, n, k) == (max(16, row[1]) if decode else 0), (n, k, m)
            if not decode:
                assert G._select_config(m, n, k, allow_decode=False) == cfg
    for n, k in SHAPES:
        rows = sorted((r[2], r[3]) for r in ROWS if (r[0], r[1]) == (n, k))
        assert rows[0][0] == 1 and rows[-1][1] == 65536, (n, k)
        assert all(a[1] + 1 == b[0] for a, b in zip(rows, rows[1:])), (n, k)  # contiguous, no overlap


_WEIGHTS = {}


def _weights(n, k):
    if (n, k) not in _WEIGHTS:
        gen = torch.Generator(device="cuda").manual_seed(n + k)
        q, s = quantize_mxfp8((_spread((n, k), gen) * 0.02).bfloat16())
        _WEIGHTS[(n, k)] = (q, s, *G.preshuffle_mxfp8_weight(q, s))
    return _WEIGHTS[(n, k)]


def _check_rows(out, exact, lo, tag):
    got = out[lo:].double()
    cos = torch.nn.functional.cosine_similarity(got, exact[lo:], dim=1)
    rel = (got - exact[lo:]).norm(dim=1) / exact[lo:].norm(dim=1)
    assert cos.min() >= 0.99999 and rel.max() <= 1e-3, (tag, float(cos.min()), float(rel.max()))


DECODE_CASES = []
for _n, _k, _lo, _hi, _row in DECODE:
    for _m in sorted({_lo, _hi, (_lo + _hi) // 2}):
        DECODE_CASES.append((_n, _k, _m, _row))


@pytest.mark.parametrize("n,k,m,row", DECODE_CASES, ids=[f"n{c[0]}_k{c[1]}_m{c[2]}" for c in DECODE_CASES])
@torch.inference_mode()
def test_decode_rows(n, k, m, row):
    require_gfx950()
    qw, sw, w, ws = _weights(n, k)
    gen = torch.Generator(device="cuda").manual_seed(m + n)
    x = _spread((m, k), gen).bfloat16()
    if m > 1:
        x[0].zero_()
    out = G.mxfp8_linear(x, w, ws, BF16)
    # the table's row is the pinned one (test_dispatch_edges): its explicit config gives the same bytes
    explicit = G.mxfp8_linear(x, w, ws, BF16, config=G._select_config(m, n, k))
    assert torch.equal(explicit.view(torch.int16), out.view(torch.int16))
    qx, sx = quantize_mxfp8(x)
    exact = mxfp8_gemm_reference(qx, sx, qw, sw, accumulate=torch.float64).double()
    lo = 1 if m > 1 else 0
    if m > 1:
        assert not out[0].any()
    _check_rows(out, exact, lo, (n, k, m))
    for _ in range(3):
        assert torch.equal(G.mxfp8_linear(x, w, ws, BF16).view(torch.int16), out.view(torch.int16))
    tile = G.prequant_code_tile_rows(m, n, k)
    q, s = G.mxfp8_e4m3_quantize_shuffled(x, tile)
    shuffled = G.mxfp8_gemm_prequant(q, s, w, ws, BF16, m)
    _check_rows(shuffled, exact, lo, (n, k, m, "codes"))
    if row[0] in ("decode", "decode_splitk"):
        assert torch.equal(shuffled.view(torch.int16), out.view(torch.int16))
    xs = x.clone()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        gout = G.mxfp8_linear(xs, w, ws, BF16)
    gout.fill_(float("nan"))
    gr.replay()
    torch.cuda.synchronize()
    assert torch.equal(gout.view(torch.int16), out.view(torch.int16))
    xs.copy_(_spread((m, k), gen).bfloat16())
    gr.replay()
    eager = G.mxfp8_linear(xs, w, ws, BF16)
    torch.cuda.synchronize()
    assert torch.equal(gout.view(torch.int16), eager.view(torch.int16))


PREFILL_CASES = []
for _n, _k, _lo, _hi, _row in PREFILL:
    for _m in sorted({_lo, min(_hi, 8192), *([_lo + 255] if _lo + 255 < min(_hi, 8192) else [])}):
        PREFILL_CASES.append((_n, _k, _m, _row))


@pytest.mark.parametrize("n,k,m,row", PREFILL_CASES, ids=[f"n{c[0]}_k{c[1]}_m{c[2]}" for c in PREFILL_CASES])
@torch.inference_mode()
def test_prefill_rows(n, k, m, row):
    require_gfx950()
    qw, sw, w, ws = _weights(n, k)
    torch.manual_seed(m)
    x = torch.randn((m, k), device="cuda", dtype=BF16)
    out = G.mxfp8_linear(x, w, ws, BF16)
    ref = mxfp8_gemm_reference(*quantize_mxfp8(x), qw, sw)
    metrics = compare(out, ref)
    assert metrics["cos"] >= COS_MIN and metrics["rel_l2"] <= REL_L2_MAX, (n, k, m, metrics)
    assert torch.equal(G.mxfp8_linear(x, w, ws, BF16), out)
    a, s = G.mxfp8_e4m3_quantize_aiter(x)
    assert torch.equal(G.mxfp8_gemm_prequant(a, s, w, ws, BF16, m), out)


@pytest.mark.parametrize("n,k", SHAPES)
@torch.inference_mode()
def test_no_compile_after_warmup(n, k, monkeypatch):
    """After ``warmup_mxfp8_gemm_weight`` up to 32768 rows nothing compiles: every M of 1-1024 and every window edge
    above, through the bf16, codes and producer entries."""
    from flydsl.compiler.jit_function import MlirCompiler

    require_gfx950()
    _, _, w, ws = _weights(n, k)
    G.warmup_mxfp8_gemm_weight(w, ws, 32768, BF16)
    torch.cuda.synchronize()
    compiled = []
    real = MlirCompiler.compile

    def counting(self, *a, **kw):
        compiled.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(MlirCompiler, "compile", counting)
    edges = {e for r in ROWS if (r[0], r[1]) == (n, k) for e in (r[2], r[3]) if 1024 < e <= 32768}
    x = torch.zeros((32768, k), device="cuda", dtype=BF16)
    for m in [*range(1, 1025), *sorted(edges)]:
        G.mxfp8_linear(x[:m], w, ws, BF16)
        tile = G.prequant_code_tile_rows(m, n, k)
        if tile:
            q, s = G.mxfp8_e4m3_quantize_shuffled(x[:m], tile)
        else:
            q, s = G.mxfp8_e4m3_quantize_aiter(x[:m])
        G.mxfp8_gemm_prequant(q, s, w, ws, BF16, m)
    torch.cuda.synchronize()
    assert not compiled, len(compiled)


@pytest.mark.parametrize("n,k,m", [(r[0], r[1], r[3]) for r in DECODE])
def test_mfma_co_residency(n, k, m):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    _, _, w, ws = _weights(n, k)
    x = torch.randn((m, k), device="cuda", dtype=BF16)
    ref = G.mxfp8_linear(x, w, ws, BF16)
    torch.cuda.synchronize()
    spinner = mfma_spinner(256)
    side = torch.cuda.Stream(priority=-1)
    for _ in range(3):
        spinner(100_000, side)
        got = G.mxfp8_linear(x, w, ws, BF16)
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
