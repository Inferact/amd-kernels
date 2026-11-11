# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The bf16 decode rows of the MiniMax-M3.1 TP4 shared expert: gate_up N 1536 x K 6144 and down N 6144 x K 768 at
1-256 rows (the AITER "any" rows, split-K 8 / 4 / 1 by M, are slower than vLLM's Triton kernel at small M). These
rows change the K order, so the outputs are not the AITER rows' bytes: each row is checked against the fp64 contract of reference.py per output row (a zero row, per-32
scales over +-5 binades) at the range edges and at M that are not multiples of 16, through the bf16 entry
(``mxfp8_linear``) and the producer entry (``mxfp8_gemm_shuffled``); the launched kernel name, four launches
bitwise equal, graph replay with a poisoned output and changed inputs, and MFMA co-residency. Above 256 rows, fp16
and row-major codes (``allow_decode=False``) keep the AITER rows."""
import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import mxfp8_gemm_reference, quantize_mxfp8

from .test_gemm import require_gfx950
from .test_windows import kernel_names

BF16 = torch.bfloat16
# (N, K, first M, last M, (kernel, tile_m, tile_n, K-waves, split-K, w_cm))
ROWS = [
    (1536, 6144, 1, 4, ("decode_fq", 4, 16, 3, 1, None)),
    (1536, 6144, 5, 8, ("decode_fq", 8, 16, 3, 1, None)),
    (1536, 6144, 9, 16, ("decode_fq", 16, 32, 2, 3, None)),
    (1536, 6144, 17, 32, ("decode", 16, 16, 3, 1, None)),
    (1536, 6144, 33, 64, ("decode", 16, 16, 1, 1, 0)),
    (1536, 6144, 65, 80, ("decode", 16, 32, 3, 1, None)),
    (1536, 6144, 81, 96, ("decode", 32, 16, 1, 1, 0)),
    (1536, 6144, 97, 128, ("decode", 16, 32, 1, 1, 0)),
    (1536, 6144, 129, 192, ("decode", 32, 32, 1, 1, 0)),
    (1536, 6144, 193, 256, ("decode", 64, 32, 4, 1, 0)),
    (6144, 768, 1, 8, ("decode_fq", 8, 32, 3, 1, None)),
    (6144, 768, 9, 32, ("decode_fq", 16, 48, 3, 1, None)),
    (6144, 768, 33, 64, ("decode_fq", 16, 96, 3, 1, None)),
    (6144, 768, 65, 80, ("decode_fq", 16, 64, 1, 1, None)),
    (6144, 768, 81, 128, ("decode_fq", 32, 96, 3, 1, None)),
    (6144, 768, 129, 160, ("decode_fq", 32, 64, 3, 1, None)),
    (6144, 768, 161, 192, ("decode", 48, 64, 1, 1, 0)),
    (6144, 768, 193, 256, ("decode_fq", 64, 96, 3, 1, None)),
]


def _key(cfg):
    return (cfg.kernel, cfg.tile_m, cfg.tile_n, cfg.waves_m, cfg.split_k, cfg.w_cm)


def _any_row(m, n, k):
    """The AITER "any" row of these shapes."""
    rows = G._load_tuned_configs()[(n, k, "any")]
    return next((c for tm, c in rows if m <= tm), rows[-1][1])


def _edges(lo, hi):
    return sorted({lo, lo + 1, (lo + hi) // 2, hi - 1, hi} & set(range(lo, hi + 1)))


def test_dispatch_edges():
    for n, k, lo, hi, row in ROWS:
        for m in _edges(lo, hi):
            cfg = G._select_config(m, n, k)
            assert _key(cfg) == row, (n, k, m, cfg)
            assert G.prequant_code_tile_rows(m, n, k) == max(16, row[1]), (n, k, m)
            assert G._select_config(m, n, k, torch.float16) == _any_row(m, n, k)
            assert G._select_config(m, n, k, allow_decode=False) == _any_row(m, n, k)
    for n, k in {(r[0], r[1]) for r in ROWS}:
        covered = sorted(m for r in ROWS if (r[0], r[1]) == (n, k) for m in range(r[2], r[3] + 1))
        assert covered == list(range(1, 257)), (n, k)  # contiguous, no overlap, 1-256
        for m in (257, 300, 512, 1024, 4096):
            assert G._select_config(m, n, k) == _any_row(m, n, k), (n, k, m)


def _spread(shape, gen):
    """normal values with per-32 scales spread over +-5 binades"""
    rows, cols = shape
    x = torch.randn(shape, generator=gen, device="cuda")
    e = torch.randint(-5, 6, (rows, cols // 32, 1), generator=gen, device="cuda")
    return x * torch.exp2(e.expand(rows, cols // 32, 32).reshape(rows, cols).float())


_WEIGHTS = {}


def _weights(n, k):
    if (n, k) not in _WEIGHTS:
        gen = torch.Generator(device="cuda").manual_seed(n + k)
        q, s = quantize_mxfp8((_spread((n, k), gen) * 0.02).bfloat16())
        _WEIGHTS[(n, k)] = (q, s, *G.preshuffle_mxfp8_weight(q, s))
    return _WEIGHTS[(n, k)]


def _ran(names, n, k, row):
    """the decode kernel of the row ran: its name carries the row tiles, column tiles, K-waves and K split, and
    ``_fq`` for the in-kernel quantizer (mxfp8_gemm_decode.py)"""
    kernel, tm, tn, kw, sk, _ = row
    tag = f"flydsl_mxfp8_dec_n{n}_k{k}_rt{max(1, tm // 16)}_ni{tn // 16}_kw{kw}_ks{sk}_"
    return any(tag in nm and ("_fq" in nm) == (kernel == "decode_fq") for nm in names)


CASES = []
for _n, _k, _lo, _hi, _row in ROWS:
    for _m in sorted({_lo, _hi, (_lo + _hi) // 2, *([_lo + 5] if _lo + 5 < _hi else [])}):
        CASES.append((_n, _k, _m, _row))


@pytest.mark.parametrize("n,k,m,row", CASES, ids=[f"n{c[0]}_k{c[1]}_m{c[2]}" for c in CASES])
@torch.inference_mode()
def test_fp64_entries_graph(n, k, m, row):
    require_gfx950()
    qw, sw, w, ws = _weights(n, k)
    gen = torch.Generator(device="cuda").manual_seed(m + n)
    x = _spread((m, k), gen).bfloat16()
    if m > 1:
        x[0].zero_()
    out = G.mxfp8_linear(x, w, ws, BF16)
    torch.cuda.synchronize()
    names = kernel_names(lambda: G.mxfp8_linear(x, w, ws, BF16))
    assert _ran(names, n, k, row), (n, k, m, names)
    qx, sx = quantize_mxfp8(x)
    exact = mxfp8_gemm_reference(qx, sx, qw, sw, accumulate=torch.float64).double()
    lo = 1 if m > 1 else 0
    if m > 1:
        assert not out[0].any()
    got = out[lo:].double()
    cos = torch.nn.functional.cosine_similarity(got, exact[lo:], dim=1)
    rel = (got - exact[lo:]).norm(dim=1) / exact[lo:].norm(dim=1)
    assert cos.min() >= 0.99999 and rel.max() <= 1e-3, (n, k, m, float(cos.min()), float(rel.max()))
    for _ in range(3):  # deterministic
        assert torch.equal(G.mxfp8_linear(x, w, ws, BF16).view(torch.int16), out.view(torch.int16))
    # the producer entry: (16, 16) codes in the row's tile, then the plain decode kernel
    tile = G.prequant_code_tile_rows(m, n, k)
    q, s = G.mxfp8_e4m3_quantize_shuffled(x, tile)
    shuffled = G.mxfp8_gemm_shuffled(q, s, w, ws, BF16, m)
    got = shuffled[lo:].double()
    cos = torch.nn.functional.cosine_similarity(got, exact[lo:], dim=1)
    rel = (got - exact[lo:]).norm(dim=1) / exact[lo:].norm(dim=1)
    assert cos.min() >= 0.99999 and rel.max() <= 1e-3, (n, k, m, float(cos.min()), float(rel.max()))
    if row[0] == "decode":  # the same kernel on the same codes: the same bytes
        assert torch.equal(shuffled.view(torch.int16), out.view(torch.int16))
    # graph replay of the bf16 entry: a poisoned output, then a changed input
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


def test_no_compile_after_warmup():
    from flydsl.compiler.jit_function import MlirCompiler

    shapes = sorted({(r[0], r[1]) for r in ROWS})
    for n, k in shapes:
        _, _, w, ws = _weights(n, k)
        G._warmup_weight(w, ws, 256, BF16)
    torch.cuda.synchronize()
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        for n, k in shapes:
            _, _, w, ws = _weights(n, k)
            x = torch.zeros((256, k), device="cuda", dtype=BF16)
            for m in range(1, 257):
                G.mxfp8_linear(x[:m], w, ws, BF16)
                q, s = G.mxfp8_e4m3_quantize_shuffled(x[:m], G.prequant_code_tile_rows(m, n, k))
                G.mxfp8_gemm_shuffled(q, s, w, ws, BF16, m)
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig


@pytest.mark.parametrize("n,k,m", [(r[0], r[1], r[3]) for r in ROWS])
def test_mfma_co_residency(n, k, m):
    from tests.mfma_spinner import mfma_spinner

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
