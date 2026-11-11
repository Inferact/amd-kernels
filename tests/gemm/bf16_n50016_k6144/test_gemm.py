# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""amd_kernels.gemm.bf16_n50016_k6144 against its fp64 reference, both weight layouts."""
import pytest
import torch

from amd_kernels.gemm.bf16_n50016_k6144 import (bf16_gemm_n50016_k6144, shuffle_weight, unshuffle_weight,
                                                 warmup_bf16_gemm_n50016_k6144)
from amd_kernels.gemm.bf16_n50016_k6144 import host
from amd_kernels.gemm.bf16_n50016_k6144.reference import K, N, bf16_gemm_n50016_k6144_reference

MS = [1, 2, 7, 15, 16, 17, 33, 64, 70, 80, 112, 127, 128, 129, 160, 161, 200, 224, 256, 257, 300, 513]


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    return {False: w, True: shuffle_weight(w)}


def check_numeric(y, ref):
    yf, rf = y.double(), ref.double()
    cos = torch.nn.functional.cosine_similarity(yf, rf, dim=1)
    l2 = (yf - rf).norm(dim=1) / rf.norm(dim=1)
    assert bool(torch.all(cos >= 0.99999)), cos.min().item()
    assert bool(torch.all(l2 <= 1e-3)), l2.max().item()


def test_shuffle_is_a_permutation(weights):
    w, ws = weights[False], weights[True]
    assert torch.equal(unshuffle_weight(ws).view(torch.int16), w.view(torch.int16))
    flat = ws.view(-1)
    for n, k in [(0, 0), (15, 63), (16, 64), (50015, 6143), (12345, 4321)]:
        idx = (((n // 16) * (K // 64) + k // 64) * 16 + n % 16) * 64 + k % 64
        assert flat[idx].view(torch.int16) == w[n, k].view(torch.int16)


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("m", MS)
def test_reference(weights, shuffled, m):
    w = weights[shuffled]
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    y = torch.full((m, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    bf16_gemm_n50016_k6144(x, w, y, shuffled=shuffled)
    check_numeric(y, bf16_gemm_n50016_k6144_reference(x, weights[False]))
    old = y.clone()
    for _ in range(2):
        y.fill_(float("nan"))
        bf16_gemm_n50016_k6144(x, w, y, shuffled=shuffled)
        assert torch.equal(y.view(torch.int16), old.view(torch.int16))


def test_layouts_agree(weights):
    x = torch.randn(80, K, device="cuda").to(torch.bfloat16)
    a = bf16_gemm_n50016_k6144(x, weights[False], shuffled=False)
    b = bf16_gemm_n50016_k6144(x, weights[True], shuffled=True)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_zero_rows(weights):
    x = torch.empty(0, K, device="cuda", dtype=torch.bfloat16)
    assert bf16_gemm_n50016_k6144(x, weights[False]).shape == (0, N)


@pytest.mark.parametrize("shuffled", [False, True])
def test_nonfinite_propagation(weights, shuffled):
    w = weights[False].clone()
    w[0, 2] = float("nan")
    w[50015, 8] = float("inf")
    w[16].zero_()
    x = torch.randn(3, K, device="cuda").to(torch.bfloat16)
    x[0, 1] = float("nan")
    x[1, 3] = float("inf")
    x[2].zero_()
    y = bf16_gemm_n50016_k6144(x, shuffle_weight(w) if shuffled else w, shuffled=shuffled)
    ref = bf16_gemm_n50016_k6144_reference(x, w)
    assert torch.equal(torch.isnan(y), torch.isnan(ref))
    assert torch.equal(torch.isposinf(y), torch.isposinf(ref))
    assert torch.equal(torch.isneginf(y), torch.isneginf(ref))
    fin = torch.isfinite(ref[2])  # a zero row stays 0 except at the NaN / inf weight columns (0 x inf = NaN)
    assert bool(torch.all(y[2][fin] == 0)) and int((~fin).sum()) == 2


def warm_up_one_mode(monkeypatch, shuffled, mode):
    """Warm up one call mode; check it asks for exactly that mode's tile variants (one per WARMUP_ROWS entry), no
    variant of another mode (each mode is its own kernel: the caller warms the modes it runs)."""
    host._kernel.cache_clear()
    keys, real = set(), host._kernel

    def spy(*key):
        keys.add(key + (False,) * (5 - len(key)))  # (mt, shuffled, halves, large, publish)
        return real(*key)

    monkeypatch.setattr(host, "_kernel", spy)
    warmup_bf16_gemm_n50016_k6144(torch.device("cuda"), shuffled, mode)
    monkeypatch.setattr(host, "_kernel", real)
    assert len(keys) == len(host.WARMUP_ROWS) == 14, sorted(keys)
    assert {k[4] for k in keys} == {mode == "publish"}, sorted(keys)


@pytest.mark.parametrize("shuffled", [False, True])
def test_graph_no_compile_after_warmup(weights, shuffled, monkeypatch):
    from flydsl.compiler.jit_function import MlirCompiler
    warm_up_one_mode(monkeypatch, shuffled, "contiguous")
    compiled = []
    original = MlirCompiler.compile

    def track(*args, **kwargs):
        compiled.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(MlirCompiler, "compile", track)
    w = weights[shuffled]
    for m in (1, 17, 80, 128, 129, 161, 192, 193, 224, 256, 300, 513):
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        y = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bf16_gemm_n50016_k6144(x, w, y, shuffled=shuffled)
        for _ in range(2):
            x.normal_()
            y.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check_numeric(y, bf16_gemm_n50016_k6144_reference(x, weights[False]))
    assert compiled == []




@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("m", [1, 7, 17, 70, 80, 112, 128, 129, 161, 224, 256, 257, 448, 512, 896, 1024, 1792, 2048])
def test_publish_rows(weights, shuffled, m):
    """One GPU: the row-indexed publish kernels into a private slot pair equal the contiguous GEMM of x[row_indices]
    bit for bit (duplicates, the first / last source rows), and the last workgroup publishes slot 0."""
    x = torch.randn(m + 19, K, device="cuda").to(torch.bfloat16)
    indices = torch.randint(0, len(x), (m,), device="cuda", dtype=torch.int64)
    indices[0] = len(x) - 1
    if m > 1:
        indices[1] = 0
    slot_bytes = m * N * 2
    pool = torch.full((2 * slot_bytes,), 0xFF, dtype=torch.uint8, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(16, dtype=torch.int32, device="cuda")
    publish = (pool.data_ptr(), slot_bytes, counter, flag.data_ptr())
    assert bf16_gemm_n50016_k6144(x, weights[shuffled], shuffled=shuffled, row_indices=indices, publish=publish) is None
    expected = bf16_gemm_n50016_k6144(x[indices], weights[shuffled], shuffled=shuffled)
    torch.cuda.synchronize()
    y = pool[:slot_bytes].view(torch.bfloat16).view(m, N)
    assert torch.equal(y.view(torch.int16), expected.view(torch.int16))
    check_numeric(y, bf16_gemm_n50016_k6144_reference(x, weights[False], indices))
    assert int(counter[0]) == 1 and int(flag[0]) == 1 and int(flag[1]) == 0


@pytest.mark.parametrize("shuffled", [False, True])
def test_publish_warm_up(shuffled, monkeypatch):
    warm_up_one_mode(monkeypatch, shuffled, "publish")


def test_supports(weights):
    from amd_kernels.gemm.bf16_n50016_k6144 import supports

    w = weights[False]
    assert supports(w) and supports(weights[True])
    assert not supports(w[:, :K - 64])
    assert not supports(w.T)
    assert not supports(torch.empty(N, K, dtype=torch.float16))
    assert not supports(torch.empty(N - 16, K, dtype=torch.bfloat16))
    assert not supports(w, torch.zeros(N, dtype=torch.bfloat16, device=w.device))
