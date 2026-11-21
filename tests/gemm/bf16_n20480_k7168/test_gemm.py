# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""amd_kernels.gemm.bf16_n20480_k7168 against its fp64 reference: every band and tile of the row-count dispatch, the row
map and publish modes on one GPU (the eight-rank exchange is test_push.py)."""
import pytest
import torch

from amd_kernels.gemm.bf16_n20480_k7168 import bf16_gemm_n20480_k7168, supports
from amd_kernels.gemm.bf16_n20480_k7168 import host
from amd_kernels.gemm.bf16_n20480_k7168.reference import K, N, bf16_gemm_n20480_k7168_reference

MS = [1, 2, 7, 12, 15, 16, 17, 24, 32, 33, 40, 48, 49, 64, 65, 70, 80, 96, 97, 112, 127, 128, 129, 160, 161, 192, 200,
      224, 256]


@pytest.fixture(scope="module")
def weight():
    g = torch.Generator(device="cuda").manual_seed(0)
    return (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)


def check_numeric(y, ref):
    yf, rf = y.double(), ref.double()
    cos = torch.nn.functional.cosine_similarity(yf, rf, dim=1)
    l2 = (yf - rf).norm(dim=1) / rf.norm(dim=1)
    assert bool(torch.all(cos >= 0.99999)), cos.min().item()
    assert bool(torch.all(l2 <= 1e-3)), l2.max().item()


@pytest.mark.parametrize("m", MS)
def test_reference(weight, m):
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    y = torch.full((m, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    bf16_gemm_n20480_k7168(x, weight, y)
    check_numeric(y, bf16_gemm_n20480_k7168_reference(x, weight))
    old = y.clone()
    for _ in range(2):
        y.fill_(float("nan"))
        bf16_gemm_n20480_k7168(x, weight, y)
        assert torch.equal(y.view(torch.int16), old.view(torch.int16))


def test_rows_do_not_depend_on_the_batch_within_a_band(weight):
    """An output row's bits do not depend on the row count of the call while the call stays in one band (each band
    has its own K start per workgroup, so bands may differ in the last bit)."""
    x = torch.randn(256, K, device="cuda").to(torch.bfloat16)
    for top, ms in ((16, (1, 5, 15)), (48, (33, 40, 47)), (80, (65, 70, 79)), (96, (81, 90, 95)), (128, (97, 112, 127)), (160, (129, 145, 159)), (192, (161, 180, 191)), (256, (193, 224, 255))):
        full = bf16_gemm_n20480_k7168(x[:top].contiguous(), weight)
        for m in ms:
            part = bf16_gemm_n20480_k7168(x[:m].contiguous(), weight)
            assert host.band_of(m) == host.band_of(top)
            assert torch.equal(part.view(torch.int16), full[:m].view(torch.int16)), (top, m)


@pytest.mark.parametrize("m", [1, 17, 80, 96, 97, 128, 150, 192, 256])
def test_gather_rows_equal_the_contiguous_gemm_of_the_gathered_rows(weight, m):
    x = torch.randn(m + 11, K, device="cuda").to(torch.bfloat16)
    indices = torch.randint(0, len(x), (m,), device="cuda", dtype=torch.int64)
    indices[0] = len(x) - 1
    y = bf16_gemm_n20480_k7168(x, weight, row_indices=indices)
    assert y.shape == (m, N)
    expected = bf16_gemm_n20480_k7168(x[indices].contiguous(), weight)
    assert torch.equal(y.view(torch.int16), expected.view(torch.int16))
    check_numeric(y, bf16_gemm_n20480_k7168_reference(x, weight, indices))


def test_splitk_rows_do_not_depend_on_their_position(weight):
    """The split-K sum runs in member order for every row: a row's bits are the same wherever it sits in the tile."""
    x = torch.randn(256, K, device="cuda").to(torch.bfloat16)
    row = x[5].clone()
    for m in (100, 150, 250):
        xs = x[:m].clone()
        positions = [0, 1, 15, 16, 17, 31, 32, m - 1]
        xs[positions] = row
        y = bf16_gemm_n20480_k7168(xs, weight)
        for p in positions[1:]:
            assert torch.equal(y[p].view(torch.int16), y[0].view(torch.int16)), (m, p)


def test_splitk_exchange_flags_are_rearmed_between_variants(weight):
    """Back-to-back launches of every split-K band over one workspace: every result is right, no launch hangs."""
    flags = host._exchange_workspace(weight.device)[1]
    for i in range(30):
        m = (100, 128, 129, 150, 192, 193, 256)[i % 7]
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        y = bf16_gemm_n20480_k7168(x, weight)
        if i % 5 == 0:
            check_numeric(y, bf16_gemm_n20480_k7168_reference(x, weight))
        torch.cuda.synchronize()
        assert not bool(flags.any())


@pytest.mark.parametrize("m", [3, 130, 200])
def test_nonfinite_rows_stay_in_their_rows(weight, m):
    w = weight.clone()
    w[7, 100] = float("nan")
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    x[m - 1, 5000] = float("inf")
    y = bf16_gemm_n20480_k7168(x, w)
    ref = bf16_gemm_n20480_k7168_reference(x, w)
    assert torch.equal(torch.isnan(y), torch.isnan(ref))
    assert torch.equal(torch.isinf(y), torch.isinf(ref))


def test_nonfinite_propagation(weight):
    w = weight.clone()
    w[0, 2] = float("nan")
    w[20479, 8] = float("inf")
    w[16].zero_()
    x = torch.randn(3, K, device="cuda").to(torch.bfloat16)
    x[0, 1] = float("nan")
    x[1, 3] = float("inf")
    x[2].zero_()
    y = bf16_gemm_n20480_k7168(x, w)
    ref = bf16_gemm_n20480_k7168_reference(x, w)
    assert torch.equal(torch.isnan(y), torch.isnan(ref))
    assert torch.equal(torch.isposinf(y), torch.isposinf(ref))
    assert torch.equal(torch.isneginf(y), torch.isneginf(ref))
    fin = torch.isfinite(ref[2])  # a zero row stays 0 except at the NaN / inf weight columns (0 x inf = NaN)
    assert bool(torch.all(y[2][fin] == 0)) and int((~fin).sum()) == 2


def warm_up(w):
    x = torch.zeros(host.MAX_M, K, device="cuda", dtype=torch.bfloat16)
    for m in host.WARMUP_ROWS:
        host._bf16_gemm(x[:m], w)
    torch.cuda.synchronize()


def test_graph_no_compile_after_warmup(weight, monkeypatch):
    from flydsl.compiler.jit_function import MlirCompiler
    warm_up(weight)
    compiled = []
    original = MlirCompiler.compile

    def track(*args, **kwargs):
        compiled.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(MlirCompiler, "compile", track)
    for m in range(1, host.MAX_M + 1, 5):
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        y = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bf16_gemm_n20480_k7168(x, weight, y)
        for _ in range(2):
            x.normal_()
            y.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check_numeric(y, bf16_gemm_n20480_k7168_reference(x, weight))
    assert compiled == []


def push_state(device, rows_cap=256):
    recvs = [torch.full((rows_cap * 8 * N,), float("nan"), dtype=torch.bfloat16, device=device) for _ in range(8)]
    flags = [torch.zeros(128, dtype=torch.int32, device=device) for _ in range(8)]
    peers = torch.tensor([r.data_ptr() for r in recvs] * 2 + [f.data_ptr() for f in flags], dtype=torch.int64,
                         device=device)
    return recvs, flags, peers, torch.zeros(1, dtype=torch.int32, device=device)


@pytest.mark.parametrize("mode", ["owner", "all"])
def test_push_graph_no_compile_after_warmup(weight, mode, monkeypatch):
    from flydsl.compiler.jit_function import MlirCompiler
    recvs, flags, peers, counter = push_state("cuda")
    x = torch.randn(256, K, device="cuda").to(torch.bfloat16)

    def call(m):
        counts = torch.zeros(8, dtype=torch.int32, device="cuda")
        counts[3] = m
        push = ((peers, counter, flags[3].data_ptr(), 3, 0), counts if mode == "owner" else None)
        host._bf16_gemm(x, weight, row_indices=torch.arange(m, device="cuda"), push=push)

    for m in host.WARMUP_ROWS:
        call(m)
    torch.cuda.synchronize()
    compiled = []
    original = MlirCompiler.compile
    monkeypatch.setattr(MlirCompiler, "compile", lambda *a, **k: compiled.append(1) or original(*a, **k))
    for m in range(1, host.MAX_M + 1, 7):
        indices = torch.arange(m, device="cuda")
        counts = torch.zeros(8, dtype=torch.int32, device="cuda")
        counts[3] = m
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            host._bf16_gemm(x, weight, row_indices=indices,
                            push=((peers, counter, flags[3].data_ptr(), 3, 0), counts if mode == "owner" else None))
        graph.replay()
        torch.cuda.synchronize()
    assert compiled == []


@pytest.mark.parametrize("mode", ["owner", "all"])
@pytest.mark.parametrize("m", [1, 7, 17, 40, 65, 80, 112, 128, 129, 161, 192, 256])
def test_push_rows(weight, m, mode):
    """One GPU as sender rank 3 of eight: the pushed rows equal the contiguous GEMM of x[row_indices] bit for bit, land
    in the owner's buffer (or every buffer) at this rank's columns, nothing else is written, and the last workgroup
    stores the sequence into word 96 + 3 of every flag block."""
    x = torch.randn(m + 19, K, device="cuda").to(torch.bfloat16)
    indices = torch.randint(0, len(x), (m,), device="cuda", dtype=torch.int64)
    indices[0] = len(x) - 1
    if m > 1:
        indices[1] = 0
    counts = torch.zeros(8, dtype=torch.int32)
    owners = torch.sort(torch.randint(0, 8, (m,))).values
    counts += torch.bincount(owners, minlength=8).int()
    recvs, flags, peers, counter = push_state("cuda")
    rank = 3
    push = ((peers, counter, flags[rank].data_ptr(), rank, 1), counts.cuda() if mode == "owner" else None)
    assert bf16_gemm_n20480_k7168(x, weight, row_indices=indices, push=push) is None
    expected = bf16_gemm_n20480_k7168(x[indices], weight)
    torch.cuda.synchronize()
    check_numeric(expected, bf16_gemm_n20480_k7168_reference(x, weight, indices))
    start = 0
    for d in range(8):
        buf = recvs[d].view(-1, 8 * N)
        own = torch.arange(m) if mode == "all" else torch.arange(start, start + int(counts[d])) 
        got = buf[: len(own), rank * N:(rank + 1) * N]
        if mode == "all" or int(counts[d]):
            assert torch.equal(got.view(torch.int16), expected[own].view(torch.int16)), (d, mode)
        rest = torch.ones_like(buf, dtype=torch.bool)
        rest[: len(own) if (mode == "all" or int(counts[d])) else 0, rank * N:(rank + 1) * N] = False
        assert bool(torch.isnan(buf[rest]).all()), d
        start += int(counts[d])
        assert int(flags[d][96 + rank]) == 1
    assert int(counter[0]) == 1 and int(flags[rank][1]) == 0 and not bool(flags[rank][16:72].any())


def test_supports(weight):
    assert supports(weight)
    assert not supports(weight[:, :K - 64])
    assert not supports(weight.T)
    assert not supports(torch.empty(N, K, dtype=torch.float16))
    assert not supports(torch.empty(N - 16, K, dtype=torch.bfloat16))
    assert not supports(weight, torch.zeros(N, dtype=torch.bfloat16, device=weight.device))
