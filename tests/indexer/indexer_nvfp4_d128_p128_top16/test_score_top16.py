# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode top-16 entry (``score_decode.nvfp4_index_score_top16``) against the two calls
(``nvfp4_index_score_decode`` + ``index_topk_decode_indices``) and ``reference.topk_reference``: ids bit for bit,
with and without a caller's workspace, graph replay at other lengths, determinism. Served calls: one index head x 8
or 4 rows, the host at max_model_len."""

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import (
    BLK,
    decode_kv_lens,
    score_width,
    topk_reference,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import (
    nvfp4_index_score_decode,
    nvfp4_index_score_top16,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import (
    index_topk_decode_indices,
)

from .inputs import block_table, index_cache, queries, require_gfx950

MAX_MODEL_LEN = 1 << 20  # what decode graphs are captured with


def two_calls(iq, cache, bt, seq, init, local, ql, hmax):
    rows = iq.shape[0]
    score = torch.empty(1, rows, score_width(hmax), device="cuda")
    ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    nvfp4_index_score_decode(
        iq, cache, score, bt, seq, init, local, query_len=ql, max_seq_len=hmax
    )
    index_topk_decode_indices(score, ids, seq, max_seq_len=hmax, query_len=ql)
    return score, ids


def entry(iq, cache, bt, seq, init, local, ql, hmax, workspace):
    rows = iq.shape[0]
    ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    score = (
        torch.full((1, rows, score_width(hmax)), float("nan"), device="cuda")
        if workspace
        else None
    )
    nvfp4_index_score_top16(
        iq, cache, bt, seq, init, local, ids, query_len=ql, max_seq_len=hmax, score=score
    )
    return ids


@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize(
    "lens,init,local",
    [
        ([32768], 0, 1),
        ([131072 + 77, 5000], 0, 1),
        ([196608, 1, 0], 0, 1),
        ([1, 2, 3, 300], 0, 1),
        ([65536 + 9] * 4, 2, 3),
        ([300000], 0, 1),  # past one 1664-block window
    ],
)
@pytest.mark.parametrize("workspace", [True, False])
def test_matches_two_calls(ql, lens, init, local, workspace):
    require_gfx950()
    torch.manual_seed(len(lens) * 31 + ql + init)
    bt, pages = block_table(lens)
    cache = index_cache(pages)
    iq = queries(len(lens) * ql, 1)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    score, want = two_calls(iq, cache, bt, seq, init, local, ql, MAX_MODEL_LEN)
    got = entry(iq, cache, bt, seq, init, local, ql, MAX_MODEL_LEN, workspace)
    torch.cuda.synchronize()
    assert torch.equal(got, want)
    kv = torch.cat([decode_kv_lens(L, ql, "cuda") for L in lens])
    nblk = (kv.long() + BLK - 1) // BLK
    assert torch.equal(got, topk_reference(score, nblk, MAX_MODEL_LEN))
    again = entry(iq, cache, bt, seq, init, local, ql, MAX_MODEL_LEN, workspace)
    assert torch.equal(again, got)  # deterministic


@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize("reqs", [1, 2, 3, 4, 5, 8, 16, 17])
def test_graph_replay_other_lengths(ql, reqs):
    require_gfx950()
    torch.manual_seed(5)
    pages = 2600
    cache = index_cache(pages)
    bt = torch.randint(0, pages, (reqs, MAX_MODEL_LEN // BLK), dtype=torch.int32, device="cuda")
    iq = queries(reqs * ql, 1)
    seq = torch.tensor(([40000, 9000, 1, 120000] * ((reqs + 3) // 4))[:reqs], dtype=torch.int32, device="cuda")
    score = torch.empty(1, reqs * ql, score_width(MAX_MODEL_LEN), device="cuda")
    ids = torch.empty(1, reqs * ql, 16, dtype=torch.int32, device="cuda")
    run = lambda: nvfp4_index_score_top16(  # noqa: E731
        iq, cache, bt, seq, 0, 1, ids, query_len=ql, max_seq_len=MAX_MODEL_LEN, score=score
    )
    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for lens in ([100000, 50000, 200000, 1000], [0, 0, 0, 0], [7, 129, 1000, 1], [250000, 0, 3, 212993]):
        lens = (lens * ((reqs + len(lens) - 1) // len(lens)))[:reqs]
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        # The capture also observes new query bits and raw page bytes, including
        # either NaN scale encoding. No input value may be retained across calls.
        iq.view(torch.uint8).copy_(queries(reqs * ql, 1).view(torch.uint8))
        cache.random_(0, 256)
        score.fill_(float("nan"))
        ids.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        _, want = two_calls(iq, cache, bt, seq, 0, 1, ql, MAX_MODEL_LEN)
        assert torch.equal(ids, want), lens


@pytest.mark.parametrize("ql", [4, 8])
@pytest.mark.parametrize("kind", ["random", "equal", "zeros", "negative_zero", "tiny", "tiny_negative", "nan_q", "nan_q_neg", "nan_scale", "nan_scale_neg", "scale_extremes"])
def test_single_request_score_bits(ql, kind):
    """The entry's scorer writes ``nvfp4_index_score_decode``'s bits, including special inputs."""
    require_gfx950()
    torch.manual_seed(31)
    cache = index_cache(16, page_step=2)
    iq = queries(ql * 2, 1)[::2]  # nontrivial query and page strides
    if kind == "equal":
        cache[:] = cache[:1].clone()
    elif kind == "zeros":
        iq.zero_()
    elif kind == "negative_zero":
        iq.view(torch.uint8).fill_(0x80)
    elif kind in ("tiny", "tiny_negative"):
        iq.view(torch.uint8).fill_(0x81 if kind.endswith("negative") else 1)  # smallest e4m3 subnormal
    elif kind in ("nan_q", "nan_q_neg"):
        iq.view(torch.uint8)[0, 0, 5] = 0xff if kind.endswith("neg") else 0x7f
    elif kind in ("nan_scale", "nan_scale_neg"):
        cache[:, 0, 0] = 0  # codes are followed by the planar scale area
        cache.as_strided((16, 9216), (cache.stride(0), 1))[:, 8192 + 5] = 0xff if kind.endswith("neg") else 0x7f
    elif kind == "scale_extremes":
        scales = cache.as_strided((16, 9216), (cache.stride(0), 1))[:, 8192:]
        values = torch.tensor([0, 1, 0x7e, 0x80, 0x81, 0xfe], device="cuda", dtype=torch.uint8)
        scales.copy_(values[torch.arange(1024, device="cuda") % len(values)])
    bt = torch.randint(0, 16, (1, 8192), dtype=torch.int32, device="cuda")
    seq = torch.zeros(1, dtype=torch.int32, device="cuda")
    for length in (0, 1, 7, 8, 127, 128, 129, 32768, 131077, 196607, 212993, 300001, 1048576):
        seq.fill_(length)
        for init, local in ((0, 1), (2, 3)):
            old, want = two_calls(iq, cache, bt, seq, init, local, ql, MAX_MODEL_LEN)
            # A row view checks rebased score stores and an oversized row stride.
            base = torch.full((1, ql + 2, 8192 + 16), float("nan"), device="cuda")
            sc = base[:, 1:ql + 1, :8192]
            ids = torch.full_like(want, -7)
            nvfp4_index_score_top16(iq, cache, bt, seq, init, local, ids,
                                  query_len=ql, max_seq_len=MAX_MODEL_LEN, score=sc)
            nb = (length + 127) // 128
            assert torch.equal(sc[:, :, :nb].view(torch.int32), old[:, :, :nb].view(torch.int32)), (kind, length)
            assert torch.equal(ids, want), (kind, length)
            assert base[:, :1].isnan().all() and base[:, ql + 1:].isnan().all()
            assert sc[:, :, nb:].isnan().all() and base[:, :, 8192:].isnan().all()


@pytest.mark.parametrize("kind", ["real", "ties", "equal", "neginf", "nan", "zeros", "sentinels", "inf", "rising"])
def test_plain_wave4_all_kinds(kind):
    """Every adversarial score kind through the plain-order four-wave selector, item by item."""
    from .test_topk import make_scores
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import host
    require_gfx950()
    # 4097 / 5233: a short second window (its own bound when 16 lanes reach the running key)
    counts = torch.tensor([0, 1, 7, 15, 16, 17, 256, 512, 1024, 1536, 1664, 1665, 1792, 1793, 4097, 5233, 8192])
    rows = len(counts)
    sc = make_scores(kind, 1, rows, 8192, counts, torch.Generator().manual_seed(133)).cuda()
    seq = (counts * 128).int().cuda()
    ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    def run():
        host._run_compiled(host._get_top16_plain_wave4(), sc.data_ptr(), sc.stride(0),
            sc.stride(1), 8192, seq.data_ptr(), 1, ids.data_ptr(), ids.stride(0),
            ids.stride(1), rows, 1, host._current_stream())
    run()
    want = topk_reference(sc, counts.cuda(), MAX_MODEL_LEN)
    assert torch.equal(ids, want)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for new in (counts.flip(0), torch.zeros_like(counts), counts):
        seq.copy_((new * 128).int())
        ids.fill_(-7)
        graph.replay()
        assert torch.equal(ids, topk_reference(sc, new.cuda(), MAX_MODEL_LEN))


@pytest.mark.parametrize("ql", [4, 8])
@pytest.mark.parametrize("reqs", [2, 3, 4, 5, 8, 16])
@pytest.mark.parametrize("special", [False, True])
def test_independent_request_score_bits(ql, reqs, special):
    require_gfx950()
    torch.manual_seed(171)
    cache = index_cache(32)
    iq = queries(ql * reqs, 1)
    if special:
        iq.view(torch.uint8)[0, 0, 5] = 0x7f
        cache.as_strided((32, 9216), (cache.stride(0), 1))[:, 8192 + 11] = 0x7f
    bt = torch.randint(0, 32, (reqs, 8192), device="cuda", dtype=torch.int32)
    # repeated pages within and across requests
    for lengths in ([131077, 1, 196608, 0], [0, 0, 0, 0], [7, 129, 212993, 1048576]):
        lens = (lengths * ((reqs + 3) // 4))[:reqs]
        seq = torch.tensor(lens, device="cuda", dtype=torch.int32)
        for init, local in ((0, 1), (2, 3)):
            old, want = two_calls(iq, cache, bt, seq, init, local, ql, MAX_MODEL_LEN)
            sc = torch.full_like(old, float("nan"))
            ids = torch.full_like(want, -7)
            nvfp4_index_score_top16(iq, cache, bt, seq, init, local, ids,
                                  query_len=ql, max_seq_len=MAX_MODEL_LEN, score=sc)
            assert torch.equal(ids, want)
            for r, length in enumerate(lens):
                nb = (length + 127) // 128
                a = sc[:, r * ql:(r + 1) * ql, :nb].view(torch.int32)
                b = old[:, r * ql:(r + 1) * ql, :nb].view(torch.int32)
                assert torch.equal(a, b), (ql, reqs, r, length)


@pytest.mark.parametrize("reqs", [1, 2, 3, 4])
def test_score_top16_with_mfma_corunner(reqs):
    from tests.mfma_spinner import mfma_spinner
    require_gfx950()
    torch.manual_seed(292)
    ql = 8
    iq = queries(reqs * ql, 1)
    cache = index_cache(32)
    bt = torch.randint(0, 32, (reqs, 8192), device="cuda", dtype=torch.int32)
    seq = torch.tensor([131077, 7, 0, 300001][:reqs], device="cuda", dtype=torch.int32)
    old, want = two_calls(iq, cache, bt, seq, 2, 3, ql, MAX_MODEL_LEN)
    sc = torch.full_like(old, float("nan"))
    ids = torch.full_like(want, -7)
    def run():
        nvfp4_index_score_top16(iq, cache, bt, seq, 2, 3, ids,
            query_len=ql, max_seq_len=MAX_MODEL_LEN, score=sc)
    run()
    spin = mfma_spinner(256)
    high = torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(high):
        spin(100000, high)
    for _ in range(20):
        run()
    torch.cuda.synchronize()
    assert torch.equal(ids, want)
    for r, length in enumerate(seq.cpu().tolist()):
        nb = (length + 127) // 128
        assert torch.equal(sc[:, r * ql:(r + 1) * ql, :nb].view(torch.int32),
                           old[:, r * ql:(r + 1) * ql, :nb].view(torch.int32))


@pytest.mark.parametrize("reqs", [1, 2, 3, 4, 5, 17])
def test_entry_graph_kernel_count(reqs):
    """Two intended kernels in the capture, including the ordinary fallback."""
    from torch.profiler import ProfilerActivity, profile
    require_gfx950()
    ql = 8
    iq = queries(reqs * ql, 1)
    cache = torch.zeros(2, 128, 72, dtype=torch.uint8, device="cuda")
    bt = torch.zeros(reqs, 8192, dtype=torch.int32, device="cuda")
    seq = torch.full((reqs,), ql, dtype=torch.int32, device="cuda")
    sc = torch.empty(1, reqs * ql, 8192, device="cuda")
    ids = torch.empty(1, reqs * ql, 16, dtype=torch.int32, device="cuda")
    def run():
        nvfp4_index_score_top16(iq, cache, bt, seq, 0, 1, ids,
            query_len=ql, max_seq_len=MAX_MODEL_LEN, score=sc)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    kernels = [e for e in prof.key_averages() if "index_score" in e.key or "index_top16" in e.key]
    assert len(kernels) == 2 and all(e.count == 1 for e in kernels), [(e.key, e.count) for e in kernels]
    scorer = next(e.key for e in kernels if "index_score" in e.key)
    assert ("index_score_page_d128" in scorer) == (reqs in (1, 3))
    assert ("firstpage" in scorer) == (reqs == 2)
    assert ("_xcd" in scorer) == (reqs == 2)  # 8 rows: the XCD-local block order
    assert ("index_score_sector8" in scorer) == (reqs == 4)
    if reqs == 5:
        assert "index_score_d128" in scorer
    selector = next(e.key for e in kernels if "index_top16" in e.key)
    assert ("plain_wave4" in selector) == (reqs <= 16)


@pytest.mark.parametrize("candidates", [15, 16, 17, 31, 32, 33, 63, 64, 65])
def test_plain_wave4_candidate_rank_boundary(candidates):
    """Both rank widths and the exact overflow path preserve tied-id order."""
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import host
    require_gfx950()
    sc = torch.zeros((1, 1, 8192), device="cuda")
    sc[0, 0, :candidates] = 1.0
    seq = torch.tensor([1024 * 128], dtype=torch.int32, device="cuda")
    ids = torch.full((1, 1, 16), -7, dtype=torch.int32, device="cuda")
    host._run_compiled(host._get_top16_plain_wave4(), sc.data_ptr(), sc.stride(0),
        sc.stride(1), 8192, seq.data_ptr(), 1, ids.data_ptr(), ids.stride(0),
        ids.stride(1), 1, 1, host._current_stream())
    assert torch.equal(ids, topk_reference(sc, torch.tensor([1024], device="cuda"), MAX_MODEL_LEN))


def test_plain_wave4_seed_boundaries():
    """Ordered-key bucket boundaries, a failed seed, signed values and wraparound."""
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import host
    require_gfx950()
    centers = [0x00000000, 0x04000000, 0x3c000000, 0x40000000, 0x44000000,
               0x7c000000, 0x80000000, 0xbc000000, 0xc0000000, 0xc4000000,
               0xfc000000, 0xff800000]
    bits = torch.tensor(centers, dtype=torch.int64)[:, None]
    # Values straddle each boundary. Wrapping around signed zero and -inf
    # also produces signed NaN payloads for the exact fallback.
    offsets = torch.arange(8192, dtype=torch.int64) - 4096
    raw = ((bits + offsets) & 0xffffffff).to(torch.int32)
    extra = torch.arange(8192, dtype=torch.float32)[None].repeat(3, 1) / 8192 + 1
    extra[0, 31] = 1e29  # lower endpoint has fewer than 16 maxima
    extra[1, 31::64] = -100  # upper endpoint has at least 16 maxima
    extra[2, 31] = 1e38  # biased upper endpoint wraps to zero
    sc = torch.cat([raw.view(torch.float32), extra], dim=0)[None].cuda()
    rows = len(centers) + 3
    counts = torch.full((rows,), 1024, dtype=torch.int32, device="cuda")
    seq = counts * 128
    ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    def run():
        host._run_compiled(host._get_top16_plain_wave4(), sc.data_ptr(), sc.stride(0),
            sc.stride(1), 8192, seq.data_ptr(), 1, ids.data_ptr(), ids.stride(0),
            ids.stride(1), rows, 1, host._current_stream())
    run()
    assert torch.equal(ids, topk_reference(sc, counts, MAX_MODEL_LEN))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for new in (1792, 1793, 8192, 0, 1024):
        counts.fill_(new)
        seq.copy_(counts * 128)
        ids.fill_(-7)
        graph.replay()
        assert torch.equal(ids, topk_reference(sc, counts, MAX_MODEL_LEN))
