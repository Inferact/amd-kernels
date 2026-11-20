# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode indexer split over two GPUs (``split2.host.IndexSplit2``, TP2): each rank's two heads' top-16 ids bit for
bit the one-GPU ``nvfp4_index_score_top16`` (itself ``reference.topk_reference``), over short / long / mixed contexts,
the init / local sentinels, saturated keys, a NaN query, shared prefixes, repeated calls, graph replay with changed
lengths and the device call counts; ``IndexDecodeTop16`` at two ranks (route, warm-up, no compile after it)."""
import torch

from ...multi_gpu import run_on_2_gpus
from .inputs import block_table, index_cache, require_gfx950

MAX = 1 << 20


def _case(lens, ql, rank, *, saturate=False, nan_q=False, share=False, seed=0):
    torch.manual_seed(seed)
    table, pages = block_table(lens, share_prefixes=share)
    cache = index_cache(pages, saturate=saturate)
    wide = torch.zeros(len(lens), MAX // 128, dtype=torch.int32, device="cuda")
    wide[:, : table.shape[1]] = table
    rows = len(lens) * ql
    q4 = (torch.randn(rows, 4, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    if nan_q:
        q4.view(torch.uint8)[rows // 2, :, 5] = 0x7F
    iq = q4[:, 2 * rank : 2 * rank + 2].contiguous()
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return iq, cache, wide, seq


def _one_gpu(iq, cache, table, seq, init, local, ql):
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import nvfp4_index_score_top16

    want = torch.full((2, iq.shape[0], 16), -7, dtype=torch.int32, device="cuda")
    nvfp4_index_score_top16(iq, cache, table, seq, init, local, want, query_len=ql, max_seq_len=MAX)
    return want


CASES = [
    # (lengths, init, local, options)
    ([1], 2, 3, {}),
    ([129, 3, 2047], 2, 3, {}),
    ([5000], 0, 1, {}),
    ([196608 - 977 * r for r in range(5)], 2, 3, {}),
    ([100000 - 3001 * r for r in range(17)], 0, 1, {"saturate": True}),
    ([65536 + 129 * r for r in range(64)], 2, 3, {}),
    ([40000 - 1111 * r for r in range(12)], 2, 3, {"nan_q": True}),
    ([30000 - 977 * r for r in range(9)], 2, 3, {"share": True}),
    ([1048576, 1, 300001], 1, 1, {}),
]


def _ids_body(rank, errors):
    import torch.distributed as dist
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split2 import host
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split2.host import IndexSplit2

    dev = torch.device("cuda", rank)
    # the served bound, then runs and page ranges forced on every case
    for ql, pin in ((8, None), (4, None), (8, 0), (8, 1 << 20)):
        host.PIN_PAGES_RUNS = pin
        cp = IndexSplit2(rank, dist.group.WORLD, dev, 64 * ql, score_width(MAX), ql)
        calls = 0
        for i, (lens, init, local, opt) in enumerate(CASES):
            iq, cache, table, seq = _case(lens, ql, rank, seed=100 * ql + i, **opt)
            want = _one_gpu(iq, cache, table, seq, init, local, ql)
            for it in range(2):
                got = torch.full_like(want, -9)
                cp.score_topk(iq, cache, table, seq, init, local, got, MAX)
                calls += 1
                torch.cuda.synchronize()
                if not torch.equal(got, want):
                    bad = (got != want).any(-1)
                    errors.append(f"rank {rank} ql {ql} pin {pin} case {i} call {it}: {int(bad.sum())} rows differ, "
                                  f"first {bad.nonzero()[:2].tolist()}")
        counts = cp.counts[:2].tolist()
        if counts != [calls, calls]:
            errors.append(f"rank {rank} ql {ql}: device counts {counts}, {calls} calls")
    host.PIN_PAGES_RUNS = None


def _replay_body(rank, errors):
    import torch.distributed as dist
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split2.host import IndexSplit2

    dev = torch.device("cuda", rank)
    ql, n = 8, 24
    cp = IndexSplit2(rank, dist.group.WORLD, dev, 32 * ql, score_width(MAX), ql)
    iq, cache, table, seq = _case([150000 - 2999 * r for r in range(n)], ql, rank, seed=7)
    got = torch.full((2, n * ql, 16), -9, dtype=torch.int32, device="cuda")
    cp.score_topk(iq, cache, table, seq, 2, 3, got, MAX)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            cp.score_topk(iq, cache, table, seq, 2, 3, got, MAX)
    for pattern in ([32768, 65543, 150001, 1, 0], [0], [129, 3, 1, 0, 140000]):
        lens = (pattern * ((n + len(pattern) - 1) // len(pattern)))[:n]
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        got.fill_(-9)
        graph.replay()
        torch.cuda.synchronize()
        want = _one_gpu(iq, cache, table, seq, 2, 3, ql)
        if not torch.equal(got, want):
            errors.append(f"rank {rank} replay {pattern}: {int((got != want).any(-1).sum())} rows differ")
    if cp.counts[:2].tolist() != [10, 10]:
        errors.append(f"rank {rank}: device counts {cp.counts[:2].tolist()} after 1 + 3 x 3 calls")


def _route_body(rank, errors):
    import torch.distributed as dist
    from amd_kernels import warm_up
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import decode_top16
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.decode_top16 import IndexDecodeTop16

    dev = torch.device("cuda", rank)
    warm_up.set_auto(True)
    try:
        for ql in (8, 4):
            index = IndexDecodeTop16(dev, query_len=ql, max_reqs=64, max_seq_len=MAX, num_index_heads=2, rank=rank,
                                     group=dist.group.WORLD)
            if index.split2 is None or index.split is not None:
                errors.append(f"rank {rank} ql {ql}: split2 {index.split2} split4 {index.split} ({index.split_off})")
                continue
            m = decode_top16.SPLIT2_MIN_REQS[ql]
            routes = [index.route(m - 1, ql, 2), index.route(m, ql, 2), index.route(m, ql, 2, split=False),
                      index.route(64, ql, 2), index.route(65, ql, 2)]
            if routes != ["one_gpu", "split2", "one_gpu", "split2", "one_gpu"]:
                errors.append(f"rank {rank} ql {ql}: routes {routes}")
            for n in (m, 33, 64):
                lens = [90000 - 1301 * r for r in range(n)]
                iq, cache, table, seq = _case(lens, ql, rank, seed=n)
                got = torch.full((2, n * ql, 16), -9, dtype=torch.int32, device="cuda")
                index(iq, cache, table, seq, 2, 3, got, query_len=ql, max_seq_len=MAX)
                torch.cuda.synchronize()
                want = _one_gpu(iq, cache, table, seq, 2, 3, ql)
                if not torch.equal(got, want):
                    errors.append(f"rank {rank} ql {ql} n {n}: route ids differ")
        if warm_up.late_compiles() != 0:
            errors.append(f"rank {rank}: {warm_up.late_compiles()} compiles after the warm-up")
    finally:
        warm_up.set_auto(False)


def test_split2_ids_two_gpus():
    require_gfx950()
    run_on_2_gpus(_ids_body)


def test_split2_graph_replay_two_gpus():
    require_gfx950()
    run_on_2_gpus(_replay_body)


def test_index_decode_top16_two_ranks():
    require_gfx950()
    run_on_2_gpus(_route_body)
