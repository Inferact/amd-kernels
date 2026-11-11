# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The prefill index scorer (``score_prefill``) against ``reference.py``: both kernels and their tiles, graph
replay, determinism. Served calls (fp8 keys): 1 index head, ragged rows of up to 32768 new tokens over
contexts up to 1M; small work takes ``kernel_shared`` (128-row tiles), larger work ``kernel.py`` (512-row tiles).
"""

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import score_prefill as fly
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import (
    BLK,
    EXACT_RTOL,
    SCORE_ATOL,
    SCORE_RTOL,
    cdiv,
    prefill_scores_reference,
    request_keys,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill import host

from .inputs import assert_scores, block_table, index_cache, nan_like, queries, require_gfx950

INIT, LOCAL = 0, 1  # the model's sparse_init_block / sparse_local_block


def record_kernels(monkeypatch) -> list[str]:
    names: list[str] = []
    get_kernel, get_shared = host._get_kernel, host._get_kernel_shared

    def record_kernel(heads, staged=True):
        names.append("tile512" + ("" if staged else "_g4"))
        return get_kernel(heads, staged)

    def record_shared(heads, mq, fp8):
        names.append(f"shared_mq{mq}" + ("" if fp8 else "_f16mfma"))
        return get_shared(heads, mq, fp8)

    monkeypatch.setattr(host, "_get_kernel", record_kernel)
    monkeypatch.setattr(host, "_get_kernel_shared", record_shared)
    return names


def make_case(q_lens, seq_lens, heads, page_step=1):
    table, pages = block_table(seq_lens)  # pages out of order
    cache = index_cache(pages, page_step)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(q_lens, dtype=torch.int32, device="cuda").cumsum(0)
    q = queries(sum(q_lens), heads)
    return q, cache, table, cu, seq


def score(q, cache, table, cu, seq, q_lens, seq_lens, width, exact=False, init=INIT, local=LOCAL):
    out = nan_like(q.shape[1], q.shape[0], width)
    assert fly.supports_nvfp4_index_score_prefill(q, out)
    fly.nvfp4_index_score_prefill(
        q, cache, out, table, cu, seq, init, local, max(q_lens), max(seq_lens), exact=exact
    )
    return out


def check(out, q, cache, table, q_lens, seq_lens, exact=False, init=INIT, local=LOCAL):
    """Every row's blocks up to its own against the reference (rows in slices)."""
    stats = []
    q0 = 0
    for req, (ql, s) in enumerate(zip(q_lens, seq_lens)):
        nb = cdiv(s, BLK)
        keys = request_keys(cache, table, req, nb, exact)
        pos = s - ql + torch.arange(ql, device="cuda")
        for r0 in range(0, ql, 512):
            r1 = min(ql, r0 + 512)
            want, seen = prefill_scores_reference(
                q[q0 + r0 : q0 + r1], keys, pos[r0:r1], init, local
            )
            seen = seen[None].expand(q.shape[1], -1, -1)
            got = out[:, q0 + r0 : q0 + r1, :nb]
            stats.append(
                assert_scores(
                    got[seen],
                    want[seen],
                    EXACT_RTOL if exact else SCORE_RTOL,
                    SCORE_ATOL,
                    (ql, s),
                )
            )
        q0 += ql
    rel = max(x[0] for x in stats)
    l2 = max(x[1] for x in stats)
    cos = min(x[2] for x in stats)
    print(f"max rel {rel:.2e}, max rel L2 {l2:.2e}, min cos {cos:.9f}")


def seen_mask(q_lens, seq_lens, heads, width):
    rows = []
    for ql, s in zip(q_lens, seq_lens):
        pos = s - ql + torch.arange(ql, device="cuda")
        rows.append(torch.arange(width, device="cuda")[None, :] <= (pos // BLK)[:, None])
    return torch.cat(rows)[None].expand(heads, -1, -1)


SERVED = [
    # (q_lens, seq_lens, the kernel the host picks)
    ([64], [65536], "shared_mq1"),
    ([4096], [4096], "shared_mq1"),
    ([2000, 513, 3000], [9000, 513, 20000], "tile512"),
    ([700, 1500], [30000, 1500], "shared_mq1"),
    ([8192], [131072], "tile512"),
    ([3072, 3072, 3072], [203072, 3072, 207872], "tile512"),
]


@pytest.mark.parametrize(
    "q_lens,seq_lens,kernel", SERVED, ids=[f"{c[0]}-{c[1]}".replace(" ", "") for c in SERVED]
)
@torch.inference_mode()
def test_prefill_served(q_lens, seq_lens, kernel, monkeypatch):
    """The served call: the kernel the host picks, and the scores against the reference."""
    require_gfx950()
    heads = 1
    names = record_kernels(monkeypatch)
    torch.manual_seed(sum(q_lens) + heads)
    q, cache, table, cu, seq = make_case(q_lens, seq_lens, heads)
    width = cdiv(max(seq_lens), BLK) + 3
    out = score(q, cache, table, cu, seq, q_lens, seq_lens, width)
    torch.cuda.synchronize()
    assert names == [kernel], names
    check(out, q, cache, table, q_lens, seq_lens)


SHAPES = [
    ([4096], [4096]),  # fresh prompt: every tile diagonal
    ([2000, 513, 3000], [9000, 513, 20000]),  # ragged, unaligned prefixes
    ([700, 1500], [30000, 1500]),  # long cached prefix + fresh
    ([129, 300, 5], [129, 2500, 2049]),  # small work
]


@pytest.mark.parametrize("exact", [False, True], ids=["fp8mfma", "f16mfma"])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 2)])
@pytest.mark.parametrize("page_step", [1, 2])
@pytest.mark.parametrize("q_lens,seq_lens", SHAPES, ids=[f"{a}-{b}".replace(" ", "") for a, b in SHAPES])
@torch.inference_mode()
def test_prefill_kernels_agree(q_lens, seq_lens, page_step, init_blocks, local_blocks, heads, exact, monkeypatch):
    """Every kernel and tile on the same rows (pinned): ``kernel_shared`` with 128- and 256-row tiles and, on the
    fp8 route, ``kernel.py`` with staged and per-group stores agree bit for bit on the contract's blocks (staged and
    per-group stores on every element); the first matches the reference."""
    require_gfx950()
    torch.manual_seed(0)
    q, cache, table, cu, seq = make_case(q_lens, seq_lens, heads, page_step)
    width = cdiv(max(seq_lens), BLK) + 3

    def go(shared, mq, staged=True):
        monkeypatch.setattr(host, "PIN_SHARED", shared)
        monkeypatch.setattr(host, "PIN_MQ", mq)
        monkeypatch.setattr(host, "STAGED_STORES", staged)
        return score(q, cache, table, cu, seq, q_lens, seq_lens, width, exact, init_blocks, local_blocks)

    variants: list[tuple] = [(True, 1), (True, 2)]
    if not exact:
        variants += [(False, None), (False, None, False)]
    outs = [go(*v) for v in variants]
    torch.cuda.synchronize()
    if not exact:  # staged vs per-group stores: the same elements, the same bytes
        assert torch.equal(outs[-1].nan_to_num(7.0), outs[-2].nan_to_num(7.0))
    seen = seen_mask(q_lens, seq_lens, heads, width)
    for v, out in zip(variants[1:], outs[1:]):
        assert torch.equal(out[seen], outs[0][seen]), v
    check(outs[0], q, cache, table, q_lens, seq_lens, exact, init_blocks, local_blocks)


@pytest.mark.parametrize("q_lens,seq_lens", [([129, 300], [129, 2500]), ([3000, 700], [20000, 30000])], ids=["small", "tile512"])
@torch.inference_mode()
def test_prefill_graph_replay(q_lens, seq_lens):
    """Captured once, replayed on new queries and block tables (scores NaN-filled before each replay)."""
    require_gfx950()
    heads = 1
    torch.manual_seed(7)
    q, cache, table, cu, seq = make_case(q_lens, seq_lens, heads)
    width = cdiv(max(seq_lens), BLK) + 3
    out = nan_like(heads, q.shape[0], width)

    def call():
        fly.nvfp4_index_score_prefill(
            q, cache, out, table, cu, seq, INIT, LOCAL, max(q_lens), max(seq_lens)
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(2):
        q.copy_(queries(q.shape[0], heads))
        perm = torch.randperm(cache.shape[0], device="cuda", dtype=torch.int32)
        table.copy_(perm[table.long()])
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(out, q, cache, table, q_lens, seq_lens)


@pytest.mark.parametrize("q_lens,seq_lens", [([64], [65536]), ([8192], [131072])], ids=["small", "tile512"])
@torch.inference_mode()
def test_prefill_deterministic(q_lens, seq_lens):
    require_gfx950()
    heads = 1
    torch.manual_seed(9)
    q, cache, table, cu, seq = make_case(q_lens, seq_lens, heads)
    width = cdiv(max(seq_lens), BLK)
    outs = [
        score(q, cache, table, cu, seq, q_lens, seq_lens, width).nan_to_num(7.0).view(torch.int32)
        for _ in range(3)
    ]
    torch.cuda.synchronize()
    assert all(torch.equal(outs[0], o) for o in outs[1:])
