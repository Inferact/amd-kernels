# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``decode_top16.IndexDecodeTop16``: the route by request count, the one-GPU route's ids bit for bit
``nvfp4_index_score_top16``'s, and nothing compiling after the warm-up at the served sizes. The four-GPU route:
``tests/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/test_index_decode_ranks4.py``."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import decode_top16
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.decode_top16 import IndexDecodeTop16, use_split
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import BLK

from .inputs import block_table, index_cache, queries, require_gfx950


@pytest.mark.parametrize("ql,reqs,want", [
    (8, 1, False), (8, 4, False), (8, 5, True), (8, 256, True), (8, 257, False),
    (4, 1, False), (4, 7, False), (4, 8, True), (4, 256, True), (6, 1, True),
])
def test_use_split(ql, reqs, want):
    assert use_split(reqs, ql, 1, split_query_len=ql, split_max_rows=256 * ql) == want


def test_route_needs_one_head_same_rows_and_switch():
    d = IndexDecodeTop16("cpu", query_len=8, max_reqs=64, max_seq_len=1 << 20, warm_up=False)
    assert d.split is None and d.route(64, 8, 1) == "one_gpu"
    d.split = SimpleNamespace(query_len=8, max_rows=64 * 8)
    assert decode_top16.SPLIT_MIN_REQS == {8: 5, 4: 8}  # the served values
    assert d.route(5, 8, 1) == "split4" and d.route(4, 8, 1) == "one_gpu"
    assert d.route(64, 8, 1) == "split4" and d.route(65, 8, 1) == "one_gpu"  # past the buffers
    assert d.route(16, 4, 1) == "one_gpu"  # rows per request the route was not built for
    assert d.route(16, 8, 2) == "one_gpu"  # several index heads
    assert d.route(16, 8, 1, split=False) == "one_gpu"


@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize("ctx", [32768, 196608])
@pytest.mark.parametrize("reqs", [1, 2, 3, 5, 17, 64])
def test_one_gpu_ids(ql, ctx, reqs):
    require_gfx950()
    torch.manual_seed(reqs * 31 + ql)
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import nvfp4_index_score_top16

    lens = [max(ql, ctx - 977 * r) for r in range(reqs)]
    table, pages = block_table(lens)
    cache = index_cache(pages)
    iq = queries(reqs * ql, 1)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    d = IndexDecodeTop16(torch.device("cuda"), query_len=ql, max_reqs=64, max_seq_len=1 << 20, warm_up=False)
    got = torch.full((1, reqs * ql, 16), -7, dtype=torch.int32, device="cuda")
    d(iq, cache, table, seq, 2, 3, got, query_len=ql, max_seq_len=1 << 20)
    want = torch.full_like(got, -7)
    nvfp4_index_score_top16(iq, cache, table, seq, 2, 3, want, query_len=ql, max_seq_len=1 << 20)
    torch.cuda.synchronize()
    assert torch.equal(got, want)


def scenario(ql):
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up

    torch.set_num_threads(8)
    warm_up.set_auto(True)
    warm_up.set_workers(1)
    count = []
    original = MlirCompiler.compile

    def spy(*a, **kw):
        count.append(1)
        return original(*a, **kw)

    MlirCompiler.compile = spy
    hmax = 1 << 20
    d = IndexDecodeTop16(torch.device("cuda"), query_len=ql, max_reqs=256, max_seq_len=hmax)
    warm = len(count)
    cache = torch.zeros(2, BLK, 72, device="cuda", dtype=torch.uint8)
    late = []
    for n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 31, 32, 33, 64, 65, 128, 129, 256):
        for ctx in (32768, 196608, hmax):
            before = len(count)
            iq = torch.zeros(n * ql, 1, 128, device="cuda", dtype=torch.float8_e4m3fn)
            table = torch.zeros(n, ctx // BLK, device="cuda", dtype=torch.int32)
            seq = torch.full((n,), ctx, device="cuda", dtype=torch.int32)
            out = torch.empty(1, n * ql, 16, device="cuda", dtype=torch.int32)
            d(iq, cache, table, seq, 0, 1, out, query_len=ql, max_seq_len=ctx)
            torch.cuda.synchronize()
            if len(count) != before:
                late.append((n, ctx))
    return dict(warm=warm, late=late, reported=warm_up.late_compiles())


@pytest.mark.parametrize("ql", [8, 4])
def test_warm_up_then_no_compile(ql, tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__, str(ql)], capture_output=True, text=True, timeout=1500,
                          env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                               "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["warm"] > 0 and not result["late"] and result["reported"] == 0, result


if __name__ == "__main__":
    print(json.dumps(scenario(int(sys.argv[1]))))
