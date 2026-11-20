# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``index_decode.nvfp4_index_sparse_decode``, the decode block (index top-16 + sparse decode): the route choice, and
on one GPU both routes (``fused``: the scorer + the attention's in-kernel top-16; ``chain``: scorer + selector, then
the attention) bit for bit the three-launch chain (``nvfp4_index_score_top16`` + ``nvfp4_sparse_decode``), ids and
output, bf16 and MXFP8 outputs, bf16 and fp16 compute, at the served shapes (TP4: one KV / index head x 16 heads, TP2:
two, 4 / 8 rows per request, 1-64 requests, 32K-192K). The four-GPU route: ``test_index_decode_ranks4.py``; the
two-GPU one: ``test_index_decode_ranks2.py``. Warm-up:
``test_index_decode_warm_up.py``."""
from types import SimpleNamespace

import pytest
import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host as ah
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import index_decode as blk
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import decode_top16
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.decode_top16 import IndexDecodeTop16
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import nvfp4_index_score_top16

from .test_index_select import MAX, build
from .test_sparse_decode import quant_buffers, require_gfx950


def test_route():
    one = IndexDecodeTop16("cpu", query_len=8, max_reqs=64, max_seq_len=MAX, warm_up=False)
    assert blk.route(None, 32, 8, 1, 1) == "fused" and blk.route(one, 32, 8, 1, 1) == "fused"
    assert blk.route(None, 33, 8, 1, 1) == "chain"  # past SELECT_MAX_ROWS
    assert blk.route(None, 4, 8, 1, 1, torch.float16) == "fused"
    assert blk.route(None, 4, 8, 1, 1, torch.float8_e4m3fn) == "chain"  # fp8 selects in no attention kernel
    assert blk.route(None, 4, 8, 4, 1) == "chain"  # index heads != KV heads
    assert blk.route(None, 128, 8, 2, 2) == "fused" and blk.route(None, 129, 8, 2, 2) == "chain"  # TP2
    one.split = SimpleNamespace(query_len=8, max_rows=256 * 8)
    n = decode_top16.SPLIT_MIN_REQS[8]  # split4 from n requests on (decode_top16.route)
    assert blk.route(one, n - 1, 8, 1, 1) == "fused" and blk.route(one, n, 8, 1, 1) == "split4"
    assert blk.route(one, n, 8, 1, 1, split=False) == "fused"
    two = IndexDecodeTop16("cpu", query_len=8, max_reqs=64, max_seq_len=MAX, num_index_heads=2, warm_up=False)
    two.split2 = SimpleNamespace(query_len=8, max_rows=64 * 8)
    n2 = decode_top16.SPLIT2_MIN_REQS[8]  # split2 from n2 requests on (TP2: two index heads per rank)
    assert blk.route(two, n2 - 1, 8, 2, 2, torch.float16) == "fused"
    assert blk.route(two, n2, 8, 2, 2, torch.float16) == "split2"
    assert blk.route(two, n2, 8, 2, 2, torch.float16, split=False) == "fused"
    assert blk.route(two, 65, 8, 2, 2, torch.float16) == "fused" and blk.route(two, n2, 8, 1, 1) == "fused"


def three_launch_chain(c, init, local, ids, out, quant, compute=torch.bfloat16):
    one = torch.ones((), dtype=torch.float32, device="cuda")
    nvfp4_index_score_top16(c["iq"], c["icache"], c["bt"], c["seq"], init, local, ids,
                            query_len=c["ql"], max_seq_len=MAX)
    ah.nvfp4_sparse_decode(c["q"], c["kv"], ids, c["bt"], c["seq"], 128**-0.5, out, k_scale=one, v_scale=one,
                           decode_query_len=c["ql"], quant_out=quant, compute_type=compute)


@pytest.mark.parametrize("hkv", [1, 2], ids=["kv1", "kv2"])
@pytest.mark.parametrize("compute", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize("reqs", [1, 2, 3, 4, 5, 8, 17, 32, 33, 64])
@pytest.mark.parametrize("ctx", [32768, 196608])
@pytest.mark.parametrize("quant", [False, True])
def test_one_gpu_block_matches_three_launch_chain(ql, reqs, ctx, quant, compute, hkv):
    require_gfx950()
    c = build([max(1, ctx - 4093 * r) for r in range(reqs)], ql, seed=reqs * 7 + ql + ctx % 97, hkv=hkv)
    rows = c["rows"]
    res = []
    for new in (False, True):
        ids = torch.full((hkv, rows + 3, 16), -7, dtype=torch.int32, device="cuda")
        out = torch.full((rows, 16 * hkv, 128), 7.0, dtype=torch.bfloat16, device="cuda")
        q = (*quant_buffers(rows, 2048 * hkv), 16) if quant else None
        if new:
            r = blk.nvfp4_index_sparse_decode(None, c["iq"], c["icache"], c["bt"], c["seq"], 2, 3, c["q"], c["kv"], ids, c["bt"],
                                              c["seq"], 128**-0.5, out, k_scale=None, v_scale=None,
                                              decode_query_len=ql, max_seq_len=MAX, quant_out=q, compute_type=compute)
            assert r == ("fused" if rows <= ah.SELECT_MAX_ROWS[hkv] else "chain")
        else:
            three_launch_chain(c, 2, 3, ids[:, :rows], out, q, compute)
        torch.cuda.synchronize()
        res.append((ids, out, q))
    (ia, oa, qa), (ib, ob, qb) = res
    assert torch.equal(ia, ib)
    if quant:
        assert torch.equal(qa[0], qb[0]) and torch.equal(qa[1], qb[1])
    else:
        assert torch.equal(oa.view(torch.int16), ob.view(torch.int16))
