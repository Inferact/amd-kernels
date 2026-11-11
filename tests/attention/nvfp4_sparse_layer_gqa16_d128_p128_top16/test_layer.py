# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The layer calls are the per-kernel calls at the served compute dtypes, bit for bit: the decode, the index top-16 +
decode block (ids and output) and the prefill, bf16 and MXFP8 outputs; ``quant_out_buffers`` declines below
``QUANT_OUT_MIN_ROWS`` rows and is ``gemm.prequant_buffers`` from there. The warm-up: ``test_init_warm_up.py``."""

import pytest
import torch
from amd_kernels.attention import nvfp4_sparse_layer_gqa16_d128_p128_top16 as layer
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host as dh
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import index_decode as blk
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import nvfp4_sparse_prefill
from amd_kernels.gemm.mxfp8_common import gemm

from tests.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.test_index_select import MAX, build
from tests.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.test_sparse_decode import (
    SM_SCALE,
    lengths_for,
    make_case,
    quant_buffers,
    require_gfx950,
)
from tests.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.cases import TP4_HKV
from tests.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.cases import make_case as prefill_case


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def _same(a, b):
    return all(torch.equal(_bytes(x), _bytes(y)) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("reqs", [1, 4, 33])
def test_sparse_decode_is_the_fp16_decode(reqs, quant):
    require_gfx950()
    c = make_case(1, 8, lengths_for(reqs, reqs), reqs)
    rows = c["q"].shape[0]
    res = []
    for fn, kw in ((layer.sparse_decode, {}), (dh.nvfp4_sparse_decode, {"compute_type": torch.float16})):
        out = torch.full_like(c["q"], float("nan"))
        q = (*quant_buffers(rows, 2048), 16) if quant else None
        fn(c["q"], c["kv_cache"], c["topk_idx"], c["block_table"], c["seq_lens"], SM_SCALE, out,
           k_scale=c["k_scale"], v_scale=c["v_scale"], decode_query_len=c["ql"], quant_out=q, **kw)
        res.append((out, *(q[:2] if quant else ())))
    torch.cuda.synchronize()
    assert _same(*res)


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("reqs", [1, 5, 32, 33])
def test_index_sparse_decode_is_the_fp16_block(reqs, quant):
    require_gfx950()
    c = build([max(1, 65536 - 4093 * r) for r in range(reqs)], 8, seed=reqs)
    rows = c["rows"]
    res, routes = [], []
    for fn, kw in ((layer.index_sparse_decode, {}), (blk.nvfp4_index_sparse_decode, {"compute_type": torch.float16})):
        ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
        out = torch.full((rows, 16, 128), 7.0, dtype=torch.bfloat16, device="cuda")
        q = (*quant_buffers(rows, 2048), 16) if quant else None
        routes.append(fn(None, c["iq"], c["icache"], c["bt"], c["seq"], 2, 3, c["q"], c["kv"], ids, c["bt"], c["seq"],
                         128**-0.5, out, k_scale=None, v_scale=None, decode_query_len=8, max_seq_len=MAX, quant_out=q,
                         **kw))
        res.append((ids, out, *(q[:2] if quant else ())))
    torch.cuda.synchronize()
    assert routes[0] == routes[1] and _same(*res)


@pytest.mark.parametrize("q_lens", [[7], [600, 40], [3000]])
def test_sparse_prefill_is_the_fp16_lazy_prefill(q_lens):
    require_gfx950()
    c = prefill_case(q_lens, [q + 5000 for q in q_lens], TP4_HKV, seed=len(q_lens))
    res = []
    for fn, kw in ((layer.sparse_prefill, {}), (nvfp4_sparse_prefill, {"compute_dtype": "fp16", "lazy_rescale": True})):
        out = torch.full(c.q.shape, float("nan"), dtype=torch.bfloat16, device="cuda")
        fn(*c.args(), out, k_scale=c.k_scale, v_scale=c.v_scale, **kw)
        res.append((out,))
    torch.cuda.synchronize()
    assert _same(*res)


def test_quant_out_buffers():
    n, k = 6144, 2048
    for rows in range(1, layer.QUANT_OUT_MIN_ROWS):
        assert layer.quant_out_buffers(rows, n, k, "cuda") is None
    for rows in (layer.QUANT_OUT_MIN_ROWS, 9, 64, 129, 256, 2048):
        got, want = layer.quant_out_buffers(rows, n, k, "cuda"), gemm.prequant_buffers(rows, n, k, "cuda")
        assert got[2] == want[2]
        for a, b in zip(got[:2], want[:2], strict=True):
            assert (a.shape, a.dtype, a.device) == (b.shape, b.dtype, b.device)


def test_init_without_routes_builds_nothing():
    assert layer.init("cuda", num_heads=16, num_kv_heads=1, num_index_heads=1, query_len=8, max_reqs=4,
                      max_seq_len=MAX, insert=False, attention=False, index=False) is None
