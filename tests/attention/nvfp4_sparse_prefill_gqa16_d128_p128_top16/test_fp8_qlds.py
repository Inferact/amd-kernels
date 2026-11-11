# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""A once-converted BF16 query has the original FP8 query bytes and outputs."""
import dataclasses

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import host, reference
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.query import pack_query
from .cases import make_case, require_gfx950
from .test_sparse_prefill import (
    run, run_quant, run_pack, check_quant_bytes,
    test_nan_scales_past_length as check_nan_tail,
    test_bitwise_next_to_mfma_waves as check_mfma,
)


@pytest.mark.parametrize("tokens", [257, 511, 512, 513, 768, 1023, 1024])
@torch.inference_mode()
def test_qlds_matches_register_queries(tokens, monkeypatch):
    require_gfx950()
    case = make_case([tokens - 3, 3], [4096, 1024], 1, scales=(0.5, 2.0), seed=tokens, picks="shared")
    out = run(case, "fp8")
    get = host._get_kernel

    def register_query(*args, **kwargs):
        if tokens <= 512:
            kwargs["single"] = 1
            return get(*args, **kwargs)
        from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.fp8.kernel import compile_nvfp4_prefill_fp8
        return compile_nvfp4_prefill_fp8(args[1], 1, args[3], True)

    with monkeypatch.context() as patch:
        patch.setattr(host, "_get_kernel", register_query)
        assert torch.equal(out, run(case, "fp8"))
    prepacked = dataclasses.replace(case, q=pack_query(case.q))
    assert torch.equal(out, run(prepacked, "fp8"))
    for _ in range(2):
        assert torch.equal(out, run(case, "fp8"))
    target = torch.full_like(out, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(case, "fp8", out=target)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, target)
    codes, scales = run_quant(case, "fp8")
    check_quant_bytes(out, codes, scales, tokens)
    assert torch.equal(run_pack(case, "fp8")[:tokens], reference.sp_pack_rows(out.flatten(1)))


def test_qlds_poison_and_mfma():
    check_nan_tail("fp8", 511)
    check_mfma("fp8", 1, 512)
    check_nan_tail("fp8", 1023)
    check_mfma("fp8", 1, 1024)


@pytest.mark.parametrize("tokens", [513, 1024])
@torch.inference_mode()
def test_qlds_empty_and_partial_tiles(tokens):
    require_gfx950()
    case = make_case([tokens], [tokens + 4096], 1, seed=tokens, picks="shared")
    case.topk[:, :4].fill_(-1)
    case.topk[:, -1].fill_(-1)
    case.topk[:, 4:6, :15].fill_(-1)  # only the last pick slot can contribute
    out = run(case, "fp8")
    assert torch.isfinite(out).all()
    assert (out[:4] == 0).all() and (out[-1] == 0).all()
    prepacked = dataclasses.replace(case, q=pack_query(case.q))
    assert torch.equal(out, run(prepacked, "fp8"))
    codes, scales = run_quant(case, "fp8")
    check_quant_bytes(out, codes, scales, tokens)
    assert torch.equal(run_pack(case, "fp8")[:tokens], reference.sp_pack_rows(out.flatten(1)))
