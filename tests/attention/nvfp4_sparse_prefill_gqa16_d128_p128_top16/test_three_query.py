# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Three-query boundary, ownership, precision and fused-output contracts."""
import dataclasses

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import host, reference
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.query import pack_query
from .cases import make_case, require_gfx950
from .test_sparse_prefill import (
    run, run_quant, run_pack, check_quant_bytes, check_reference,
    test_nan_scales_past_length as check_nan_tail,
    test_bitwise_next_to_mfma_waves as check_mfma,
)


@pytest.mark.parametrize("variant", ["bf16", "fp16", "fp8"])
@pytest.mark.parametrize("tokens", [1026, 1027, 1088, 1089, 1280, 1535, 1536])
@torch.inference_mode()
def test_three_query_contract(variant, tokens, monkeypatch):
    require_gfx950()
    case = make_case([tokens - 3, 3], [8192, 700], 1, scales=(0.5, 2.0), seed=tokens)
    out = run(case, variant)
    check_reference(case, out, variant, rows=case.rows(12))
    # Compare the prior tile on every row/head; only page and sum order changes.
    with monkeypatch.context() as patch:
        if variant == "fp8":
            get = host._get_kernel
            def old_tile(*args, **kwargs):
                kwargs["single"] = 0
                return get(*args, **kwargs)
            patch.setattr(host, "_get_kernel", old_tile)
        else:
            get = host._get_bf16_local
            def old_tile(tq, pack, quant, p_f16=False, q_f16=False, **kwargs):
                return get(2, pack, quant, p_f16, q_f16, q_shared=True)
            patch.setattr(host, "_get_bf16_local", old_tile)
        old = run(case, variant)
    a, b = out.double(), old.double()
    assert (a - b).norm() / b.norm() <= 2e-3
    assert torch.nn.functional.cosine_similarity(a, b, dim=-1).min() >= 0.99999
    for _ in range(2):
        assert torch.equal(out, run(case, variant))
    target = torch.full_like(out, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(case, variant, out=target)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, target)
    codes, scales = run_quant(case, variant)
    check_quant_bytes(out, codes, scales, tokens)
    assert (codes[tokens:] == 0xA5).all()
    assert (reference.aiter_unshuffle_scales(scales)[tokens:] == 0xA5).all()
    packed = run_pack(case, variant)
    assert torch.equal(packed[:tokens], reference.sp_pack_rows(out.flatten(1)))
    assert (packed[tokens:] == 0xA5).all()
    padded = torch.empty((case.cache.shape[0] * 2, *case.cache.shape[1:]), dtype=torch.uint8, device="cuda")
    padded[::2] = case.cache
    assert torch.equal(out, run(dataclasses.replace(case, cache=padded[::2]), variant))
    if variant == "fp8":
        assert torch.equal(out, run(dataclasses.replace(case, q=pack_query(case.q)), variant))


@pytest.mark.parametrize("variant", ["bf16", "fp16", "fp8"])
@torch.inference_mode()
def test_three_query_empty_and_partial(variant):
    require_gfx950()
    case = make_case([1281], [6000], 1, seed=999, picks="shared")
    case.topk[:, :3].fill_(-1)
    case.topk[:, -1].fill_(-1)
    case.topk[:, 3:6, :15].fill_(-1)
    out = run(case, variant)
    assert torch.isfinite(out).all()
    assert (out[:3] == 0).all() and (out[-1] == 0).all()
    codes, scales = run_quant(case, variant)
    check_quant_bytes(out, codes, scales, 1281)
    assert torch.equal(run_pack(case, variant)[:1281], reference.sp_pack_rows(out.flatten(1)))


@pytest.mark.parametrize("variant", ["bf16", "fp16", "fp8"])
def test_three_query_poison_and_mfma(variant):
    check_nan_tail(variant, 1281)
    check_mfma(variant, 1, 1280)


@pytest.mark.parametrize("prepacked", [False, True])
@torch.inference_mode()
def test_three_query_compile_tier(prepacked, monkeypatch):
    require_gfx950()
    from .test_sparse_prefill import case_slice
    from .test_compile import count_compiles
    host._get_kernel.cache_clear()
    case = make_case([2048], [6144], 1, seed=112, picks="shared")
    if prepacked:
        case = dataclasses.replace(case, q=pack_query(case.q))
    launchers = [run, run_quant, run_pack]
    for size in [1024, 1280, 2048]:
        for launch in launchers:
            launch(case_slice(case, 0, size), "fp8")
    compiled = count_compiles(monkeypatch)
    for size in [1087, 1088, 1089, 1281, 1535, 1536, 1537]:
        for launch in launchers:
            launch(case_slice(case, 0, size), "fp8")
    assert not compiled, compiled


@pytest.mark.parametrize("tokens", [1026, 1027, 1089, 1281, 1535])
@pytest.mark.parametrize("q_half,lazy", [(True, False), (False, True), (True, True)])
@torch.inference_mode()
def test_three_query_fp16_rules(tokens, q_half, lazy):
    require_gfx950()
    case = make_case([tokens - 7, 7], [tokens + 3000, 700], 1, scales=(0.5, 2.0), seed=tokens, picks="shared")
    if q_half:
        case = dataclasses.replace(case, q=case.q.half())
    kw = dict(lazy_rescale=True) if lazy else {}
    out = run(case, "fp16", **kw)
    check_reference(case, out, "fp16", rows=case.rows(12))
    assert torch.equal(out, run(case, "fp16", **kw))
    codes, scales = run_quant(case, "fp16", **kw)
    check_quant_bytes(out, codes, scales, tokens)
    assert (codes[tokens:] == 0xA5).all()
    assert (reference.aiter_unshuffle_scales(scales)[tokens:] == 0xA5).all()
    packed = run_pack(case, "fp16", **kw)
    assert torch.equal(packed[:tokens], reference.sp_pack_rows(out.flatten(1)))
    assert (packed[tokens:] == 0xA5).all()
    target = torch.full_like(out, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(case, "fp16", out=target, **kw)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(target, out)
