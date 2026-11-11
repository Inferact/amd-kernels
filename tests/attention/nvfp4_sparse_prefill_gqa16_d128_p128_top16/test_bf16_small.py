# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""BF16 and FP16 small-grid variants (one- / two-query tiles) preserve the arithmetic and fused output contracts."""
import dataclasses

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import host, reference
from .cases import make_case, require_gfx950
from .test_sparse_prefill import (
    run, run_quant, run_pack, check_quant_bytes, test_nan_scales_past_length as check_nan_tail,
    test_bitwise_next_to_mfma_waves as check_mfma,
)


@pytest.mark.parametrize('variant', ['bf16', 'fp16'])
@pytest.mark.parametrize('tokens', [255, 256, 257, 511, 512, 513, 1024, 1536, 1537])
@torch.inference_mode()
def test_small_bf16_contract(tokens, variant, monkeypatch):
    require_gfx950()
    case = make_case([tokens - 3, 3], [3000, 700], 1, scales=(0.5, 2.0), seed=tokens, picks='shared')
    out = run(case, variant)
    rows = case.rows(12)
    ref = reference.sparse_prefill_reference(*case.args(), k_scale=case.k_scale, v_scale=case.v_scale, rows=rows)
    cos = torch.nn.functional.cosine_similarity(out[rows].double(), ref, dim=-1)
    assert cos.min().item() >= 0.99999
    with monkeypatch.context() as patch:
        patch.setattr(host, '_get_bf16_local',
                      lambda tq, pack, quant, p_f16=False, q_f16=False, **kw: host._get_kernel(variant, pack, 1, quant))
        old = run(case, variant)
    a, b = out.double(), old.double()
    assert (a - b).norm() / b.norm() <= 2e-3
    assert torch.nn.functional.cosine_similarity(a, b, dim=-1).min() >= 0.99999
    for _ in range(2):
        assert torch.equal(out, run(case, variant))
    codes, scales = run_quant(case, variant)
    check_quant_bytes(out, codes, scales, tokens)
    target = torch.full_like(out, float('nan'))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(case, variant, out=target)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, target)
    padded = torch.zeros((case.cache.shape[0] * 2, *case.cache.shape[1:]), dtype=torch.uint8, device='cuda')
    padded[::2] = case.cache
    assert torch.equal(out, run(dataclasses.replace(case, cache=padded[::2]), variant))


@pytest.mark.parametrize('variant', ['bf16', 'fp16'])
@pytest.mark.parametrize('tokens', [129, 512, 1024, 1281])
def test_small_bf16_poison_and_mfma(tokens, variant):
    check_nan_tail(variant, tokens)
    check_mfma(variant, 1, tokens)


@pytest.mark.parametrize('variant,tokens', [('bf16', 513), ('fp16', 129), ('fp16', 513), ('fp8', 511)])
@torch.inference_mode()
def test_small_page_walk_empty_and_last_slot(variant, tokens):
    require_gfx950()
    case = make_case([tokens], [4096], 1, seed=tokens, picks='shared')
    case.topk[:, :4].fill_(-1)
    case.topk[:, -1].fill_(-1)
    case.topk[:, 4:6, :15].fill_(-1)
    case.topk[:, 4:6, 15] = 0
    out = run(case, variant)
    assert torch.isfinite(out).all()
    assert (out[:4] == 0).all() and (out[-1] == 0).all()
    rows = [4, 5, 6, tokens - 2]
    ref = reference.sparse_prefill_reference(*case.args(), rows=rows, fp8=variant == 'fp8')
    assert reference.rel_l2(out[rows], ref) < reference.REL_L2_TOL[variant]
    assert torch.equal(out, run(case, variant))
    codes, scales = run_quant(case, variant)
    check_quant_bytes(out, codes, scales, tokens)
    assert torch.equal(run_pack(case, variant)[:tokens], reference.sp_pack_rows(out.flatten(1)))


@pytest.mark.parametrize('tokens', [513, 769, 1023])
@pytest.mark.parametrize('variant,q_half,lazy', [
    ('bf16', False, False), ('fp16', False, False), ('fp16', True, False),
    ('fp16', False, True), ('fp16', True, True),
])
@torch.inference_mode()
def test_shared_query_bitwise(tokens, variant, q_half, lazy, monkeypatch):
    require_gfx950()
    case = make_case([tokens - 7, 7], [tokens + 3000, 700], 1, scales=(0.5, 2.0), seed=tokens, picks='shared')
    if q_half:
        case = dataclasses.replace(case, q=case.q.half())
    kw = dict(lazy_rescale=True) if lazy else {}
    out = run(case, variant, **kw)
    codes, scales = run_quant(case, variant, **kw)
    packed = run_pack(case, variant, **kw)
    check_quant_bytes(out, codes, scales, tokens)
    assert torch.equal(packed[:tokens], reference.sp_pack_rows(out.flatten(1)))
    original = host._get_bf16_local
    with monkeypatch.context() as patch:
        patch.setattr(host, '_get_bf16_local',
                      lambda *a, **k: original(*a, **{**k, 'q_shared': False}))
        assert torch.equal(out, run(case, variant, **kw))
        old_codes, old_scales = run_quant(case, variant, **kw)
        assert torch.equal(codes[:tokens], old_codes[:tokens])
        assert torch.equal(reference.aiter_unshuffle_scales(scales)[:tokens],
                           reference.aiter_unshuffle_scales(old_scales)[:tokens])
        assert torch.equal(packed[:tokens], run_pack(case, variant, **kw)[:tokens])


@pytest.mark.parametrize('q_half', [False, True])
@torch.inference_mode()
def test_quant_query_preload_next_to_mfma(q_half, monkeypatch):
    require_gfx950()
    from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.f16mfma_gqa16_d128_p128_top16_tq1_walk import (
        compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq1_walk,
    )
    from tests.mfma_spinner import mfma_spinner

    case = make_case([255], [7000], 1, seed=810, picks='shared')
    if q_half:
        case = dataclasses.replace(case, q=case.q.half())
    out = run(case, 'fp16')
    expected = run_quant(case, 'fp16')
    check_quant_bytes(out, *expected, case.q.shape[0])
    with monkeypatch.context() as patch:
        patch.setattr(host, '_get_bf16_local',
                      lambda *a, **k: compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq1_walk(False, True, q_half))
        old = run_quant(case, 'fp16')
    assert torch.equal(expected[0][:255], old[0][:255])
    assert torch.equal(reference.aiter_unshuffle_scales(expected[1])[:255],
                       reference.aiter_unshuffle_scales(old[1])[:255])
    main, side = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)
    spin = mfma_spinner(torch.cuda.get_device_properties(0).multi_processor_count)
    for _ in range(8):
        spin(200_000, side)
        with torch.cuda.stream(main):
            codes, scales = run_quant(case, 'fp16')
        torch.cuda.synchronize()
        assert torch.equal(codes[:255], expected[0][:255])
        assert torch.equal(reference.aiter_unshuffle_scales(scales)[:255],
                           reference.aiter_unshuffle_scales(expected[1])[:255])
