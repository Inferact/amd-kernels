# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compressor tail + indexer key store against ``reference.py``: the ring, the main cache and the indexer cache bit for
bit, except the few indexer bytes whose ``k_pre`` the fp32 MFMA sum rounds differently from the contract's exact sum (a
bf16 ulp in about 1 in 1000 columns); and, with the vLLM / AITER chain installed, against that chain."""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.attention.compressor_insert_d512_idx128_rot64 import (
    compressor_insert,
    shuffle_wk,
    supports_compressor_insert,
)
from amd_kernels.attention.compressor_insert_d512_idx128_rot64.reference import compressor_insert_ref

from .common import make, require_gfx950

ROWS = [16, 48, 96, 192, 384, 768]
IDX_BAD_BYTES = 4  # indexer bytes that may differ (k_pre rounding)


def run(args):
    a = dict(args)
    a["ring"] = None if a["ring"] is None else a["ring"].clone()
    a["main_cache"] = a["main_cache"].clone()
    a["idx_cache"] = a["idx_cache"].clone()
    a["wk"] = shuffle_wk(a["wk"])
    compressor_insert(**a)
    torch.cuda.synchronize()
    return a["ring"], a["main_cache"], a["idx_cache"]


def check(args, got, want):
    ring, main, idx = got
    r_ring, r_main, r_idx = want
    if ring is not None:
        assert torch.equal(ring, r_ring), "ring"
    assert torch.equal(main.view(torch.uint8), r_main.view(torch.uint8)), "main cache"
    assert (idx != r_idx).sum().item() <= IDX_BAD_BYTES, "indexer cache"


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_decode_matches_reference(ratio, rows):
    require_gfx950()
    args = make(ratio, rows)
    assert supports_compressor_insert(args["raw"], ratio, args["main_cache"], args["idx_cache"], shuffle_wk(args["wk"]),
                                      args["cos_sin_main"], args["cos_sin_idx"])
    got = run(args)
    want = compressor_insert_ref(**args)
    check(args, got, want)
    again = run(args)
    for a, b in zip(got, again):
        if a is not None:
            assert torch.equal(a.view(torch.uint8) if a.dtype != torch.uint8 else a, b.view(torch.uint8) if b.dtype != torch.uint8 else b)


@pytest.mark.parametrize("ratio", [1, 2])
@torch.inference_mode()
def test_prefill_chunks_match_reference(ratio):
    require_gfx950()
    args = make(ratio, 160, lens=[37, 1, 2, 5, 64, 3, 9, 20], seed=3)
    check(args, run(args), compressor_insert_ref(**args))


@pytest.mark.parametrize("cs_dtype", [torch.bfloat16])
@torch.inference_mode()
def test_bf16_indexer_cache(cs_dtype):
    require_gfx950()
    args = make(2, 96, cs_idx_dtype=cs_dtype)
    check(args, run(args), compressor_insert_ref(**args))


def baseline_chain(args):
    """The vLLM / AITER chain the kernel replaces; None when they are not installed."""
    try:
        from aiter.ops.triton.fusions.k_norm_rope_mxfp4_cache import k_norm_rope_mxfp4_cache
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            fused_save_compress_norm,
            rope_quant_insert,
        )
    except ImportError:
        return None
    ratio, n = args["ratio"], args["raw"].shape[0]
    ring = None if args["ring"] is None else args["ring"].clone()
    main, idx = args["main_cache"].clone(), args["idx_cache"].clone()
    latent = torch.empty((n, 512), dtype=torch.bfloat16, device="cuda")
    fused_save_compress_norm(args["raw"], args["positions"], ring, args["ring_slots"], args["query_start_loc"],
                             args["token_to_req"], args["norm_weight"], args["eps"], ratio, latent)
    rope_quant_insert(latent, args["positions"], args["cos_sin_main"], main, args["main_slots"], ratio,
                      fp8_scale=args["kv_scale"])
    k_pre = latent @ args["wk"].T
    entries = idx.shape[1] // 68
    k_norm_rope_mxfp4_cache(k_pre, args["positions"], args["cos_sin_idx"], args["k_norm_weight"], args["k_eps"], idx,
                            args["idx_slots"], ratio, shuffle=(32, 16, 2))
    return ring, main, idx


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("rows", [96, 768])
@torch.inference_mode()
def test_matches_vllm_aiter_chain(ratio, rows):
    require_gfx950()
    args = make(ratio, rows)
    want = baseline_chain(args)
    if want is None:
        pytest.skip("vLLM / AITER chain not available")
    ring, main, idx = run(args)
    if ring is not None:
        assert torch.equal(ring, want[0])
    n_main = (main.view(torch.uint8) != want[1].view(torch.uint8)).sum().item()
    n_idx = (idx != want[2]).sum().item()
    # the chain sums its squares in the Triton tree order: a latent bf16 ulp now and then moves a code
    assert n_main <= 1e-4 * rows * 512 and n_idx <= 2e-3 * rows * 68, (n_main, n_idx)


@pytest.mark.parametrize("ratio", [1, 2])
@torch.inference_mode()
def test_no_compile_after_one_call(ratio, monkeypatch):
    require_gfx950()
    args = make(ratio, 768)
    keys = ("raw", "positions", "main_slots", "idx_slots", "token_to_req", "ring_slots")

    def call(n):
        a = {k: (v[:n] if k in keys and v is not None else v) for k, v in args.items()}
        a["wk"] = shuffle_wk(a["wk"])
        compressor_insert(**a)

    call(16)
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile
    monkeypatch.setattr(flyc, "compile", lambda fn, *a, **k: compiled.append(fn) or real(fn, *a, **k))
    for n in (48, 96, 192, 384, 768):
        call(n)
    assert not compiled
