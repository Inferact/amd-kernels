# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The opt-in lazy rescale of fp16 compute (``lazy_rescale=True``): fp16 P against a running reference re-based only
when a row max passes it by 2^8, on the two- / three-query tiles (257-1536 tokens at one KV head) and the two-round
dimension-split kernel. Another numeric rule than the eager fp16 path: the same tolerance against the fp64
contract, deterministic, the fused outputs equal the reference quant / pack of its own bf16 output."""

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import reference
from .cases import TP4_HKV, make_case, require_gfx950
from .test_sparse_prefill import check_quant_bytes, check_reference, run, run_pack, run_quant

LAZY = dict(lazy_rescale=True)


@pytest.mark.parametrize(
    "hkv, tokens, picks",
    [
        (TP4_HKV, 300, "shared"),  # two-query tile
        (TP4_HKV, 1024, "random"),
        (TP4_HKV, 1536, "shared"),
        (TP4_HKV, 2048, "random"),  # dimension-split kernel, two rounds
        (TP4_HKV, 4099, "shared"),  # incomplete final four-query tile
    ],
)
@torch.inference_mode()
def test_lazy_against_reference(hkv, tokens, picks):
    require_gfx950()
    case = make_case([tokens], [tokens + 3000], hkv, seed=310 + tokens, picks=picks)
    out = run(case, "fp16", **LAZY)
    check_reference(case, out, "fp16")
    assert not torch.equal(out, run(case, "fp16")), "the lazy path must run its own kernel"
    for _ in range(2):
        assert torch.equal(out, run(case, "fp16", **LAZY))
    target = torch.full_like(out, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(case, "fp16", out=target, **LAZY)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, target)
    codes, scales = run_quant(case, "fp16", **LAZY)
    check_quant_bytes(out, codes, scales, tokens)
    if hkv == TP4_HKV:
        assert torch.equal(run_pack(case, "fp16", **LAZY)[:tokens], reference.sp_pack_rows(out.flatten(1)))


@torch.inference_mode()
def test_lazy_global_scales():
    require_gfx950()
    case = make_case([2048], [6000], TP4_HKV, scales=(0.5, 2.0), seed=321, picks="shared")
    check_reference(case, run(case, "fp16", **LAZY), "fp16")


@torch.inference_mode()
def test_lazy_keeps_the_eager_one_query_tile():
    """T <= 256 at one KV head runs the eager one-query tile (the lazy form is slower there)."""
    require_gfx950()
    case = make_case([200], [5000], TP4_HKV, seed=322, picks="shared")
    assert torch.equal(run(case, "fp16", **LAZY), run(case, "fp16"))


@pytest.mark.parametrize("variant", ["bf16", "fp8"])
def test_lazy_needs_fp16(variant):
    require_gfx950()
    case = make_case([16], [400], TP4_HKV, seed=323)
    with pytest.raises(ValueError):
        run(case, variant, **LAZY)


@pytest.mark.parametrize("hkv,n_tok", [(TP4_HKV, 1024), (TP4_HKV, 1280), (TP4_HKV, 4096)])
def test_lazy_bitwise_next_to_mfma_waves(hkv, n_tok):
    """With another kernel's MFMA waves on every CU the lazy output equals an undisturbed run's."""
    require_gfx950()
    from tests.mfma_spinner import mfma_spinner

    case = make_case([n_tok], [n_tok + 30000], hkv, seed=7)
    main, side = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)

    def on(stream):
        with torch.cuda.stream(stream):
            return run(case, "fp16", **LAZY)

    ref = on(main)
    torch.cuda.synchronize()
    spin = mfma_spinner(torch.cuda.get_device_properties(0).multi_processor_count)
    for _ in range(8):
        spin(200_000, side)
        out = on(main)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
