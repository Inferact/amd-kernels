# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The small tiles at two KV heads (TP2, 32 query heads; ``host.HKV2_TILES``) at every tile boundary, and the
one-KV-head (TP4) tile choice unchanged for every token count."""

import dataclasses

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import host, reference
from .cases import TP2_HKV, make_case, require_gfx950
from .test_sparse_prefill import check_quant_bytes, check_reference, run, run_quant

LAZY = dict(lazy_rescale=True)
# both sides of every Hkv 2 tile boundary, and the four-query kernel past the last
EDGES = [1, 128, 129, 256, 257, 384, 385, 512, 513, 768, 769]


def tp4_tile(t):
    """The one-KV-head small tile of the TP4 dispatch: (queries per workgroup, shared Q), or None."""
    if t > 1536:
        return None
    return (1 if t <= 256 else 2 if t <= 1025 else 3), 512 < t <= 1025


def test_tp4_tiles_unchanged():
    for t in range(1, 65537):
        assert host._local_tile(t, 1) == tp4_tile(t), t
        assert host._local_tile(t, 4) is None


def test_tp2_tiles():
    want = {1: (1, False), 128: (1, False), 129: (2, True), 256: (2, True), 257: (3, False), 384: (3, False),
            385: (2, True), 512: (2, True), 513: (3, False), 768: (3, False), 769: None, 65536: None}
    for t, tile in want.items():
        assert host._local_tile(t, TP2_HKV) == tile, t


@pytest.mark.parametrize("variant,lazy", [("fp16", True), ("fp16", False), ("bf16", False)],
                         ids=["fp16lazy", "fp16", "bf16"])
@pytest.mark.parametrize("tokens", EDGES)
@torch.inference_mode()
def test_tp2_tile_contract(tokens, variant, lazy):
    """Two requests (one of 3 tokens) across the tiles' token groups: the reference, deterministic, graph replay bit for
    bit, the MXFP8 output = the reference quant of the bf16 output."""
    require_gfx950()
    kw = LAZY if lazy else {}
    q_lens = [tokens] if tokens < 8 else [tokens - 3, 3]
    case = make_case(q_lens, [tokens + 5000, 700][: len(q_lens)], TP2_HKV, scales=(0.5, 2.0), seed=tokens,
                     picks="shared")
    out = run(case, variant, **kw)
    check_reference(case, out, variant, rows=case.rows(24))
    for _ in range(2):
        assert torch.equal(out, run(case, variant, **kw))
    target = torch.full_like(out, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(case, variant, out=target, **kw)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, target)
    codes, scales = run_quant(case, variant, **kw)
    check_quant_bytes(out, codes, scales, tokens)
    assert (codes[tokens:] == 0xA5).all()
    assert (reference.aiter_unshuffle_scales(scales)[tokens:] == 0xA5).all()


@pytest.mark.parametrize("tokens", [129, 300, 400, 600])
@torch.inference_mode()
def test_tp2_lazy_runs_its_own_kernel(tokens):
    """The two- / three-query tiles take the lazy rescale at Hkv 2; the one-query tile keeps the eager one."""
    require_gfx950()
    case = make_case([tokens], [tokens + 3000], TP2_HKV, seed=330 + tokens, picks="random")
    lazy = run(case, "fp16", **LAZY)
    check_reference(case, lazy, "fp16")
    assert not torch.equal(lazy, run(case, "fp16"))
    small = make_case([100], [3000], TP2_HKV, seed=331, picks="random")
    assert torch.equal(run(small, "fp16", **LAZY), run(small, "fp16"))


@pytest.mark.parametrize("tokens", [128, 256, 384, 512, 768])
@torch.inference_mode()
def test_tp2_nan_scales_past_length(tokens):
    """NaN scale bytes past the sequence end (unwritten tokens, both KV heads) leave the output finite and unchanged."""
    require_gfx950()
    case = make_case([tokens], [tokens + 17], TP2_HKV, seed=61, picks="shared")
    clean = run(case, "fp16", **LAZY)
    kc, ks, vc, vs = reference.unpack_kv_pages(case.cache, TP2_HKV)
    end = int(case.pos[-1]) + 1
    page = int(case.table[0, end // 128])
    ks[page, :, end % 128:] = 0x7F
    vs[page, :, end % 128:] = 0x7F
    dirty = run(dataclasses.replace(case, cache=reference.pack_kv_pages(kc, ks, vc, vs)), "fp16", **LAZY)
    assert torch.isfinite(dirty).all()
    assert torch.equal(clean, dirty)


@pytest.mark.parametrize("tokens", [128, 256, 384, 512, 768, 2048])
def test_tp2_lazy_bitwise_next_to_mfma_waves(tokens):
    """With another kernel's MFMA waves on every CU the served output (fp16 lazy) equals an undisturbed run's."""
    require_gfx950()
    from tests.mfma_spinner import mfma_spinner

    case = make_case([tokens], [tokens + 30000], TP2_HKV, seed=7)
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
