# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``sparse_mla_decode`` against the contract in ``sparse_mla_decode_fp8kv_h64_d512_rot64/reference.py``.

Served shapes (README, "What it does"): 6 verify rows per request or 1..6 mixed, 16..768 rows, the 128-key window
and the top-512 list, 32- / 64- / 128-row cache blocks, a padded page pitch, the window alone (a layer without
indexer), padding rows with empty lists. Every case: the bf16 output against the fp64 reference per row, the MXFP8
bytes equal to the reference's quantization of that bf16 output, three launches bit for bit equal, and a CUDA graph
replayed into NaN-filled outputs. The caches carry e4m3 NaN bytes in every row no list names.
"""

import pytest
import torch
from amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64 import reference as R
from amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64 import sparse_mla_decode

from .cases import D, H, NAN_BYTE, make_case, outputs, require_gfx950, row_lengths

ARGS = ("q", "main_cache", "main_indptr", "main_indices", "extra_cache", "extra_indptr", "extra_indices", "attn_sink",
        "softmax_scale", "q_scale", "kv_scale", "positions", "cos_sin")  # fmt: skip
EDGE = [(1, 0, 5), (1, 1, 7), (63, 0, 70), (64, 64, 300), (65, 63, 301), (128, 512, 4000), (128, 511, 4000),
        (5, 64, 9), (100, 2, 101), (128, 449, 2000), (2, 3, 3), (128, 130, 300), (64, 0, 64), (0, 0, 10), (0, 64, 128),
        (128, 512, 9000), (31, 33, 100), (1, 512, 1500)]  # fmt: skip
# (id, rows spec, kwargs of make_case)
CASES = [
    ("t16_mixed", [6, 6, 4], {}),
    ("t48_ql6", [6] * 8, {}),
    ("t96_mixed", [6, 1, 3, 2, 6, 5, 4, 6, 2, 6, 1, 3, 6, 5, 6, 6, 6], {}),
    ("t192_ql6", [6] * 32, {}),
    ("t30_pad", [6, 6, 6, 6, 6], {"pad": 3}),
    ("t384_ql6", [6] * 64, {"pool_rows": 1 << 17}),
    ("t100_blocks128", [6] * 16 + [4], {"swa_block": 128, "comp_block": 128}),
    ("t100_blocks64_32", [6] * 16 + [4], {"swa_block": 64, "comp_block": 32}),
    ("t60_pitch", [6] * 10, {"pitch": 8}),
    ("t60_swa_only", [6] * 10, {"has_extra": False}),
    ("t60_peaked", [6] * 10, {"q_std": 4.0}),
    ("t60_flat", [6] * 10, {"q_std": 0.25}),
    ("t768_ql6", [6] * 128, {"pool_rows": 1 << 17}),
]  # fmt: skip


def build(name, spec, kw):
    lens = row_lengths(spec) if name != "edge" else EDGE
    return make_case(lens, **kw)


def launch(case, rows, *, mxfp8, **kw):
    args = [case[k] for k in ARGS]
    if mxfp8:
        codes, scales = outputs(rows)
        sparse_mla_decode(*args, codes, scales, **kw)
        return codes, scales
    out = torch.full((rows, H, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    sparse_mla_decode(*args, None, None, out, **kw)
    return out


def reference(case):
    stride = lambda c: None if c is None else c.stride(0) // D
    return R.attention_reference(
        case["q"], 1.0, case["main_cache"], case["main_indptr"], case["main_indices"], case["extra_cache"],
        case["extra_indptr"], case["extra_indices"], 1.0, case["attn_sink"], case["softmax_scale"],
        stride(case["main_cache"]), stride(case["extra_cache"]),
    )  # fmt: skip


def check_attention(out, ref, name):
    cos = torch.nn.functional.cosine_similarity(out.double().flatten(1), ref.double().flatten(1), dim=1)
    empty = (ref.flatten(1) == 0).all(dim=1)
    assert not torch.isnan(out.float()).any(), name
    assert (out[empty] == 0).all(), name  # a row without keys is zero
    assert cos[~empty].min() >= R.COS_MIN, (name, cos[~empty].min().item())
    return cos[~empty].min().item()


@pytest.mark.parametrize("name,spec,kw", CASES + [("edge", None, {})], ids=[c[0] for c in CASES] + ["edge"])
@torch.inference_mode()
def test_decode(name, spec, kw):
    require_gfx950()
    case = build(name, spec, kw)
    rows = case["q"].shape[0]
    out = launch(case, rows, mxfp8=False)
    check_attention(out, reference(case), name)
    # the MXFP8 output: byte for byte the reference's quantization of the bf16 output
    codes, scales = launch(case, rows, mxfp8=True)
    rc, rs = R.epilogue_reference(out, case["positions"], case["cos_sin"])
    assert torch.equal(codes.view(torch.uint8), rc) and torch.equal(scales, rs), name
    again = launch(case, rows, mxfp8=True)
    assert torch.equal(again[0].view(torch.uint8), codes.view(torch.uint8)) and torch.equal(again[1], scales), name
    assert torch.equal(launch(case, rows, mxfp8=False), out), name


@pytest.mark.parametrize("name,spec,kw", [CASES[1], CASES[4], CASES[8]], ids=[CASES[1][0], CASES[4][0], CASES[8][0]])
@torch.inference_mode()
def test_graph_replay(name, spec, kw):
    require_gfx950()
    case = build(name, spec, kw)
    rows = case["q"].shape[0]
    codes, scales = launch(case, rows, mxfp8=True)
    ref_codes, ref_scales = codes.view(torch.uint8).clone(), scales.clone()
    args = [case[k] for k in ARGS]
    sparse_mla_decode(*args, codes, scales)  # compile before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla_decode(*args, codes, scales)
    for _ in range(3):
        codes.view(torch.uint8).fill_(NAN_BYTE)
        scales.fill_(255)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(codes.view(torch.uint8), ref_codes) and torch.equal(scales, ref_scales), name


# (heads per workgroup, workgroups per XCD): the three head-group kernels, and the 64-head one with a few workgroups so that
# each runs several rows in a row (the next row's first tiles load during the epilogue of the previous one)
HEAD_GROUPS = [(16, None), (32, None), (64, None), (64, 1), (64, 3)]


@pytest.mark.parametrize("hg,wgs", HEAD_GROUPS, ids=[f"hg{h}_wgs{w}" for h, w in HEAD_GROUPS])
@pytest.mark.parametrize("name", ["edge", "t60_pitch", "t96_mixed", "t100_blocks128"])
@torch.inference_mode()
def test_head_groups(name, hg, wgs):
    require_gfx950()
    spec, kw = next((s, k) for n, s, k in CASES if n == name) if name != "edge" else (None, {})
    case = build(name, spec, kw)
    rows = case["q"].shape[0]
    out = launch(case, rows, mxfp8=False, hg=hg, wgs=wgs)
    check_attention(out, reference(case), name)
    codes, scales = launch(case, rows, mxfp8=True, hg=hg, wgs=wgs)
    rc, rs = R.epilogue_reference(out, case["positions"], case["cos_sin"])
    assert torch.equal(codes.view(torch.uint8), rc) and torch.equal(scales, rs), name
    again = launch(case, rows, mxfp8=True, hg=hg, wgs=wgs)
    assert torch.equal(again[0].view(torch.uint8), codes.view(torch.uint8)) and torch.equal(again[1], scales), name


@pytest.mark.parametrize("splits", [2, 3, 5])
@torch.inference_mode()
def test_splits(splits):
    require_gfx950()
    case = make_case(row_lengths([6, 6, 3, 1, 6, 6, 6, 2]) + [(7, 10, 20), (0, 0, 1)], pad=2, seed=3)
    rows = case["q"].shape[0]
    base = launch(case, rows, mxfp8=False)
    out = launch(case, rows, mxfp8=False, splits=splits)
    cos = torch.nn.functional.cosine_similarity(out.double().flatten(1), base.double().flatten(1), dim=1)
    live = base.flatten(1).abs().amax(dim=1) > 0
    assert not torch.isnan(out.float()).any() and cos[live].min() >= 0.99999 and (out[~live] == 0).all()
    assert torch.equal(launch(case, rows, mxfp8=False, splits=splits), out)  # the combine order is fixed
    assert torch.equal(launch(case, rows, mxfp8=False, splits=splits), out)
    codes, scales = launch(case, rows, mxfp8=True, splits=splits)
    rc, rs = R.epilogue_reference(out, case["positions"], case["cos_sin"])
    assert torch.equal(codes.view(torch.uint8), rc) and torch.equal(scales, rs)


@torch.inference_mode()
def test_one_term_p():
    """``p_terms=1`` (one e4m3 P): the numerics of an e4m3 P and fp32 sum (cosine ~0.9997 to the fp64 reference)."""
    require_gfx950()
    case = make_case(row_lengths([6] * 8), seed=5)
    rows = case["q"].shape[0]
    out = launch(case, rows, mxfp8=False, p_terms=1)
    ref = reference(case)
    cos = torch.nn.functional.cosine_similarity(out.double().flatten(1), ref.double().flatten(1), dim=1)
    assert 0.999 <= cos.min() < R.COS_MIN
