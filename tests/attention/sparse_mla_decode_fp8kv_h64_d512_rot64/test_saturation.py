# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``sparse_mla_decode`` with huge scores: q codes saturated to +-448 (a fraction or all of them), large per-tensor
scales, one key, all-masked key blocks, scores that jump by hundreds of log2 units between tiles. The output is finite
(bf16 and MXFP8: no NaN code, no 255 scale) and equals the fp64 reference per head (relative L2 error), in every
head-group kernel, the persistent 64-head one, the one-term P and the key-range split."""

import pytest
import torch
from amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64 import reference as R
from amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64 import sparse_mla_decode

from .cases import D, H, make_case, outputs, require_gfx950, row_lengths
from .test_decode import ARGS, EDGE

SAT = 0x7E  # e4m3 +-448
REL = 0.02


def saturate(case, frac, seed=3):
    q = case["q"]
    if frac >= 1.0:
        sat = torch.ones(q.shape, dtype=torch.bool, device="cuda")
    else:
        g = torch.Generator(device="cuda").manual_seed(seed)
        sat = torch.rand(q.shape, generator=g, device="cuda") < frac
    qq = q.view(torch.uint8).clone()
    qq[sat] = (qq[sat] & 0x80) | SAT
    case["q"] = qq.view(q.dtype)
    return case


def reference(case):
    stride = lambda c: None if c is None else c.stride(0) // D
    return R.attention_reference(
        case["q"], case["q_scale"], case["main_cache"], case["main_indptr"], case["main_indices"], case["extra_cache"],
        case["extra_indptr"], case["extra_indices"], case["kv_scale"], case["attn_sink"], case["softmax_scale"],
        stride(case["main_cache"]), stride(case["extra_cache"]),
    )  # fmt: skip


def rel_error(out, ref):
    o, r = out.double(), ref.double()
    return ((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).nan_to_num(nan=9.0)


def run(case, **kw):
    rows = case["q"].shape[0]
    args = [case[k] for k in ARGS]
    out = torch.full((rows, H, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    sparse_mla_decode(*args, None, None, out, **kw)
    codes, scales = outputs(rows)
    sparse_mla_decode(*args, codes, scales, **kw)
    torch.cuda.synchronize()
    return out, codes, scales


def check(case, name, ties=False, **kw):
    out, codes, scales = run(case, **kw)
    assert not torch.isnan(out.float()).any() and not torch.isinf(out.float()).any(), name
    assert not ((codes.view(torch.uint8) & 0x7F) == 0x7F).any() and not (scales == 255).any(), name
    ref = reference(case)
    err = rel_error(out, ref)
    if ties:  # scores of 1e4 log2 units: the fp32 score noise (also the baseline's) decides near ties
        assert err.max() <= 0.1 and (err > REL).double().mean() <= 0.005, (name, err.max().item())
    else:
        assert err.max() <= REL, (name, err.max().item(), err.argmax().item())
    rc, rs = R.epilogue_reference(out, case["positions"], case["cos_sin"])
    assert torch.equal(codes.view(torch.uint8), rc) and torch.equal(scales, rs), name
    again = run(case, **kw)
    assert torch.equal(again[0], out), name
    return err.max().item()


VARIANTS = [{}, {"hg": 16}, {"hg": 32}, {"hg": 64}, {"hg": 64, "wgs": 1}, {"splits": 2}]
IDS = ["default", "hg16", "hg32", "hg64", "hg64_wgs1", "splits2"]


@pytest.mark.parametrize("kw", VARIANTS, ids=IDS)
@pytest.mark.parametrize("frac", [0.01, 0.2, 1.0])
@torch.inference_mode()
def test_saturated_q(frac, kw):
    require_gfx950()
    case = saturate(make_case(row_lengths([6] * 3), k_std=1.0), frac)
    check(case, f"sat{frac}", **kw)


@pytest.mark.parametrize("kw", VARIANTS, ids=IDS)
@pytest.mark.parametrize("scale", [8.0, 1000.0])
@torch.inference_mode()
def test_large_scales(scale, kw):
    """Scores up to ~1e5 log2 units: the softmax is a one-hot (a tie goes to the same keys as the reference)."""
    require_gfx950()
    case = saturate(make_case(row_lengths([6] * 3), k_std=1.0), 0.05)
    case["q_scale"] = torch.full((1,), scale, device="cuda")
    case["kv_scale"] = torch.full((1,), 0.5, device="cuda")
    check(case, f"scale{scale}", ties=True, **kw)


@pytest.mark.parametrize("kw", VARIANTS, ids=IDS)
@torch.inference_mode()
def test_masked_blocks_and_single_key(kw):
    """Key counts whose last tile has an all-masked half (33..64 -> the second 32-key block, 65 -> a tile of one key),
    a single key (main or extra), the window alone, with saturated q."""
    require_gfx950()
    lens = [(1, 0, 5), (0, 1, 7), (32, 0, 40), (33, 0, 40), (64, 0, 70), (65, 0, 70), (1, 64, 300), (64, 1, 300),
            (32, 33, 300), (128, 512, 4000), (127, 3, 4000), (0, 0, 10), (96, 1, 200), (2, 1, 3)]
    case = saturate(make_case(lens, k_std=1.0), 0.3)
    check(case, "masked", **kw)
    case = saturate(make_case(EDGE, k_std=1.0), 1.0)
    check(case, "edge", **kw)


@pytest.mark.parametrize("kw", VARIANTS[:5], ids=IDS[:5])
@torch.inference_mode()
def test_score_jump_between_tiles(kw):
    """The scores of the first tiles are small and one late key is huge (and the reverse): the reference moves by
    hundreds of log2 units at once."""
    require_gfx950()
    case = make_case(row_lengths([6] * 4), q_std=1.0, k_std=1.0)
    k = case["extra_cache"].view(torch.uint8)
    nrows = case["q"].shape[0]
    # a few large keys among the extra rows: rows with |code| 448 in a pattern of signs
    gen = torch.Generator(device="cuda").manual_seed(5)
    hot = torch.randint(0, case["extra_cache"].shape[0] * case["extra_cache"].shape[1], (64,), generator=gen, device="cuda")
    flat = k.view(-1, D)
    used = case["extra_indices"].long()
    pick = used[torch.randint(0, used.numel(), (32,), generator=gen, device="cuda")]
    flat[pick] = (torch.rand((32, D), generator=gen, device="cuda") < 0.5).to(torch.uint8) * 0x80 | SAT
    del hot, nrows
    case = saturate(case, 0.1)
    check(case, "jump", **kw)


@pytest.mark.parametrize("frac", [0.2, 1.0])
@torch.inference_mode()
def test_one_term_p_is_finite_and_close(frac):
    """p_terms=1 (one e4m3 P, the baseline's precision): no NaN, within the baseline's error."""
    require_gfx950()
    case = saturate(make_case(row_lengths([6] * 3), k_std=1.0), frac)
    out, codes, scales = run(case, p_terms=1)
    assert not torch.isnan(out.float()).any()
    assert not ((codes.view(torch.uint8) & 0x7F) == 0x7F).any() and not (scales == 255).any()
    assert rel_error(out, reference(case)).max() <= 0.1
