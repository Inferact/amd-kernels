# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The scorer + top-2048 against ``reference.py`` (torch logits) and, where AITER's scorer takes the cache layout, against
its logits bit for bit: served row counts and widths, short and empty rows, concentrated and tied scores."""

import pytest
import torch
from amd_kernels.indexer.mxfp4_paged_score_topk_d128_h32_top2048 import host
from amd_kernels.indexer.mxfp4_paged_score_topk_d128_h32_top2048.reference import (
    LOGIT_RTOL,
    TOPK,
    check_selection,
    logits_reference,
)

from .inputs import make_inputs, require_gfx950


def run(inputs, **kwargs):
    rows = inputs["q_payload"].shape[0]
    idx = torch.full((rows, TOPK), -7, dtype=torch.int32, device="cuda")
    val = torch.full((rows, TOPK), 3.0, dtype=torch.float32, device="cuda")
    host._score_topk(**inputs, out_indices=idx, out_scores=val, **kwargs)
    torch.cuda.synchronize()
    return idx, val


def reference_logits(inputs):
    payload, scales = host.cache_views(inputs["cache"])
    return logits_reference(
        inputs["q_payload"], inputs["q_scales"], inputs["weights"], payload, scales, inputs["block_table"],
        inputs["query_start_loc"], inputs["row_ends"],
    )


def aiter_logits(inputs):
    """AITER's row-group scorer's logits on the same operands, or None when it does not take the cache layout."""
    try:
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
            flydsl_pa_mqa_logits_fp4_rowgroup,
            make_fp4_mqa_plan,
        )

        payload, scales = host.cache_views(inputs["cache"])
        width = inputs["block_table"].shape[1] * 64
        rows = inputs["q_payload"].shape[0]
        plan = make_fp4_mqa_plan(
            num_seqs=inputs["query_start_loc"].numel() - 1, max_qlen=inputs["max_query_len"], num_rows=rows, heads=32,
            page_size=64, max_seq_len=width,
        )
        out = flydsl_pa_mqa_logits_fp4_rowgroup(
            plan, inputs["q_payload"], inputs["q_scales"], payload, scales, inputs["block_table"], inputs["weights"],
            inputs["query_start_loc"], inputs["row_ends"],
        )
    except (AssertionError, ImportError, TypeError):
        return None
    return [out[r, : int(inputs["row_ends"][r])] for r in range(rows)]


SHAPES = [  # (sequences, rows a sequence, width): the served decode batches at DCP8 widths
    (3, 4, 8192),
    (3, 4, 32768),
    (3, 4, 94208),
    (8, 5, 12288),
    (13, 5, 24576),
    (20, 4, 16384),
    (48, 4, 8192),
    (2, 1, 20480),
    (4, 3, 4096),
    (2, 6, 40960),
    (100, 1, 4096),  # more sequences than a wave's lanes
    (66, 2, 8192),
]


@pytest.mark.parametrize("seqs,qlen,width", SHAPES)
def test_matches_reference(seqs, qlen, width):
    require_gfx950()
    inputs = make_inputs(seqs, qlen, width, seed=seqs + width)
    idx, val = run(inputs)
    logits = reference_logits(inputs)
    check_selection(idx, val, logits, rtol=LOGIT_RTOL)
    exact = aiter_logits(inputs)
    if exact is not None:
        check_selection(idx, val, exact)


@pytest.mark.parametrize("seqs,qlen,width", [(2, 6, 15360), (13, 5, 24576), (3, 1, 8192)])
def test_uniform_batches(seqs, qlen, width):
    """``uniform=True`` (every sequence has ``max_query_len`` rows: the row ranges need no table read) selects the same."""
    require_gfx950()
    inputs = make_inputs(seqs, qlen, width, seed=seqs + width)
    idx, val = run(inputs, uniform=True)
    check_selection(idx, val, reference_logits(inputs), rtol=LOGIT_RTOL)


@pytest.mark.parametrize("length", [512, 1500, 4000, 20000, 20480, 40000, 64000, 131072])
def test_chunk_size_ladder(length):
    """One sequence per bound the chunk size steps at (512 .. 2048 keys a chunk, up to 64 chunks)."""
    require_gfx950()
    inputs = make_inputs(2, 4, max(64, -(-length // 64) * 64), lengths=[length, length // 2 + 1], seed=length)
    idx, val = run(inputs)
    check_selection(idx, val, reference_logits(inputs), rtol=LOGIT_RTOL)


HOT = [
    ((28672, 32768, 6),),  # all of the top in the last chunk
    ((0, 512, 5), (30000, 32000, 4)),
    ((4096, 20480, 3),),  # a third of the keys scored above the rest
]


@pytest.mark.parametrize("hot", HOT)
def test_concentrated_scores(hot):
    """Chunks holding more selected keys than their lists: the merge scores them again."""
    require_gfx950()
    inputs = make_inputs(3, 4, 32768, seed=5, hot=hot)
    idx, val = run(inputs)
    check_selection(idx, val, reference_logits(inputs), rtol=LOGIT_RTOL)


@pytest.mark.parametrize("hot", HOT)
def test_concentrated_scores_wide_batch(hot):
    """The same with whole rows a workgroup and half chunks a list (more than four sequences)."""
    require_gfx950()
    inputs = make_inputs(6, 4, 32768, seed=7, hot=hot)
    idx, val = run(inputs)
    check_selection(idx, val, reference_logits(inputs), rtol=LOGIT_RTOL)


@pytest.mark.parametrize("kind", ["tied", "hot"])
def test_repeated_calls_wide_batch(kind):
    require_gfx950()
    kw = dict(hot=((10000, 14000, 4),)) if kind == "hot" else {}
    inputs = make_inputs(7, 4, 24576, seed=4, positive_weights=kind == "tied", **kw)
    if kind == "tied":
        payload, scales = host.cache_views(inputs["cache"])
        payload.fill_(0x24)
        scales.fill_(121)
    first = run(inputs)
    for _ in range(20):
        idx, val = run(inputs)
        assert (idx >= 0).sum(1).tolist() == (first[0] >= 0).sum(1).tolist()
        assert torch.equal(val.sort(1).values, first[1].sort(1).values)
    check_selection(first[0], first[1], reference_logits(inputs), rtol=LOGIT_RTOL)


def test_short_empty_and_padding_rows():
    require_gfx950()
    inputs = make_inputs(5, 3, 8192, lengths=[1, 64, 2047, 2048, 8192], seed=2)
    inputs["row_ends"][0:2] = 0  # a sequence's first rows may see no key
    idx, val = run(inputs)
    check_selection(idx, val, reference_logits(inputs), rtol=LOGIT_RTOL)


def test_rows_past_the_last_sequence_are_padding():
    require_gfx950()
    inputs = make_inputs(3, 4, 16384, seed=9)
    rows = 12 + 5
    pad = lambda t, v=0: torch.cat((t, torch.full((5, *t.shape[1:]), v, dtype=t.dtype, device="cuda")))
    for name in ("q_payload", "q_scales", "weights"):
        inputs[name] = pad(inputs[name])
    inputs["row_ends"] = pad(inputs["row_ends"])
    idx, val = run(inputs)
    assert idx.shape[0] == rows
    assert (idx[12:] == -1).all() and (val[12:] == float("-inf")).all()
    logits = reference_logits({**inputs, "row_ends": inputs["row_ends"][:12]})
    check_selection(idx[:12], val[:12], logits, rtol=LOGIT_RTOL)


def test_tied_scores():
    """Every key of a run scores the same (identical key bytes): ties at the 2048th score are all valid choices."""
    require_gfx950()
    inputs = make_inputs(2, 2, 16384, lengths=[16384, 16000], seed=4, positive_weights=True)
    payload, scales = host.cache_views(inputs["cache"])
    payload.fill_(0x24)
    scales.fill_(121)
    idx, val = run(inputs)
    check_selection(idx, val, reference_logits(inputs), rtol=LOGIT_RTOL)
    for row in range(val.shape[0]):
        live = val[row][idx[row] >= 0]
        assert (live == live[0]).all()


def test_negative_scores():
    require_gfx950()
    inputs = make_inputs(3, 4, 16384, seed=6)
    inputs["weights"] = -inputs["weights"].abs()
    idx, val = run(inputs)
    check_selection(idx, val, reference_logits(inputs), rtol=LOGIT_RTOL)


@pytest.mark.parametrize("kind", ["tied", "hot", "random"])
def test_repeated_calls_are_stable(kind):
    """Back-to-back launches (the merge workgroups wait on counts the chunk workgroups leave; LDS and scratch hold the
    last call's data) all give the same selection."""
    require_gfx950()
    kw = dict(hot=((10000, 14000, 4),)) if kind == "hot" else {}
    inputs = make_inputs(2, 2, 16384, lengths=[16384, 16000], seed=4, positive_weights=kind == "tied", **kw)
    if kind == "tied":
        payload, scales = host.cache_views(inputs["cache"])
        payload.fill_(0x24)
        scales.fill_(121)
    first = run(inputs)
    for _ in range(40):
        idx, val = run(inputs)
        assert (idx >= 0).sum(1).tolist() == (first[0] >= 0).sum(1).tolist()
        assert torch.equal(val.sort(1).values, first[1].sort(1).values)
    check_selection(first[0], first[1], reference_logits(inputs), rtol=LOGIT_RTOL)


def test_two_streams_do_not_share_scratch():
    """Launches overlapping on two streams (the scratch is a set a stream) select what each would alone."""
    require_gfx950()
    ins = [make_inputs(3, 4, 16384, seed=11), make_inputs(5, 3, 24576, seed=12)]
    outs = [(torch.empty(i["q_payload"].shape[0], TOPK, dtype=torch.int32, device="cuda"),
             torch.empty(i["q_payload"].shape[0], TOPK, dtype=torch.float32, device="cuda")) for i in ins]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    for _ in range(10):
        for inp, (idx, val), stream in zip(ins, outs, streams):
            with torch.cuda.stream(stream):
                host._score_topk(**inp, out_indices=idx, out_scores=val)
        torch.cuda.synchronize()
        for inp, (idx, val) in zip(ins, outs):
            check_selection(idx, val, reference_logits(inp), rtol=LOGIT_RTOL)


def test_deterministic():
    require_gfx950()
    inputs = make_inputs(6, 4, 24576, seed=8, hot=((10000, 14000, 4),))
    first = run(inputs)
    for _ in range(3):
        again = run(inputs)
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
