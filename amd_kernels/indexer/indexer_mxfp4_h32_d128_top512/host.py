# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the MXFP4 indexer: argument checks, workspace, the shape -> kernel dispatch, the warm-up."""

import functools
from typing import NamedTuple

import torch

from amd_kernels.warm_up import entry

from .common import BLOCK, CANDIDATES, ENTRY_BYTES, H, TOPK
from .kernel_blocks import PLAIN_SLOTS, SEGB, SOURCE_SLOTS, compile_block_select
from .kernel_cand_logits import compile_candidate_logits
from .kernel_scores import compile_block_scores
from .kernel_select import compile_select
from .kernel_tokens import compile_tokens_select

MAX_ROWS = 2048  # rows of a step the ragged lists cover
MAX_ENTRIES = 1 << 20  # entries per row the entries support
BLOCK_CTX_MAX = 262144  # entries per row of the one-workgroup block select; above, the long path (kernel_blocks)
MAX_CTX = MAX_ENTRIES
SHORT_ENTRIES = 262144  # the slot table of the gathered token select up to this many entries
TIERS = (4096, 8192, 16384, 32768, 65536, 131072)  # blocks of a row's context
SEG_GRID = 1024
LONG_TILES = 8192  # tiles of 32 entries: 262144 entries
# score workgroups (requests x segments) a step stays within: 4 a CU is one resident wave of the walk (218 VGPRs: 2
# waves a SIMD, 2-wave workgroups); a few more would leave a partial second wave. Segments of 8-128 tiles of 32
# entries.
SELECT_THREADS = 512  # the dense select's workgroup (two rows a CU)
GATHER_SELECT_THREADS = 1024  # the candidate consumer's gather select
# Short contexts: the walk writes every token's logit and a small workgroup a row selects from them (kernel_tokens);
# contexts up to max(TOKEN_TIERS) entries use it (dense layers, with or without the candidate list). A tier is the
# logits row width and its select workgroup's threads.
TOKEN_TIERS = {4096: 256, 8192: 512, 16384: 512}
TOKEN_MAX_CTX = max(TOKEN_TIERS)
# The candidate consumer of a few rows: one workgroup a row (the gather select) leaves most CUs idle and is VALU
# bound on its one CU; up to CAND_SPLIT_ROWS rows the candidates are scored by (row, segment of CAND_TPS tiles)
# workgroups into token logits and the token select picks the 512 best from them. The split costs about the same at
# every context (25 us at 18 rows, 31 at 66, 38 at 126); the gather select grows with the context (24 us at 4k, 30
# at 8k, 34 at 12k, 38 at 16k, 44 at 130k, 18 rows): split when the context is above CAND_SPLIT_CTX0 +
# CAND_SPLIT_CTX_ROW * rows.
CAND_SPLIT_ROWS = 160
CAND_TPS = 16
CAND_SPLIT_CTX0 = 3000
CAND_SPLIT_CTX_ROW = 110


class Ragged(NamedTuple):
    """The sparse attention's input lists, written by the select (``reference.ragged_reference``): ``slots``
    [rows * 512]
    int32 (row r's slots at ``indptr[r]``: ``slot_block_table[request][position // B] * B + position % B``, B =
    ``slot_block_size`` entries per page of the attention cache, 64 / 128 / 256), ``indptr`` [rows + 1], ``lens`` [rows]
    (min(512, row end) where ``is_valid`` [rows] int32 is set, else 0: a candidate list holds all its row's blocks, or
    2048 of them); ``slot_block_table`` [requests, pages] int32."""

    slots: torch.Tensor
    indptr: torch.Tensor
    lens: torch.Tensor
    is_valid: torch.Tensor
    slot_block_table: torch.Tensor
    slot_block_size: int


def _ragged_args(ragged: "Ragged | None", rows: int):
    if ragged is None:
        return (0, 0, 0, 0, 0, 0), 0
    r = ragged
    assert r.slot_block_size in (64, 128, 256), r.slot_block_size
    assert r.slots.numel() >= rows * TOPK and r.slots.dtype == torch.int32 and r.slots.is_contiguous()
    assert r.indptr.shape == (rows + 1,) and r.indptr.dtype == torch.int32 and r.lens.shape == (rows,)
    assert r.is_valid.shape == (rows,) and r.is_valid.dtype == torch.int32 and r.slot_block_table.dtype == torch.int32
    assert r.slot_block_table.stride(1) == 1 and rows <= MAX_ROWS
    return (
        r.slots.data_ptr(),
        r.indptr.data_ptr(),
        r.lens.data_ptr(),
        r.is_valid.data_ptr(),
        r.slot_block_table.data_ptr(),
        r.slot_block_table.stride(0),
    ), r.slot_block_size.bit_length() - 1


def slot_rows(rows: int) -> int:
    return 1024 * -(-rows // 1024)  # a multiple of every select's threads


def groups(reqs: int, next_n: int) -> int:
    """Workgroups of the select: a request's rows on one XCD (workgroup w runs on XCD w % 8), requests round robin."""
    return 8 * next_n * (-(-reqs // 8))


def tokens_tier(max_ctx: int) -> int | None:
    """The token-logit tier a dense call of ``max_ctx`` entries uses (None: the block path)."""
    return next((t for t in sorted(TOKEN_TIERS) if t >= max_ctx), None)


def blocks_tier(max_ctx: int) -> int:
    need = -(-max_ctx // BLOCK)
    return next(t for t in TIERS if t >= need)


def blocker(page_entries: int, max_ctx: int, next_n: int) -> str | None:
    """Why the entries cannot serve a layer (None: they can)."""
    from . import available

    if not available():
        return "needs gfx950 with FlyDSL"
    if page_entries not in (64, 128):
        return f"{page_entries} entries per page (64 or 128)"
    if not 1 <= next_n <= 8:
        return f"{next_n} rows per request (1-8)"
    if max_ctx > MAX_ENTRIES:
        return f"context {max_ctx} entries (at most {MAX_ENTRIES})"
    return None


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _scores_exe(next_n: int, page_entries: int, tokens: bool = False):
    return compile_block_scores(next_n, page_entries, tokens)


@functools.cache
def _tokens_exe(
    next_n: int,
    tokens: int,
    slot_log2: int,
    slot_rows: int,
    emit_candidates: bool = False,
    gathered: bool = False,
    max_position: int = SHORT_ENTRIES,
    min_end: int = 0,
):
    return compile_tokens_select(
        next_n, tokens, TOKEN_TIERS[tokens], slot_log2, slot_rows, emit_candidates, gathered, max_position, min_end
    )


@functools.cache
def _blocks_exe(mode: str, k: int, newest: bool, min_end: int, merged: int = 4096):
    return compile_block_select(mode, k, newest, min_end, merged)


@functools.cache
def _compact(device: torch.device, rows: int, entries: int):
    """The long path's compact lists (ids, maxima) [rows, entries]: slots no segment writes stay -1 / -inf."""
    return (
        torch.full((rows, entries), -1, dtype=torch.int32, device=device),
        torch.full((rows, entries), float("-inf"), dtype=torch.float32, device=device),
    )


@functools.cache
def _aux_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device)


@functools.cache
def _cand_buffer(device: torch.device, rows: int) -> torch.Tensor:
    """The long path's candidate blocks [rows, 2048] int32, one per (device, rows): launches on one stream."""
    return torch.empty(rows, CANDIDATES, dtype=torch.int32, device=device)


@functools.cache
def _cand_logits_exe(next_n: int, page_entries: int, min_end: int = 0, cand_blocks: int = CANDIDATES):
    return compile_candidate_logits(next_n, page_entries, CAND_TPS, min_end, cand_blocks)


@functools.cache
def _select_exe(
    next_n: int,
    page_entries: int,
    blocks: int,
    gather: bool,
    emit_candidates: bool = False,
    slot_log2: int = 0,
    slot_rows: int = SELECT_THREADS,
    skip_long: bool = False,
):
    return compile_select(
        next_n,
        page_entries,
        blocks,
        gather,
        GATHER_SELECT_THREADS if gather else SELECT_THREADS,
        emit_candidates=emit_candidates,
        slot_log2=slot_log2,
        slot_rows=slot_rows,
        skip_long=skip_long,
    )


@functools.cache
def _workspace(device: torch.device, rows: int, blocks: int, tokens: bool = False) -> torch.Tensor:
    """Block maxima [rows, blocks + 64] fp32 (``tokens``: token logits [rows, tokens + 64]), one per (device, rows,
    tier): launches on one stream."""
    return torch.empty(rows, blocks + 64, dtype=torch.float32, device=device)


def _page_entries(kv_cache: torch.Tensor) -> int:
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == ENTRY_BYTES, kv_cache.shape
    return kv_cache.shape[1]


def _check(q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n):
    rows = q.shape[0]
    assert q.shape == (rows, H, 64) and q.dtype == torch.uint8 and q.is_contiguous()
    assert q_scale.is_contiguous() and q_scale.numel() * q_scale.element_size() == rows * H * 4
    assert weights.shape == (rows, H) and weights.dtype == torch.float32 and weights.is_contiguous()
    assert kv_cache.stride(-1) == 1 and kv_cache.stride(-2) == ENTRY_BYTES and kv_cache.stride(0) % 16 == 0
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert rows == block_table.shape[0] * next_n, (rows, block_table.shape, next_n)
    assert row_ends.shape == (rows,) and row_ends.dtype == torch.int32 and row_ends.is_contiguous()
    assert out.shape == (rows, TOPK) and out.dtype == torch.int32 and out.is_contiguous()
    return rows


def _indexer_block_scores(
    q,
    q_scale,
    weights,
    kv_cache,
    block_table,
    row_ends,
    *,
    next_n: int,
    max_ctx: int,
    scores: torch.Tensor | None = None,
    tokens: bool = False,
) -> torch.Tensor:
    """The maximum logit of each 8-token block of every decode row: ``scores`` [rows, tier + 64] fp32 (block j of a
    row in column j, columns past the row's blocks unspecified, a masked block -inf); allocated per (rows, tier) when
    not given."""
    rows = q.shape[0]
    P = _page_entries(kv_cache)
    assert blocker(P, max_ctx, next_n) is None, blocker(P, max_ctx, next_n)
    nb = tokens_tier(max_ctx) if tokens else blocks_tier(max_ctx)
    if scores is None:
        scores = _workspace(q.device, rows, nb, tokens)
    assert scores.shape[0] >= rows and scores.shape[1] >= nb + 64 and scores.dtype == torch.float32
    tiles = -(-max_ctx // 32)
    nseg = max(-(-tiles // 128), min(tiles // 8, SEG_GRID // (rows // next_n))) or 1
    if tiles > LONG_TILES:  # a long request among short ones: finer segments (64 tiles) balance the walk
        nseg = max(nseg, -(-tiles // 64))
    ptrs = (
        q.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        row_ends.data_ptr(),
        scores.data_ptr(),
    )
    strides = (kv_cache.stride(0), block_table.stride(0), scores.stride(0))
    _run(_scores_exe(next_n, P, tokens), *ptrs, *strides, nseg, rows // next_n, torch.cuda.current_stream())
    return scores


def _indexer_topk(
    q,
    q_scale,
    weights,
    kv_cache,
    block_table,
    row_ends,
    out,
    *,
    next_n: int,
    max_ctx: int,
    candidates_out: torch.Tensor | None = None,
    ragged: Ragged | None = None,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """The 512 best keys of every decode row of the MXFP4 index cache, into ``out`` [rows, 512] int32: positions
    ascending, -1 padded; ties to the lower position (``reference.py``). ``max_ctx``: an upper bound of the entries of
    a row, at most ``MAX_ENTRIES`` (above 262144 the rows longer than that take the long path, ``kernel_blocks``). ``candidates_out`` [rows, 2048] int32 (the candidate source layer): also the row's 2048 candidate blocks, the
    8-token block of its newest key and the 2047 best others by block maximum, ids ascending, -1 padded; ``ragged``:
    also the sparse attention's slot lists (``Ragged``). CUDA-graph
    safe: lengths are read on the device, nothing is allocated when ``scores`` (the block maxima [rows, tier + 64] fp32)
    is given or the workspace exists (``warmup_indexer_topk``)."""
    rows = _check(q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n)
    P = _page_entries(kv_cache)
    nb = blocks_tier(max_ctx)
    ntok = tokens_tier(max_ctx)
    if ntok is not None:
        if candidates_out is not None:
            assert candidates_out.shape == (rows, CANDIDATES) and candidates_out.dtype == torch.int32
            assert candidates_out.is_contiguous()
        return _topk_from_tokens(
            q, q_scale, weights, kv_cache, block_table, row_ends, out, ntok, next_n, max_ctx, ragged, rows, P,
            candidates_out
        )
    long = max_ctx > BLOCK_CTX_MAX  # rows above the block select's tier go to the long path (kernel_blocks)
    scores = _indexer_block_scores(
        q, q_scale, weights, kv_cache, block_table, row_ends, next_n=next_n, max_ctx=max_ctx, scores=scores
    )
    stream = torch.cuda.current_stream()
    if long:
        nb = blocks_tier(BLOCK_CTX_MAX)
        aux = _aux_stream(q.device)  # the long rows' chain runs beside the block select of the short rows
        aux.wait_stream(stream)
        with torch.cuda.stream(aux):
            _topk_long(
                q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n, max_ctx, candidates_out, ragged, rows,
                P, scores
            )
    cand = candidates_out
    if cand is not None:
        assert cand.shape == (rows, CANDIDATES) and cand.dtype == torch.int32 and cand.is_contiguous()
    rargs, slot_log2 = _ragged_args(ragged, rows)
    _run(
        _select_exe(next_n, P, nb, False, cand is not None, slot_log2, slot_rows(rows), long),
        scores.data_ptr(),
        0,
        0 if cand is None else cand.data_ptr(),
        q.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        row_ends.data_ptr(),
        out.data_ptr(),
        kv_cache.stride(0),
        block_table.stride(0),
        scores.stride(0),
        rows // next_n,
        *rargs,
        groups(rows // next_n, next_n),
        stream,
    )
    if long:
        stream.wait_stream(aux)
    return out


def _topk_from_tokens(
    q, q_scale, weights, kv_cache, block_table, row_ends, out, ntok, next_n, max_ctx, ragged, rows, P, cand=None
):
    """Short rows: the walk writes the token logits, a workgroup a row selects the 512 best."""
    logits = _indexer_block_scores(
        q, q_scale, weights, kv_cache, block_table, row_ends, next_n=next_n, max_ctx=max_ctx, tokens=True
    )
    rargs, slot_log2 = _ragged_args(ragged, rows)
    _run(
        _tokens_exe(next_n, ntok, slot_log2, slot_rows(rows), cand is not None),
        logits.data_ptr(),
        row_ends.data_ptr(),
        out.data_ptr(),
        logits.stride(0),
        rows // next_n,
        *rargs,
        0 if cand is None else cand.data_ptr(),
        rows,
        torch.cuda.current_stream(),
    )
    return out


def _topk_long(
    q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n, max_ctx, candidates_out, ragged, rows, P, scores
):
    """Rows above the block select's entries (it left them alone): the best blocks from the walk's block maxima
    (``kernel_blocks``: 512, or the candidate list of the source layer), then the candidate consumer's scorer and token
    select, also for those rows only."""
    newest = candidates_out is not None
    if newest:
        assert candidates_out.shape == (rows, CANDIDATES) and candidates_out.dtype == torch.int32
        assert candidates_out.is_contiguous()
    cand = candidates_out if newest else _cand_buffer(q.device, rows)
    split_end = blocks_tier(BLOCK_CTX_MAX) * BLOCK
    segments = max(1, blocks_tier(max_ctx) // SEGB)
    slots = SOURCE_SLOTS if newest else PLAIN_SLOTS
    merged = max(4096, segments * slots)
    ids2, vals2 = _compact(q.device, rows, merged)
    stream = torch.cuda.current_stream()
    _run(
        _blocks_exe("seg", CANDIDATES - 1 if newest else TOPK, newest, split_end),
        scores.data_ptr(),
        ids2.data_ptr(),
        vals2.data_ptr(),
        row_ends.data_ptr(),
        0,
        scores.stride(0),
        merged,
        0,
        rows,
        segments,
        stream,
    )
    _run(
        _blocks_exe("merge", CANDIDATES if newest else TOPK, newest, split_end, merged),
        vals2.data_ptr(),
        ids2.data_ptr(),
        0,
        row_ends.data_ptr(),
        cand.data_ptr(),
        merged,
        merged,
        segments * slots,
        rows,
        1,
        stream,
    )
    rargs, slot_log2 = _ragged_args(ragged, rows)
    _select_from_candidates(
        q, q_scale, weights, kv_cache, block_table, row_ends, cand, out, next_n, P, rows, rargs, slot_log2, max_ctx,
        split_end, CANDIDATES if newest else TOPK
    )


def _select_from_candidates(
    q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out, next_n, P, rows, rargs, slot_log2, max_ctx,
    min_end=0, cand_blocks=CANDIDATES
):
    """The 512 best tokens among the candidate blocks' tokens: ``kernel_cand_logits`` scores them on (row, segment)
    workgroups, ``kernel_tokens`` (gathered) selects; rows of at most ``min_end`` entries are left alone."""
    logits = _workspace(q.device, rows, cand_blocks * BLOCK, True)
    stream = torch.cuda.current_stream()
    _run(
        _cand_logits_exe(next_n, P, min_end, cand_blocks),
        q.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        row_ends.data_ptr(),
        candidates.data_ptr(),
        logits.data_ptr(),
        kv_cache.stride(0),
        block_table.stride(0),
        logits.stride(0),
        rows,
        stream,
    )
    _run(
        _tokens_exe(
            next_n, cand_blocks * BLOCK, slot_log2, slot_rows(rows), False, True,
            MAX_ENTRIES if max_ctx > SHORT_ENTRIES else SHORT_ENTRIES, min_end,
        ),
        logits.data_ptr(),
        row_ends.data_ptr(),
        out.data_ptr(),
        logits.stride(0),
        rows // next_n,
        *rargs,
        candidates.data_ptr(),
        rows,
        stream,
    )


def _indexer_candidate_topk(
    q,
    q_scale,
    weights,
    kv_cache,
    block_table,
    row_ends,
    candidates,
    out,
    *,
    next_n: int,
    max_ctx: int,
    ragged: Ragged | None = None,
) -> torch.Tensor:
    """The 512 best keys among the tokens of each row's candidate blocks (``candidates`` [rows, 2048] int32: 8-token
    block ids ascending, -1 padded), same outputs as ``indexer_topk``."""
    rows = _check(q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n)
    P = _page_entries(kv_cache)
    assert blocker(P, max_ctx, next_n) is None, blocker(P, max_ctx, next_n)
    assert candidates.shape == (rows, CANDIDATES) and candidates.dtype == torch.int32 and candidates.is_contiguous()
    rargs, slot_log2 = _ragged_args(ragged, rows)
    from amd_kernels import warm_up

    warming = warm_up._in_warm_up.get()  # the warm-up (one size) compiles both variants
    long = max_ctx > BLOCK_CTX_MAX  # the gather select's tables do not fit: always the split path
    split = long or (rows <= CAND_SPLIT_ROWS and max_ctx >= CAND_SPLIT_CTX0 + CAND_SPLIT_CTX_ROW * rows)
    if split or warming:
        _select_from_candidates(
            q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out, next_n, P, rows, rargs, slot_log2,
            max_ctx
        )
        if not warming or long:
            return out
    _run(
        _select_exe(next_n, P, blocks_tier(max_ctx), True, False, slot_log2, slot_rows(rows)),
        0,
        candidates.data_ptr(),
        0,
        q.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        row_ends.data_ptr(),
        out.data_ptr(),
        kv_cache.stride(0),
        block_table.stride(0),
        0,
        rows // next_n,
        *rargs,
        groups(rows // next_n, next_n),
        torch.cuda.current_stream(),
    )
    return out


def _mode(
    q, q_scale, weights, kv_cache, block_table, row_ends, *rest, next_n, max_ctx, candidates_out=None, ragged=None, **kw
):
    """What a call fixes about its kernels besides the row count: rows per request, entries per page, block tier, the
    candidate list, the ragged lists' page size."""
    return (
        next_n,
        _page_entries(kv_cache),
        blocks_tier(max_ctx),
        tokens_tier(max_ctx),
        bool(kw.get("tokens", False)),
        candidates_out is not None,
        0 if ragged is None else ragged.slot_block_size,
    )


def _sizes(mode):
    return (mode[0],)


def _zeros(n, next_n, kv_cache, block_table, device):
    return (
        torch.zeros(n, H, 64, dtype=torch.uint8, device=device),
        torch.zeros(n, H, 4, dtype=torch.uint8, device=device),
        torch.zeros(n, H, dtype=torch.float32, device=device),
        kv_cache,
        torch.zeros(n // next_n, block_table.shape[1], dtype=torch.int32, device=device),
        torch.zeros(n, dtype=torch.int32, device=device),
    )


def _inputs_scores(
    mode, n, q, q_scale, weights, kv_cache, block_table, row_ends, *, next_n, max_ctx, tokens=False, scores=None
):
    return _zeros(n, next_n, kv_cache, block_table, q.device), {"next_n": next_n, "max_ctx": max_ctx, "tokens": tokens}


def _zero_ragged(n, ragged, device):
    return Ragged(
        torch.zeros(n * TOPK, dtype=torch.int32, device=device),
        torch.zeros(n + 1, dtype=torch.int32, device=device),
        torch.zeros(n, dtype=torch.int32, device=device),
        torch.zeros(n, dtype=torch.int32, device=device),
        ragged.slot_block_table,
        ragged.slot_block_size,
    )


def _inputs(
    mode,
    n,
    q,
    q_scale,
    weights,
    kv_cache,
    block_table,
    row_ends,
    out,
    *,
    next_n,
    max_ctx,
    candidates_out=None,
    ragged=None,
    scores=None,
):
    kw = {"next_n": next_n, "max_ctx": max_ctx}
    if candidates_out is not None:
        kw["candidates_out"] = torch.zeros(n, CANDIDATES, dtype=torch.int32, device=q.device)
    if ragged is not None:
        kw["ragged"] = _zero_ragged(n, ragged, q.device)
    return _zeros(n, next_n, kv_cache, block_table, q.device) + (
        torch.zeros(n, TOPK, dtype=torch.int32, device=q.device),
    ), kw


def _inputs_candidates(
    mode, n, q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out, *, next_n, max_ctx, ragged=None
):
    kw = {"next_n": next_n, "max_ctx": max_ctx}
    if ragged is not None:
        kw["ragged"] = _zero_ragged(n, ragged, q.device)
    return (
        _zeros(n, next_n, kv_cache, block_table, q.device)
        + (
            torch.full((n, CANDIDATES), -1, dtype=torch.int32, device=q.device),
            torch.zeros(n, TOPK, dtype=torch.int32, device=q.device),
        )
    ), kw


indexer_block_scores = entry(
    "indexer/indexer_mxfp4_h32_d128_top512_block_scores", mode=_mode, sizes=_sizes, inputs=_inputs_scores
)(_indexer_block_scores)
indexer_topk = entry("indexer/indexer_mxfp4_h32_d128_top512", mode=_mode, sizes=_sizes, inputs=_inputs)(_indexer_topk)
indexer_candidate_topk = entry(
    "indexer/indexer_mxfp4_h32_d128_top512_candidates", mode=_mode, sizes=_sizes, inputs=_inputs_candidates
)(_indexer_candidate_topk)
