# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile caches and launch wrappers of the NVFP4 decode index scorers."""

import functools
from amd_kernels.warm_up import entry, launcher

import torch

from ..score_prefill.host import nvfp4_e4m3_lut

from . import kernel as _kernel
from ..host_stream import current_stream as _current_stream

# waves to aim for (3 per SIMD at 158 VGPRs): swept 1024-8192, cold L2, 16-100 requests: best or within 1 us everywhere
TARGET_WAVES = 3072


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _end_bytes(t: torch.Tensor) -> int:
    """Bytes from the data pointer to the end of the last element."""
    return (sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1) * t.element_size()


# Scores go through LDS units and leave as whole rows from this footprint (scattered dword stores into 32 MB cost 17%
# at 16 rows x 512 requests x 128K, into 8 MB nothing), if the units still balance over the wave slots.
BUFFER_MIN_BYTES = 8 << 20
UNITS_PER_SLOT = 4


def _unit(max_blocks: int, slots: int, umax: int, footprint: int) -> int:
    """Consecutive blocks per unit (a power of two, 4..umax) such that every wave slot gets >= UNITS_PER_SLOT units,
    or 0: single strided positions and direct stores."""
    if footprint < BUFFER_MIN_BYTES:
        return 0
    u = umax
    while u > 4 and _cdiv(max_blocks, u) < UNITS_PER_SLOT * slots:
        u //= 2
    return u if _cdiv(max_blocks, u) >= UNITS_PER_SLOT * slots else 0


def _run_compiled(exe, *args, waves_per_eu: int | None = None):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction.
    ``waves_per_eu``: the occupancy the register allocator must leave room for (``--amdgpu-waves-per-eu``)."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        compile = flyc.compile[{"waves_per_eu": waves_per_eu}] if waves_per_eu else flyc.compile
        exe._cf = compile(exe, *args)
    else:
        cf(*args)


@functools.cache
@launcher
def _get_kernel(rows: int, heads: int, buffered: bool):
    return _kernel.compile_score_decode(rows, heads, buffered)


@functools.cache
@launcher
def _get_kernel_devpos(rows: int, heads: int):
    return _kernel.compile_score_decode(rows, heads, device_positions=True)


def nvfp4_index_score_decode(
    q_idx: torch.Tensor,  # [num_reqs * query_len, H, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, num_reqs * query_len, >= cdiv(max_seq_len, 128)] fp32
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    query_len: int,
    max_seq_len: int,
) -> None:
    """The decode index scores (``reference.decode_scores_reference``) for blocks below ``cdiv(seq_len, 128)``, fp8
    keys."""
    total_q, num_heads, _ = q_idx.shape
    num_reqs = total_q // query_len
    if num_reqs == 0:
        return
    rows = num_heads * query_len
    max_blocks = max(1, _cdiv(max_seq_len, _kernel.BLK))
    # Chunk size from the work: num_reqs * max_blocks pages over ~TARGET_WAVES waves. A wave of a shared position gets
    # chunk / NW of its pages, so a chunk of NW * (pages per wave) requests; with at most one page per wave, chunks of
    # one request, each wave on its own positions.
    pages_per_wave = _cdiv(num_reqs * max_blocks, TARGET_WAVES)
    s_bytes = _end_bytes(score)
    assert s_bytes < 2**31 * 4, "score offsets must fit in int32 elements"
    footprint = num_reqs * rows * max_blocks * 4
    chunk = 1 if pages_per_wave <= 1 else _kernel.NW * pages_per_wave
    chunk_cap = _kernel.chunk_max(_kernel.padded_rows(rows))
    chunk = max(1, min(chunk, chunk_cap, num_reqs))
    nch = _cdiv(num_reqs, chunk)
    chunk = _cdiv(num_reqs, nch)
    # A workgroup's waves share its positions, a position's pages going round robin over them. With fewer requests per
    # chunk than waves, a wave takes its own positions: one shared position would hold fewer pages than waves, and
    # walking it costs a dependent table load per position.
    per_wave = chunk < _kernel.NW
    wanted = max(1, TARGET_WAVES // (_kernel.NW * nch))
    # past 16 rows: 2 waves per SIMD either way, and the buffered flush cost 10% (4 heads x 5 queries, 32K-128K)
    unit = (
        _unit(
            max_blocks,
            wanted * (_kernel.NW if per_wave else 1),
            _kernel.UMAX,
            footprint,
        )
        if _kernel.padded_rows(rows) <= 16
        else 0
    )
    units = _cdiv(max_blocks, max(1, unit))
    positions = _cdiv(units, _kernel.NW) if per_wave else units
    splits = max(1, min(positions, wanted))
    # per-wave chunks of 2-3 requests: shared positions where the device lengths fit one position per workgroup
    dev_pos = per_wave and chunk >= 2 and unit == 0
    _run_compiled(
        _get_kernel_devpos(rows, num_heads)
        if dev_pos
        else _get_kernel(rows, num_heads, unit > 0),
        q_idx.data_ptr(),
        index_cache.data_ptr(),
        nvfp4_e4m3_lut(q_idx.device).data_ptr(),
        score.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        int(index_cache.stride(0)),
        int(q_idx.stride(0)),
        int(q_idx.stride(1)),
        _end_bytes(q_idx),
        int(score.stride(0)),
        int(score.stride(1)),
        s_bytes,
        int(block_table.stride(0)),
        _end_bytes(block_table),
        int(num_reqs),
        int(chunk),
        int(nch),
        int(splits),
        int(init_blocks),
        int(local_blocks),
        int(per_wave),
        int(unit),
        _current_stream(),
        # the buffered variant at the plain one's occupancy (154-169 VGPRs, its flush 169-171); past 16 rows a hint
        # spilled
        waves_per_eu=3 if unit > 0 and _kernel.padded_rows(rows) <= 16 else None,
    )


# Tests / benchmarks: pin the scorer ("page", "first_page", "first_page_xcd", "sector", "scan"; two heads: "page",
# "first_page_xcd", "runs"), or "chunked" for ``nvfp4_index_score_decode``.
PIN_SCORER: str | None = None
# two heads: runs from 13 requests (32K-192K, 12: first page 18.7 vs runs 20.2 us, 16: 24.3 vs 21.1; sweep_decode_h2.py)
RUNS_MIN_REQS = 13


def _fast_scorer(reqs: int, query_len: int, heads: int) -> str | None:
    """The scorer of ``reqs`` x ``query_len`` rows at ``heads`` index heads, or None (chunked). One head through 16
    requests: two waves per page at 1 / 3 requests, a scalar first table entry at 2 (XCD-local score lines at 8 rows),
    sectors at 4 x 8 rows, independent scans otherwise. Two heads: two waves per page at one request, the XCD-local
    first-page scans to RUNS_MIN_REQS, then 16-block runs balanced over the workgroups."""
    if PIN_SCORER is not None:
        return None if PIN_SCORER == "chunked" else PIN_SCORER
    if query_len not in (4, 8) or reqs < 1:
        return None
    if heads == 2:
        from .score_runs_d128_p128_h2 import MAX_REQS

        return "page" if reqs == 1 else "first_page_xcd" if reqs < RUNS_MIN_REQS else "runs" if reqs <= MAX_REQS \
            else None
    if heads != 1 or reqs > 16:
        return None
    if reqs in (1, 3):
        return "page"
    if reqs == 2:
        return "first_page_xcd" if query_len == 8 else "first_page"
    if reqs == 4 and query_len == 8:
        return "sector"
    return "scan"


# Tests / benchmarks: pin the run scorer's grid and unit bound.
PIN_RUNS_GRID: int | None = None
PIN_RUNS_UNITS: int | None = None
# two workgroups per CU, 4-block units below 64 runs: within 2% of the best of grids 256-2048 (sweep_decode_h2.py)
RUNS_GRID = 512
RUNS_UNITS = 256


def _score_runs(q_idx, index_cache, score, block_table, seq_lens, init_blocks, local_blocks, query_len):
    """Two heads, runs of 16 blocks balanced over the workgroups (``score_runs_d128_p128_h2``)."""
    from .score_runs_d128_p128_h2 import compile_score_runs

    _run_compiled(
        compile_score_runs(query_len), q_idx.data_ptr(), index_cache.data_ptr(),
        nvfp4_e4m3_lut(q_idx.device).data_ptr(), score.data_ptr(), block_table.data_ptr(), seq_lens.data_ptr(),
        int(index_cache.stride(0)), int(score.stride(0)), int(score.stride(1)), _end_bytes(score),
        int(block_table.stride(0)), _end_bytes(block_table), q_idx.shape[0] // query_len, int(init_blocks),
        int(local_blocks), PIN_RUNS_GRID or RUNS_GRID, PIN_RUNS_UNITS or RUNS_UNITS, _current_stream(),
    )


def _columns_ok(q_idx, score) -> bool:
    """Two heads: Q rows ``[rows, 2, 128]`` and the score heads contiguous, 16-B aligned score rows
    (``columns.columns``)."""
    rows, heads, _ = q_idx.shape
    return heads == 1 or (q_idx.stride(1) == 128 and q_idx.stride(0) == heads * 128
                          and score.stride(0) == rows * score.stride(1) and score.stride(1) % 4 == 0)


def _score_rows(q_idx, index_cache, score, block_table, seq_lens, init_blocks,
                local_blocks, query_len, max_seq_len):
    """The scores of ``nvfp4_index_score_top16``: the scorer of ``_fast_scorer``, else ``nvfp4_index_score_decode``."""
    rows, heads, _ = q_idx.shape
    reqs = rows // query_len
    kind = _fast_scorer(reqs, query_len, heads) if rows > 0 and _columns_ok(q_idx, score) else None
    if kind == "runs":
        _score_runs(q_idx, index_cache, score, block_table, seq_lens, init_blocks, local_blocks, query_len)
    elif kind is not None:
        max_blocks = max(1, _cdiv(max_seq_len, 128))
        waves = 2 if kind == "page" else 1
        splits = min(max(1, TARGET_WAVES // (waves * reqs)), max_blocks)
        # A lane holds one strided page-table entry: a 1M context needs at most eleven per wave.
        assert _cdiv(max_blocks, splits) <= 64
        if heads == 1:
            from . import score_d128_p128_h1 as scan, score_firstpage_d128_p128_h1 as first
            from . import score_page_d128_p128_h1 as page, score_sector_d128_p128_h1 as sector

            if kind == "page":
                exe = page.compile_score_page(query_len)
            elif kind in ("first_page", "first_page_xcd"):
                exe = first.compile_score_first_page(query_len, kind == "first_page_xcd")
            elif kind == "sector":
                exe = sector.compile_score_sector(query_len)
            else:
                exe = scan.compile_score(query_len, waves)
        else:
            from . import score_firstpage_d128_p128_h2 as first, score_page_d128_p128_h2 as page

            exe = page.compile_score_page(query_len) if kind == "page" else first.compile_score_first_page(query_len)
        _run_compiled(
            exe, q_idx.data_ptr(), index_cache.data_ptr(),
            nvfp4_e4m3_lut(q_idx.device).data_ptr(), score.data_ptr(),
            block_table.data_ptr(), seq_lens.data_ptr(), int(index_cache.stride(0)),
            int(q_idx.stride(0) if heads == 1 else q_idx.stride(1)), int(score.stride(1)), _end_bytes(score),
            _end_bytes(block_table), max_blocks, splits, reqs, int(block_table.stride(0)), int(init_blocks),
            int(local_blocks), _current_stream(),
        )
    else:
        nvfp4_index_score_decode(
            q_idx, index_cache, score, block_table, seq_lens,
            init_blocks, local_blocks, query_len=query_len, max_seq_len=max_seq_len,
        )


def nvfp4_index_score_top16(
    q_idx: torch.Tensor,  # [num_reqs * query_len, H, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    topk_idx: torch.Tensor,  # [H, num_reqs * query_len, 16] int32
    *,
    query_len: int,
    max_seq_len: int,
    score: torch.Tensor | None = None,  # [H, rows, >= score_width] fp32 workspace
) -> None:
    """The decode top-16 block ids of every (head, row) in the plain block order, in two launches: the scores
    (``nvfp4_index_score_rows``) then the selection (``nvfp4_index_top16_from_scores``). ``score``: a workspace
    (contents unspecified after); None allocates one. Graph-safe (lengths are read on the device)."""
    from ..reference import score_width

    if score is None:
        score = torch.empty(
            q_idx.shape[1],
            q_idx.shape[0],
            score_width(max_seq_len),
            dtype=torch.float32,
            device=q_idx.device,
        )
    _score_rows(
        q_idx, index_cache, score, block_table, seq_lens, init_blocks, local_blocks,
        query_len, max_seq_len,
    )
    nvfp4_index_top16_from_scores(
        score, seq_lens, topk_idx, rows=q_idx.shape[0], query_len=query_len,
        max_seq_len=max_seq_len,
    )


def nvfp4_index_top16_from_scores(
    score: torch.Tensor,  # [H, rows, >= score_width] fp32 of nvfp4_index_score_rows
    seq_lens: torch.Tensor,  # [num_reqs] int32
    topk_idx: torch.Tensor,  # [H, rows, 16] int32, written
    *,
    rows: int,  # query rows (num_reqs * query_len)
    query_len: int,
    max_seq_len: int,
) -> None:
    """The selection half of ``nvfp4_index_score_top16``, one launch: the plain-order four-wave selector through 128
    rows at one head x 4 / 8 rows, else ``topk.index_topk_decode_indices``. On the scores of
    ``nvfp4_index_score_rows`` the ids equal the entry's bit for bit (for a consumer that cannot run
    ``row_top16_prologue_ids``)."""
    from ..reference import score_width
    from ..topk import index_topk_decode_indices

    heads, rows = int(topk_idx.shape[0]), int(rows)
    if _plain_selector(rows, query_len, heads):
        _run_compiled(
            _get_top16_plain_wave4(), score.data_ptr(), int(score.stride(0)),
            int(score.stride(1)), score_width(max_seq_len), seq_lens.data_ptr(),
            int(query_len), topk_idx.data_ptr(), int(topk_idx.stride(0)),
            int(topk_idx.stride(1)), int(rows), int(heads), _current_stream(),
        )
    else:
        index_topk_decode_indices(
            score, topk_idx, seq_lens, max_seq_len=max_seq_len, query_len=query_len
        )


# Tests / benchmarks: pin the selector (True: the plain four-wave one).
PIN_PLAIN: bool | None = None


def _plain_selector(rows: int, query_len: int, heads: int) -> bool:
    """The plain-order four-wave selector through 128 rows at one head, 256 at two (x 4 / 8 rows per request), else
    ``topk.index_topk_decode_indices`` (two heads, sweep_decode_h2.py: 256 rows 3.33 vs 3.62 us, 512 4.12 vs 3.87)."""
    if PIN_PLAIN is not None:
        return PIN_PLAIN
    return query_len in (4, 8) and 0 < rows <= {1: 128, 2: 256}.get(heads, 0)


@functools.cache
@launcher
def _get_top16_plain_wave4():
    from ..topk.top16_indices_p128_plain_wave4 import compile_top16_indices_wave4
    return compile_top16_indices_wave4()


def _top16_mode(q_idx, index_cache, block_table, seq_lens, init_blocks, local_blocks,
                topk_idx, *, query_len, max_seq_len, score=None):
    return q_idx.shape[1], query_len, q_idx.device.index


def _top16_sizes(mode):
    # requests: 1 / 3 two waves per page, 2 first page, 4 sectors (8 rows), 8 independent scans (5-16 requests); with
    # the larger sizes they also cover the other modes' chunked variants (device positions, direct and buffered
    # stores) and both selectors
    return (1, 2, 3, 4, 8, 32, 64, 128, 256)


def _top16_inputs(mode, n, q_idx, index_cache, block_table, seq_lens, init_blocks,
                  local_blocks, topk_idx, *, query_len, max_seq_len, score=None):
    heads, ql, _ = mode
    rows = n * ql
    dev = q_idx.device
    hmax = 1 << 20
    # Compile-only: two zero pages suffice. No served page data is copied.
    args = (
        torch.zeros(rows, heads, 128, dtype=torch.float8_e4m3fn, device=dev),
        torch.zeros(2, 128, 72, dtype=torch.uint8, device=dev),
        torch.zeros(n, hmax // 128, dtype=torch.int32, device=dev),
        torch.full((n,), ql, dtype=torch.int32, device=dev),
        init_blocks, local_blocks,
        torch.empty(heads, rows, 16, dtype=torch.int32, device=dev),
    )
    return args, dict(query_len=ql, max_seq_len=hmax)


nvfp4_index_score_top16 = entry(
    "indexer/nvfp4_index_score_top16_d128_p128", mode=_top16_mode,
    sizes=_top16_sizes, inputs=_top16_inputs,
)(nvfp4_index_score_top16)


def nvfp4_index_score_rows(
    q_idx: torch.Tensor,  # [num_reqs * query_len, H, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, rows, >= score_width] fp32
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    *,
    query_len: int,
    max_seq_len: int,
) -> None:
    """The scores ``nvfp4_index_score_top16`` selects from, in one launch, for a consumer kernel that runs the top-16
    in its own prologue (``topk.top16_indices_p128_plain_wave4.row_top16_prologue_ids``). Bitwise
    ``nvfp4_index_score_decode``'s scores below the row's block count (all a top-16 reads); entries past it
    unspecified. Graph-safe (lengths are read on the device)."""
    _score_rows(
        q_idx, index_cache, score, block_table, seq_lens, init_blocks, local_blocks,
        query_len, max_seq_len,
    )


def _rows_mode(q_idx, index_cache, score, block_table, seq_lens, init_blocks,
               local_blocks, *, query_len, max_seq_len):
    return q_idx.shape[1], query_len, q_idx.device.index


def _rows_inputs(mode, n, q_idx, index_cache, score, block_table, seq_lens,
                 init_blocks, local_blocks, *, query_len, max_seq_len):
    from ..reference import score_width

    (q, cache, table, seq, init, local, _), kw = _top16_inputs(
        mode, n, q_idx, index_cache, block_table, seq_lens, init_blocks,
        local_blocks, None, query_len=query_len, max_seq_len=max_seq_len,
    )
    heads, rows = q.shape[1], q.shape[0]
    work = torch.empty(heads, rows, score_width(kw["max_seq_len"]), dtype=torch.float32,
                       device=q.device)
    return (q, cache, work, table, seq, init, local), kw


nvfp4_index_score_rows = entry(
    "indexer/nvfp4_index_score_rows_d128_p128", mode=_rows_mode,
    sizes=_top16_sizes, inputs=_rows_inputs,
)(nvfp4_index_score_rows)
