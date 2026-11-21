# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shapes shared by the chunk role, the merge role and the host: chunk size, list sizes and the scratch layout.

A sequence's keys are cut into chunks of ``cb`` keys, ``cb`` a multiple of 512 in ``[512, CHUNK_MAX]`` chosen from the
longest row bound so that a sequence has about the planned number of chunks (``host.plan``; a long context is scored by
as many workgroups as a short one, up to the chunk size's cap). A workgroup that selects a chunk on two waves cuts it in
halves: a list is then of a "chunk" of ``cbv = cb / 2`` keys, and ``nvch`` of them (at most ``NCH_MAX``) are all the merge
sees. A row keeps from list ``c`` its ``tb`` largest keys (all of them if it has fewer) and from the sequence's last
chunk, where the newest keys are, all of its keys, so a row's pool is ``(nvch - 1) * slot + cbv`` entries, at most ``POOL``
(four 2048-entry blocks of the merge). A row's pool is ``keys`` and ``positions`` int32 planes of ``POOL`` entries;
``meta`` holds per (row, list) its lower bound, the counts of live and listed keys and the chunk's extreme keys; a
truncated chunk's scores are also stored whole, for the merge to read when its list falls short.
"""

import flydsl.expr as fx
from flydsl.expr.typing import Int32 as I

TOPK = 2048
CHUNK_MAX = 2048
CHUNK_STEP = 512
CHUNKS_TARGET = 40  # default chunks a sequence is cut into (the host may ask for fewer, see host.plan)
NCH_MAX = 64  # CHUNK_MAX chunks cover 131072 keys, 1 Mi tokens at DCP8
POOL = 8192  # entries of a row's pool at most: four 2048-entry blocks of the merge
QMAX = 6  # query rows a sequence may have
MAX_ROWS = 256  # rows one call may hold
INFO_OFFSET = MAX_ROWS * NCH_MAX * 16  # bytes: the per-row records follow the meta plane
LIST_MAX = 576  # keys a list keeps at most (its stage fits a wave's histogram region)
LIST_FLOOR = 64  # keys a list keeps at least: at most ~33 chunks can hold selected keys beyond their lists
COUNTER_BYTES = 256  # a (sequence, row tile)'s ready and consumer counts: a cache line pair of their own
COUNTER_GROUPS = 1024  # (sequence, row tile) groups a call may hold
FULL_STRIDE = NCH_MAX * CHUNK_MAX  # a row's plane of all scores of its truncated chunks
TBDIV = 4  # a list keeps a quarter of its chunk
SPREAD = 2688  # list keys a sequence's chunks share when its selected keys spread evenly (1.3 x TOPK)
POOL_TARGET = POOL  # entries a pool is cut to


def pack_target(chunks: int, min_chunk: int = CHUNK_STEP) -> int:
    """The launch's chunk plan as one int: about ``chunks`` a sequence, none shorter than ``min_chunk`` keys."""
    return chunks | (min_chunk // CHUNK_STEP) << 16


def chunk_size(max_end: int, target: int = CHUNKS_TARGET, min_chunk: int = CHUNK_STEP) -> int:
    per = -(-max_end // target)
    cb = -(-per // CHUNK_STEP) * CHUNK_STEP
    return min(max(cb, min_chunk, CHUNK_STEP), CHUNK_MAX)


def num_chunks(max_end: int, cb: int) -> int:
    return max(1, -(-max_end // cb))


def list_size(nch: int, cb: int) -> int:
    """Keys a chunk list keeps: a quarter of the chunk, cut to the pool target, but at least ``floor`` so that the
    chunks which can hold selected keys beyond their list (at most 2047 // list size) fit the overflow."""
    floor = LIST_FLOOR
    pool = (POOL_TARGET - cb) // max(nch - 1, 1) * 8 // 9 // 32 * 32 if nch > 1 else cb
    want = max(cb // TBDIV // 32 * 32, 32, SPREAD // max(nch, 1) // 32 * 32)
    return min(max(floor, min(want, pool, LIST_MAX)), cb)


def chunk_size_rt(max_end, packed):
    """``chunk_size`` on kernel values (``packed``: ``pack_target``)."""
    target = packed & 0xFFFF
    cb = (((max_end + (target - 1)) // target + (CHUNK_STEP - 1)) // CHUNK_STEP) * CHUNK_STEP
    return fx.min(fx.max(cb, (packed >> 16) * CHUNK_STEP), I(CHUNK_MAX))


def list_size_rt(nch, cb):
    """``list_size`` on kernel values."""
    floor = I(LIST_FLOOR)
    pool = ((I(POOL_TARGET) - cb) // fx.max(nch - 1, I(1)) * 8 // 9 // 32) * 32
    pool = (nch > 1).select(pool, cb)
    want = fx.max(fx.max(cb // TBDIV // 32 * 32, I(32)), I(SPREAD) // fx.max(nch, I(1)) // 32 * 32)
    return fx.min(fx.max(floor, fx.min(fx.min(want, pool), I(LIST_MAX))), cb)


def slot_size(tb):
    """Entries a chunk's list may take: its ``tb`` plus the keys that share the threshold's histogram bin."""
    return tb + tb // 8


def scratch_words(max_rows: int) -> tuple[int, int, int, int, int]:
    """Words of the key plane, the position plane, the meta plane, the full-score plane for ``max_rows`` rows and the
    ready counts."""
    return (max_rows * POOL, max_rows * POOL, max_rows * NCH_MAX * 4 + MAX_ROWS * 4, max_rows * FULL_STRIDE,
            COUNTER_GROUPS * COUNTER_BYTES // 4)
