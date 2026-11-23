# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``NgramRows``: vLLM's n-gram hash (``common/engram.py`` ``_hash_ids_kernel``) fused with the row
lookup of the Engram table (``_engram_lookup_kernel``), packed as the MXFP8 input of the ``wkv`` GEMM.

Per token t of request r (``query_start_loc``), position p, chunk start c = positions[query_start_loc[r]] and, for
lookback s = 0..3, the source id of position p - s: the batch token t - s if p - s >= c, else the request's
``lookback_ids[r, c - 1 - (p - s)]`` (-1 or outside the window: the pad id), through ``token_map``; an id equal to
``image_id`` is dead. A position before 0 or a dead id blocks itself and every older lookback (value = pad id).
Per layer l and hash column j = 0..23 (n-gram size j // 8 + 2, head j % 8): ``rolling = xor(value_s * multiplier[l, s]
for s <= j // 8 + 1)`` (int64), ``index = rolling % prime[l, j] + offset[l, j]`` (a global table row), and the row is
the E4M3 codes [256] and the E8M0 scales [8] of table l at ``index``; rank q of 8 holds rows [q * peer_rows,
(q + 1) * peer_rows). Bytes are copied, never converted: the output is bit-exact.

Output per layer: codes uint8 [M, 6144] (column j at bytes 256 j; with ``tile_rows`` > 0 the rows padded with zero
codes to a multiple of it and in the weight's (16, 16) layout, byte (m, k) at (m // 16) * 16 K + (k // 64) * 1024 +
((k % 64) // 16) * 256 + (m % 16) * 16 + k % 16), scales in AITER's shuffled activation layout
(``shuffle_scale_a16w4``: [M / 32, K / 256, klane 4, mlane 16, kpack 2, mpack 2] of the [pad32(M), 192] row-major
scales, pad rows 127 = 1.0). A row-major dequant of it equals the bf16 rows of vLLM's lookup bit for bit.
"""

import torch

DIM = 256
SCALES = 8
COLS = 24
HEADS = 8
MAX_NGRAM = 4
K = COLS * DIM
DEAD = -1
PAD_SCALE = 127


def pad32(m: int) -> int:
    return (m + 31) // 32 * 32


def shuffle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Row-major [pad32(M), 192] uint8 scales -> the shuffled layout, flat."""
    m, ks = scales.shape
    return (
        scales.view(m // 32, 2, 16, ks // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous().view(-1)
    )


def shuffle_codes(codes):
    """Row-major codes [R, 6144] (R % 16 == 0) -> the (16, 16) layout, flat."""
    r = codes.shape[0]
    return codes.view(r // 16, 16, K // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous().view(-1)


def window_values(input_ids, positions, query_start_loc, lookback_ids, token_map, pad_id, image_id):
    """int64 [M, 4]: the value of each lookback s of every token (pad id where blocked)."""
    m = input_ids.shape[0]
    rows = query_start_loc.shape[0] - 1
    depth = lookback_ids.shape[1]
    t = torch.arange(m, device=input_ids.device)
    ends = query_start_loc[1:].long()
    req = (ends[None, :] <= t[:, None]).sum(1).clamp(max=rows - 1)
    chunk_idx = query_start_loc.long()[req].clamp(max=m - 1)
    chunk_start = positions.long()[chunk_idx]
    pos = positions.long()
    ids = input_ids.long()
    blocked = torch.zeros(m, dtype=torch.bool, device=input_ids.device)
    values = []
    for s in range(MAX_NGRAM):
        lookback = pos - s
        in_batch = lookback >= chunk_start
        b_tok = ids[(t - s).clamp(min=0)]
        b_src = torch.where(b_tok == image_id, DEAD, token_map.long()[b_tok])
        col = chunk_start - 1 - lookback
        in_window = ~in_batch & (col >= 0) & (col < depth)
        w_tok = lookback_ids.long()[req, col.clamp(0, depth - 1)]
        known = in_window & (w_tok >= 0)
        w_src = torch.where(w_tok == image_id, DEAD, token_map.long()[w_tok.clamp(min=0)])
        source = torch.where(in_batch, b_src, torch.where(known, w_src, pad_id))
        blocked = blocked | (lookback < 0) | (source == DEAD)
        values.append(torch.where(blocked, pad_id, source))
    return torch.stack(values, 1)


def hash_columns(values, multipliers, primes, offsets):
    """int64 [M, COLS] table rows of one layer from the window values and the layer's tables."""
    prod = values * multipliers[None, :MAX_NGRAM]
    rolling = [prod[:, 0]]
    for s in range(1, MAX_NGRAM):
        rolling.append(rolling[-1] ^ prod[:, s])
    cols = [rolling[j // HEADS + 1] % primes[j] + offsets[j] for j in range(COLS)]
    return torch.stack(cols, 1)


def ngram_rows_reference(
    input_ids, positions, query_start_loc, lookback_ids, token_map, multipliers, primes, offsets,
    pad_id, weights, scales, peer_rows, image_id=-2, tile_rows=0,
):
    """(index [L, M, 24] int64, codes [L, rows, 6144] uint8 (flat [L, rows * 6144] when tiled), scales [L, pad32(rows) *
    192] uint8 shuffled), rows = M, or M padded to ``tile_rows``.

    ``weights[l]`` uint8 [rows, 256] and ``scales[l]`` uint8 [rows, 8] are the layer's whole table (the
    concatenation of the ranks' shards)."""
    m = input_ids.shape[0]
    values = window_values(input_ids, positions, query_start_loc, lookback_ids, token_map, pad_id, image_id)
    rows = -(-m // tile_rows) * tile_rows if tile_rows else m
    index, codes, shuf = [], [], []
    for layer, (w, s) in enumerate(zip(weights, scales)):
        idx = hash_columns(values, multipliers[layer], primes[layer], offsets[layer])
        index.append(idx)
        c = w[idx.flatten()].view(m, K)
        if tile_rows:
            c = torch.cat([c, torch.zeros(rows - m, K, dtype=torch.uint8, device=w.device)])
            c = shuffle_codes(c)
        codes.append(c)
        sc = torch.full((pad32(rows), COLS * SCALES), PAD_SCALE, dtype=torch.uint8, device=w.device)
        sc[:m] = s[idx.flatten()].view(m, COLS * SCALES)
        shuf.append(shuffle_scales(sc))
    return torch.stack(index), torch.stack(codes), torch.stack(shuf)


def dequant_reference(codes, scales_rowmajor):
    """bf16 [M, 24, 256] rows from codes [M, 6144] and row-major scales [M, 192] (vLLM lookup's output)."""
    m = codes.shape[0]
    v = codes.view(torch.float8_e4m3fn).view(m, COLS, SCALES, 32).float()
    e = (scales_rowmajor[:m].view(m, COLS, SCALES, 1).to(torch.int32) << 23).view(torch.float32)
    return (v * e).to(torch.bfloat16).view(m, COLS, DIM)
