# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Served-like inputs of ``NgramRows``: the layout of the two Engram tables (primes, offsets, multipliers as vLLM's
``EngramLayout`` / ``compute_hash_multipliers`` derive them, for a chosen bucket size), random E4M3 / E8M0 tables, and
decode batches (requests of 1-6 rows: a verified token and its draft) with the lookback windows the runner passes."""

import numpy as np
import torch

COMPRESSED_VOCAB = 99092
VOCAB = 129280
PAD_ID_TOKEN = 2
LAYERS = (1, 14)
MAX_NGRAM = 4
HEADS = 8
DEPTH = 3
RANKS = 8
IMAGE_ID = 129264
SERVED_BUCKET = 16000000


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def layout(bucket=SERVED_BUCKET):
    """(primes [L, 24], offsets [L, 24], multipliers [L, 4]) int64; total rows per layer = sum of its primes."""
    seen, primes = set(), []
    for _ in LAYERS:
        flat = []
        for _ in range(MAX_NGRAM - 1):
            cur = bucket - 1
            for _ in range(HEADS):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                flat.append(cur)
        primes.append(flat)
    primes = torch.tensor(primes, dtype=torch.int64)
    offsets = torch.cumsum(primes, 1) - primes
    bound = max(1, (np.iinfo(np.int64).max // COMPRESSED_VOCAB) // 2)
    mult = [np.random.default_rng(10007 * layer).integers(0, bound, size=(MAX_NGRAM,), dtype=np.int64) * 2 + 1
            for layer in LAYERS]
    return primes, offsets, torch.tensor(np.stack(mult))


def token_map(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, COMPRESSED_VOCAB, (VOCAB,), generator=g, dtype=torch.int32)


def table(rows, seed):
    """(codes uint8 [rows, 256] with finite E4M3 codes, scales uint8 [rows, 8] E8M0 near 1.0)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (rows, 256), generator=g, dtype=torch.uint8)
    w[(w & 0x7F) == 0x7F] = 0x38
    s = torch.randint(120, 134, (rows, 8), generator=g, dtype=torch.uint8)
    return w, s


def decode_batch(num_requests, rows_per_request, pad_to, seed, long_prefix=True, image_every=0, slots=None):
    """(input_ids [pad_to] int32, positions [pad_to] int64, query_start_loc [R + 1] int32, lookback [R, 3] int32,
    num_tokens): requests (``slots`` request slots, the idle ones empty with a -1 window) with ``rows_per_request`` rows, padded to ``pad_to`` rows (the padding rows belong to the
    last request's boundary, as the runner's CUDA graph padding leaves them); the first requests start at positions
    0-4 so the lookback runs off the sequence start."""
    g = torch.Generator().manual_seed(seed)
    counts = [rows_per_request if isinstance(rows_per_request, int) else rows_per_request[i % len(rows_per_request)]
              for i in range(num_requests)]
    total = sum(counts)
    assert total <= pad_to
    ids = torch.zeros(pad_to, dtype=torch.int32)
    pos = torch.zeros(pad_to, dtype=torch.int64)
    slots = slots or num_requests
    qsl = torch.zeros(slots + 1, dtype=torch.int32)
    lookback = torch.full((slots, DEPTH), -1, dtype=torch.int32)
    start = 0
    for r, c in enumerate(counts):
        p0 = int(torch.randint(0, 5, (1,), generator=g)) if (r % 4 == 0 or not long_prefix) else \
            int(torch.randint(5, 200000, (1,), generator=g))
        seq = torch.randint(0, VOCAB, (p0 + c,), generator=g, dtype=torch.int32)
        if image_every:
            seq[::image_every] = IMAGE_ID
        ids[start:start + c] = seq[p0:]
        pos[start:start + c] = torch.arange(p0, p0 + c)
        for col in range(DEPTH):
            if p0 - 1 - col >= 0:
                lookback[r, col] = seq[p0 - 1 - col]
        start += c
        qsl[r + 1] = start
    qsl[num_requests + 1:] = start
    return ids, pos, qsl, lookback, total
