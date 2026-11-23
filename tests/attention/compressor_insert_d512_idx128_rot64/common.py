# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Served-shape inputs of the compressor tail: decode batches of 6-row requests padded to a graph size, and prefill-like
chunks, with the ring, main and indexer caches the model builds (block 0 is the null block)."""

import pytest
import torch

RING_ROWS = 8
MAIN_ROWS = {1: 128, 2: 64}  # compressed entries of a 128-position page
IDX_ROWS = {1: 128, 2: 64}


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("needs gfx950")


def make(ratio, rows, lens=None, seed=0, cs_idx_dtype=torch.float32, kv_scale=0.9, big=False):
    """``rows`` graph rows: requests of ``lens`` tokens (default 6 each) fill the front, the rest is padding."""
    g = torch.Generator().manual_seed(seed + rows * 7 + ratio)
    dev = "cuda"
    if lens is None:
        lens = [6] * (rows // 6)
    assert sum(lens) <= rows
    n_req = len(lens)
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    t2r = torch.zeros(rows, dtype=torch.int32)
    pos = torch.zeros(rows, dtype=torch.int64)
    for r in range(n_req):
        a, b = int(qsl[r]), int(qsl[r + 1])
        t2r[a:b] = r
        p0 = int(torch.randint(0, 20000, (1,), generator=g))
        pos[a:b] = torch.arange(p0, p0 + b - a)
    n_real = int(qsl[-1])
    ring_blocks = n_req + 2
    ring_slots = torch.full((rows,), -1, dtype=torch.int64)
    blk = torch.randperm(ring_blocks - 1, generator=g)[:n_req] + 1
    for r in range(n_req):
        a, b = int(qsl[r]), int(qsl[r + 1])
        ring_slots[a:b] = blk[r] * RING_ROWS + pos[a:b] % RING_ROWS

    def slots(entries, n_pages):
        out = torch.full((rows,), -1, dtype=torch.int64)
        pages = torch.randperm(n_pages - 1, generator=g)[: n_req * 40] + 1
        for r in range(n_req):
            a, b = int(qsl[r]), int(qsl[r + 1])
            idx = pos[a:b] // ratio
            out[a:b] = pages[r * 40 + (idx // entries) % 40] * entries + idx % entries
        return out

    main_pages, idx_pages = n_req * 40 + 2, n_req * 40 + 2
    scale = 3.0 if big else 1.0
    raw = (torch.randn((rows, 512 * ratio), generator=g) * scale).float()
    if ratio == 2:
        raw[:, 512:] *= 2
    ring = torch.randn((ring_blocks, RING_ROWS, 1024), generator=g) * scale
    main = torch.randint(0, 256, (main_pages, MAIN_ROWS[ratio], 512), generator=g, dtype=torch.uint8)
    idx = torch.randint(0, 256, (idx_pages, IDX_ROWS[ratio] * 68), generator=g, dtype=torch.uint8)
    cs_main = torch.randn((20000, 64), generator=g) * 1.0
    cs_idx = (torch.randn((20000, 64), generator=g)).to(cs_idx_dtype)
    nw = (torch.rand(512, generator=g) + 0.5).bfloat16()
    kw = (torch.rand(128, generator=g) + 0.5).bfloat16()
    wk = (torch.randn((128, 512), generator=g) * 0.05).bfloat16()
    args = dict(
        raw=raw, positions=pos, ring=ring if ratio == 2 else None, ring_slots=ring_slots if ratio == 2 else None,
        query_start_loc=qsl if ratio == 2 else None, token_to_req=t2r if ratio == 2 else None,
        norm_weight=nw, eps=1e-6, main_cache=main.view(torch.float8_e4m3fn), main_slots=slots(MAIN_ROWS[ratio], main_pages),
        cos_sin_main=cs_main, kv_scale=torch.full((1,), kv_scale), wk=wk, k_norm_weight=kw, k_eps=1e-6,
        idx_cache=idx, idx_slots=slots(IDX_ROWS[ratio], idx_pages), cos_sin_idx=cs_idx, ratio=ratio,
    )
    if ratio == 1:
        args["ring_slots"] = args["main_slots"]
    out = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in args.items()}
    return out
