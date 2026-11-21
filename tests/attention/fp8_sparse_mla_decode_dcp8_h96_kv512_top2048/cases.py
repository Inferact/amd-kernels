# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Served batches of the DCP8 sparse MLA decode: rows of spec-decode requests whose keys are scattered over the pages
of an fp8_ds_mla cache, one compact CSR slice per row (this rank's share of the row's top-2048)."""

import pytest
import torch
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.reference import D, H, RECORD_BYTES

PAGE = 64
SM_SCALE = D**-0.5
SERVED_ROWS = (12, 24, 40, 65, 80, 128, 192)
GRAPH_ROWS = (5, 10, 15, 20, 25, 30, 35, 40, 42, 43, 65, 80, 192)
SPEC = 4
NAN_BYTE = 0x7F


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


def make_cache(slots, gen, device="cuda"):
    """``[slots, 656]`` uint8 records: every byte NaN (e4m3 / f32 / bf16), so a read of an unreferenced slot shows."""
    return torch.full((slots, RECORD_BYTES), NAN_BYTE, dtype=torch.uint8, device=device)


def fill_slots(cache, slots, gen, exp_range=(-4, 4)):
    """Valid records in ``slots``: N(0, 2) latent as e4m3 with a random power-of-two scale per 128 dims, bf16 rope."""
    n = slots.numel()
    scale = torch.exp2(torch.randint(exp_range[0], exp_range[1], (n, 4), generator=gen).float())
    latent = (torch.randn(n, 4, 128, generator=gen) * 2 / scale[:, :, None]).clamp(-448, 448)
    rec = torch.empty(n, RECORD_BYTES, dtype=torch.uint8)
    rec[:, :512] = latent.view(n, 512).to(torch.float8_e4m3fn).view(torch.uint8)
    rec[:, 512:528] = scale.contiguous().view(torch.uint8)
    rec[:, 528:] = (torch.randn(n, 64, generator=gen) * 1.5).bfloat16().contiguous().view(torch.uint8)
    cache[slots.long()] = rec.to(cache.device)


def make_batch(counts, gen, pages=4096, device="cuda", spec=SPEC, q_scale=0.5, exp_range=(-4, 4)):
    """Rows with ``counts[r]`` keys each. Rows of one request (``spec`` consecutive) share most of their keys.

    Returns (q, cache, indptr, slots).
    """
    rows = len(counts)
    cache = make_cache(pages * PAGE, gen, device)
    perm = torch.randperm(pages, generator=gen)
    cursor = 0
    chosen, used = [], []
    for r in range(rows):
        n = counts[r]
        if r % spec == 0:
            base_n = max(counts[r : r + spec] + [1])
            nloc = base_n * 6 // 5 + 8
            npg = -(-nloc // PAGE)
            pg = perm[cursor : cursor + npg]
            cursor += npg
            assert cursor <= pages
            local = torch.randperm(npg * PAGE, generator=gen)[:nloc]
            base = pg[local // PAGE] * PAGE + local % PAGE
        pick = base[torch.randperm(base.numel(), generator=gen)[:n]]
        chosen.append(pick.int())
        used.append(pick)
    flat = torch.cat(chosen) if chosen else torch.zeros(0, dtype=torch.int32)
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(counts, dtype=torch.int64).cumsum(0).int()
    fill_slots(cache, torch.unique(flat), gen, exp_range)
    q = (torch.randn(rows, H, D, generator=gen) * q_scale).bfloat16()
    return q.to(device), cache, indptr.to(device), flat.to(device)


def served_counts(rows, gen, mean=256, pad=0):
    """Binomial(2048, mean / 2048) keys per row (this rank's share of the top-2048); the last ``pad`` rows are CUDA-graph padding."""
    c = torch.distributions.Binomial(2048, torch.tensor(mean / 2048)).sample((rows,)).long()
    counts = [int(x) for x in c]
    for r in range(rows - pad, rows):
        counts[r] = 0
    return counts
