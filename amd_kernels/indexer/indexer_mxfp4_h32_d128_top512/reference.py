# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch contract of the MXFP4 indexer: the logits of AITER's ``paged_mxfp4_mqa_logits`` and the top-512 selection.

Inputs (decode rows of ``next_n`` per request, row r of request b = b * next_n + n):
  q [rows, 32, 64] uint8 (e2m1 pairs, low nibble first), q_scale [rows, 32, 4] uint8 (e8m0 per 32 dims; an int32 per
  head [rows, 32] is the same bytes), weights [rows, 32] fp32 (the softmax and head scales folded in);
  kv_cache [pages, P, 68] uint8, P = 64 or 128 entries per page: a page holds P * 64 code bytes then P * 4 scale
  bytes; its 32-entry groups store the codes as [chunk 0..3][entry][16 B] (chunk c: dims 32 c .. 32 c + 31) and the
  scales as [scale % 2][entry][scale // 2];
  block_table [requests, max_pages] int32, row_ends [rows] int32 (the exclusive key bound of each row).
Logit of key t of row r: sum_h weights[r, h] * relu(q_h . k_t), exact products (e2m1 x e2m1 x 2^(sq + sk - 254)).
Selection: the 512 best keys of [0, row_end), logit descending, ties to the lower position; the output is those
positions ascending, -1 padded. Candidate blocks (8 keys): the block of the row's newest key and the 2047 best others by
the block's maximum logit; the candidate consumer selects the 512 best keys among the candidate blocks' keys. The
kernels sum in fp32, so two keys whose exact logits differ by a few ulps may swap places at the 512th rank:
``topk_matches`` accepts exactly those swaps.
"""

import torch

from .common import BLOCK, CANDIDATES, D, ENTRY_BYTES, H, TOPK

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _codes(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1, dtype=torch.float64, device=packed.device)
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    return torch.stack([lut[lo], lut[hi]], -1).flatten(-2)


def dequant_q(q: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    """[rows, 32, 128] fp64."""
    if q_scale.dtype == torch.int32:
        q_scale = q_scale.contiguous().view(torch.uint8).view(*q_scale.shape, 4)
    s = torch.exp2(q_scale.double() - 127)[..., None].expand(*q_scale.shape, 32).flatten(-2)
    return _codes(q) * s


def dequant_keys(pages: torch.Tensor, page_entries: int) -> torch.Tensor:
    """[len(pages) * P, 128] fp64 of the given pages [n, P, 68] (token order)."""
    n, P = pages.shape[0], page_entries
    flat = pages.reshape(n, P * ENTRY_BYTES)
    codes = flat[:, : P * 64].reshape(n, P // 32, 4, 32, 16).permute(0, 1, 3, 2, 4).reshape(n, P, 64)
    scales = flat[:, P * 64 :].reshape(n, P // 32, 2, 32, 2).permute(0, 1, 3, 4, 2).reshape(n, P, 4)
    s = torch.exp2(scales.double() - 127)[..., None].expand(n, P, 4, 32).reshape(n, P, D)
    return (_codes(codes) * s).reshape(n * P, D)


def request_keys(kv_cache, block_table, req: int, length: int, page_entries: int) -> torch.Tensor:
    npg = -(-length // page_entries)
    pages = kv_cache.reshape(kv_cache.shape[0], page_entries, ENTRY_BYTES)[block_table[req, :npg].long()]
    return dequant_keys(pages, page_entries)[:length]


def logits_reference(q, q_scale, weights, kv_cache, block_table, row_ends, next_n: int, page_entries: int):
    """List over rows of the fp64 logits [row_end] of every row."""
    Q = dequant_q(q, q_scale)
    out = []
    for req in range(block_table.shape[0]):
        rows = range(req * next_n, (req + 1) * next_n)
        length = max(int(row_ends[r]) for r in rows)
        keys = request_keys(kv_cache, block_table, req, max(length, 0), page_entries)
        for r in rows:
            e = max(int(row_ends[r]), 0)
            x = torch.einsum("hd,td->ht", Q[r], keys[:e])
            out.append((torch.relu(x) * weights[r].double()[:, None]).sum(0))
    return out


def select_reference(logits: torch.Tensor, k: int = TOPK) -> torch.Tensor:
    """Positions of the k best (logit descending, position ascending), ascending, -1 padded: [k] int32."""
    n = logits.numel()
    order = torch.sort(logits, descending=True, stable=True).indices[: min(k, n)]
    out = torch.full((k,), -1, dtype=torch.int32, device=logits.device)
    out[: order.numel()] = order.sort().values.int()
    return out


def topk_reference(logits_rows, k: int = TOPK) -> torch.Tensor:
    return torch.stack([select_reference(x, k) for x in logits_rows])


def block_scores_reference(logits: torch.Tensor) -> torch.Tensor:
    """Maximum logit of each 8-token block of one row ([ceil(n / 8)])."""
    n = logits.numel()
    pad = torch.full((-(-n // BLOCK) * BLOCK,), float("-inf"), dtype=logits.dtype, device=logits.device)
    pad[:n] = logits
    return pad.view(-1, BLOCK).amax(1)


def candidates_reference(logits: torch.Tensor, k: int = CANDIDATES) -> torch.Tensor:
    """The k candidate 8-token blocks of one row: the block of its newest key, then the k - 1 best others by block
    maximum (ties to the lower block); ids ascending, -1 padded."""
    scores = block_scores_reference(logits)
    n = scores.numel()
    if n == 0:
        return torch.full((k,), -1, dtype=torch.int32, device=logits.device)
    pin = n - 1
    rest = torch.cat([scores[:pin], scores[pin + 1 :]])
    pos = torch.cat([torch.arange(pin, device=logits.device), torch.arange(pin + 1, n, device=logits.device)])
    order = torch.sort(rest, descending=True, stable=True).indices[: min(k - 1, rest.numel())]
    ids = torch.cat([pos[order], torch.tensor([pin], device=logits.device)]).sort().values.int()
    out = torch.full((k,), -1, dtype=torch.int32, device=logits.device)
    out[: ids.numel()] = ids
    return out


def candidate_topk_reference(logits: torch.Tensor, candidates: torch.Tensor, k: int = TOPK) -> torch.Tensor:
    """The k best tokens among the candidate blocks' tokens of one row, positions ascending, -1 padded."""
    ids = candidates[candidates >= 0].long()
    pos = (ids[:, None] * BLOCK + torch.arange(BLOCK, device=ids.device)).flatten()
    pos = pos[pos < logits.numel()].sort().values
    sel = select_reference(logits[pos], k)
    return torch.where(sel >= 0, pos[sel.clamp(min=0).long()].int(), sel)


def ragged_reference(
    topk: torch.Tensor, next_n: int, slot_block_table: torch.Tensor, slot_block_size: int, is_valid: torch.Tensor
):
    """(slots [rows * 512], indptr [rows + 1], lens [rows]) int32 of the sparse attention's lists: row r's non-negative
    positions p of ``topk`` in order as ``slot_block_table[r // next_n][p // B] * B + p % B`` (B = ``slot_block_size``),
    packed without gaps; a row that is not valid has none."""
    rows = topk.shape[0]
    lens = torch.where(is_valid.bool(), (topk >= 0).sum(1), torch.zeros_like(is_valid)).int()
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device=topk.device)
    indptr[1:] = torch.cumsum(lens, 0)
    slots = torch.zeros(rows * topk.shape[1], dtype=torch.int32, device=topk.device)
    for r in range(rows):
        p = topk[r][topk[r] >= 0].long()
        if int(lens[r]) == 0:
            continue
        pg = slot_block_table[r // next_n].long()[p // slot_block_size]
        slots[int(indptr[r]) : int(indptr[r]) + p.numel()] = (pg * slot_block_size + p % slot_block_size).int()
    return slots, indptr, lens


def topk_matches(got: torch.Tensor, logits: torch.Tensor, k: int = TOPK, rtol: float = 1e-5):
    """None if ``got`` ([k] int32) is a valid top-k of ``logits`` up to rounding: right count, positions ascending and
    inside the row, every chosen logit >= the k-th best minus rtol * scale, every other one <= it plus the same; else a
    message."""
    n = logits.numel()
    want = min(k, n)
    g = got[got >= 0].long()
    if g.numel() != want or bool((got[want:] != -1).any()) or bool((g[1:] <= g[:-1]).any()):
        return f"count {g.numel()} vs {want}, tail or order wrong"
    if want == 0:
        return None
    if int(g.max()) >= n:
        return "position outside the row"
    srt = torch.sort(logits, descending=True).values
    thr = srt[want - 1]
    tol = rtol * float(srt.abs().max()) + 1e-300
    mask = torch.zeros(n, dtype=torch.bool, device=logits.device)
    mask[g] = True
    if bool((logits[mask] < thr - tol).any()):
        return "a chosen logit is below the threshold"
    if bool((logits[~mask] > thr + tol).any()):
        return "a better logit was left out"
    return None


def random_cache(num_pages: int, page_entries: int, device, seed: int = 0, scale_lo: int = 120, scale_hi: int = 127):
    """[num_pages, P, 68] uint8: random e2m1 codes, e8m0 scales in [scale_lo, scale_hi)."""
    g = torch.Generator(device=device).manual_seed(seed)
    cache = torch.randint(
        0, 256, (num_pages, page_entries * ENTRY_BYTES), dtype=torch.uint8, device=device, generator=g
    )
    cache[:, page_entries * 64 :] = torch.randint(
        scale_lo, scale_hi, (num_pages, page_entries * 4), dtype=torch.uint8, device=device, generator=g
    )
    return cache.view(num_pages, page_entries, ENTRY_BYTES)


def random_inputs(
    lengths, next_n: int, page_entries: int, device, seed: int = 0, extra_pages: int = 0, max_pages: int | None = None
):
    """Decode inputs of len(lengths) requests of ``next_n`` rows with context ``lengths`` (row n of a request ends at
    length - next_n + n + 1), pages scattered over the cache: (q, q_scale, weights, kv_cache, block_table, row_ends)."""
    B = len(lengths)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    need = [-(-max(c, 0) // page_entries) for c in lengths]
    total = sum(need) + extra_pages
    cache = random_cache(max(total, 1), page_entries, device, seed)
    perm = torch.randperm(max(total, 1), device=device, generator=g).int()
    mp = max_pages or max(max(need), 1)
    bt = torch.zeros(B, mp, dtype=torch.int32, device=device)
    o = 0
    for i, n in enumerate(need):
        bt[i, :n] = perm[o : o + n]
        o += n
    T = B * next_n
    q = torch.randint(0, 256, (T, H, D // 2), dtype=torch.uint8, device=device, generator=g)
    qs = torch.randint(120, 127, (T, H, D // 32), dtype=torch.uint8, device=device, generator=g)
    w = torch.randn(T, H, device=device, generator=g) * 0.1
    cl = torch.tensor(lengths, dtype=torch.int32, device=device)
    ends = (cl[:, None] - next_n + torch.arange(next_n, device=device)[None] + 1).clamp(min=0).reshape(-1).int()
    return q, qs, w, cache, bt, ends.contiguous()
