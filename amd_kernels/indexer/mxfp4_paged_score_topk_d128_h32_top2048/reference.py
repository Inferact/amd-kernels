# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference for the MXFP4 paged index scorer + per-row top-2048: the contract the kernel meets.

Operands (the layout of AITER's row-group FP4 MQA kernel and of vLLM's merged MLA cache):

- Query rows ``q_payload`` ``[rows, 32, 64]`` uint8: head ``h``, byte ``i`` holds dims ``2 i`` (low nibble) and
  ``2 i + 1`` (high nibble) as e2m1 codes (sign bit 3, magnitude ``{0, .5, 1, 1.5, 2, 3, 4, 6}``). ``q_scales``
  ``[rows, 1, 4, 16, 4]`` uint8 e8m0 (``2 ** (e - 127)``): entry ``[r, 0, c, l, m]`` scales dims ``32 c .. 32 c + 31``
  of head ``16 m + l``. ``weights`` ``[rows, 32]`` fp32 already include the model's head and softmax factors.
- Keys: index pages ``[num_pages, 64, 68]`` uint8 with physical page stride ``>= 4352`` bytes. ``cache_views``
  of ``amd_kernels...host`` splits a page into ``payload`` ``[pages, 1, 4, 64, 16]`` (chunk ``c`` of key ``s``:
  16 bytes = dims ``32 c .. 32 c + 31``, same nibble order) and ``scales`` ``[pages, 1, 4, 64]`` (chunk ``c``,
  key ``s`` of the page at byte ``(s % 16) * 4 + s // 16``).
- ``block_table`` ``[B, max_blocks]`` int32: key ``j`` of sequence ``b`` is slot ``j % 64`` of page
  ``block_table[b, j // 64]``. ``query_start_loc`` ``[B + 1]`` int32 maps sequence ``b`` to rows
  ``query_start_loc[b] .. query_start_loc[b + 1] - 1``; ``row_ends`` ``[rows]`` int32 is each row's exclusive key
  bound (a DCP rank's local keys; causal across a sequence's speculative rows).

Scores. ``logit[r, j] = sum_h weights[r, h] * relu(sum_d q[r, h, d] * k[j, d])`` for ``j < row_ends[r]`` with
``q`` and ``k`` the dequantized operands (``e2m1 * 2 ** (e8m0 - 127)`` per 32-dim group), the dot accumulated in
fp32. The kernel (the MFMA accumulation order) and ``logits_reference`` (fp32 matmul) agree to ``LOGIT_RTOL`` of
``max(|logit|, 1)`` (measured in ``tests``); the kernel's logits equal AITER's ``flydsl_pa_mqa_logits_fp4_rowgroup``
bit for bit.

Selection. ``topk_reference`` returns what the kernel returns for a row of ``n = row_ends[r]`` keys: the
``min(n, 2048)`` largest logits as ``indices`` ``[rows, 2048]`` int32 (key positions) and ``scores`` ``[rows, 2048]``
fp32 (their logits), the entries past ``min(n, 2048)`` ``-1`` / ``-inf``. The kernel's order within a row is
unspecified but fixed (two runs agree bit for bit); at the 2048th logit ties are broken by an order the kernel
fixes, so equality is checked as (``check_selection``): every index is distinct and in ``[0, n)``, and, against
the kernel's own logits, the selected logits are the exact multiset of the ``min(n, 2048)`` largest with
``scores[r, i] == logit[r, indices[r, i]]``; against ``logits_reference``, each score is within ``LOGIT_RTOL`` of its
logit and no unselected logit exceeds the smallest selected one by more than that.
``topk_reference`` breaks ties toward the lower key position and sorts descending.
"""

import torch

TOPK = 2048
HEADS = 32
HEAD_DIM = 128
PAGE = 64
LOGIT_RTOL = 1e-4

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_e2m1(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``packed`` uint8 ``[..., n]`` (two codes a byte, low nibble first) times ``scales`` uint8 e8m0 ``[..., n // 16]``
    (one per 32 dims = 16 bytes) -> fp32 ``[..., 2 n]``."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
    values = lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    scale = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)
    return values * scale


def dequant_queries(q_payload: torch.Tensor, q_scales: torch.Tensor) -> torch.Tensor:
    """-> fp32 ``[rows, 32, 128]``."""
    rows = q_payload.shape[0]
    scale = q_scales[:, 0, :, :, :2].permute(0, 3, 2, 1).reshape(rows, HEADS, 4)  # [r, m, l, c] -> head 16 m + l
    return dequant_e2m1(q_payload, scale)


def dequant_keys(payload: torch.Tensor, scales: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    """Keys of the given physical pages -> fp32 ``[len(pages), 64, 128]``."""
    p = payload[pages, 0].permute(0, 2, 1, 3).reshape(len(pages), PAGE, 64)
    s = scales[pages, 0]  # [n, 4 (c), 64]: byte (slot % 16) * 4 + slot // 16
    slot = torch.arange(PAGE, device=s.device)
    s = s[:, :, (slot % 16) * 4 + slot // 16].permute(0, 2, 1)  # [n, 64, 4]
    return dequant_e2m1(p, s)


def logits_reference(
    q_payload, q_scales, weights, kv_payload, kv_scales, block_table, query_start_loc, row_ends
) -> list[torch.Tensor]:
    """Per row the fp32 logits of its ``row_ends[r]`` keys (a list: the rows differ in length)."""
    q = dequant_queries(q_payload, q_scales)
    out = []
    bounds = query_start_loc.tolist()
    for b in range(len(bounds) - 1):
        for r in range(bounds[b], bounds[b + 1]):
            n = int(row_ends[r])
            if n == 0:
                out.append(torch.zeros(0, dtype=torch.float32, device=q.device))
                continue
            pages = block_table[b, : (n + PAGE - 1) // PAGE].long()
            k = dequant_keys(kv_payload, kv_scales, pages).reshape(-1, HEAD_DIM)[:n]
            dots = torch.einsum("hd,nd->hn", q[r], k)
            out.append((weights[r].float()[:, None] * torch.relu(dots)).sum(0))
    return out


def topk_reference(logits: list[torch.Tensor], topk: int = TOPK) -> tuple[torch.Tensor, torch.Tensor]:
    """-> ``indices`` int32 and ``scores`` fp32 ``[rows, topk]``, descending, ties to the lower position."""
    device = logits[0].device if logits else "cpu"
    idx = torch.full((len(logits), topk), -1, dtype=torch.int32, device=device)
    val = torch.full((len(logits), topk), float("-inf"), dtype=torch.float32, device=device)
    for r, x in enumerate(logits):
        k = min(topk, x.numel())
        order = torch.sort(x, descending=True, stable=True).indices[:k]
        idx[r, :k] = order.int()
        val[r, :k] = x[order]
    return idx, val


def check_selection(indices, scores, logits, topk=TOPK, rtol=0.0):
    """Assert the kernel's selection meets the contract against the per-row ``logits`` (see the module doc).

    ``rtol`` 0: ``logits`` are the kernel's own scores (bit-equal scorer) and the selection is an exact top-k. Else
    ``logits`` come from ``logits_reference``: every selected score is within ``rtol`` of its logit (relative to
    ``max(|logit|, 1)``) and no unselected logit exceeds the smallest selected one by more than that.
    """
    for r, x in enumerate(logits):
        n = x.numel()
        k = min(topk, n)
        idx, val = indices[r].long(), scores[r]
        live = idx >= 0
        assert int(live.sum()) == k, (r, int(live.sum()), k)
        sel = idx[live]
        assert (sel.numel() == 0 or int(sel.max()) < n) and len(torch.unique(sel)) == k, r
        assert (idx[~live] == -1).all() and (val[~live] == float("-inf")).all(), r
        if rtol == 0.0:
            assert torch.equal(val[live], x[sel]), r
            want = torch.sort(x, descending=True).values[:k]
            assert torch.equal(torch.sort(val[live], descending=True).values, want), r
            continue
        tol = rtol * x[sel].abs().clamp(min=1.0)
        assert ((val[live] - x[sel]).abs() <= tol).all(), r
        if k < n:
            rest = torch.ones(n, dtype=torch.bool, device=x.device)
            rest[sel] = False
            assert x[rest].max() <= x[sel].min() + rtol * max(float(x[sel].min().abs()), 1.0), r
