# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch contract of the prefill indexer's top-2048 select: ``topk_rows`` = ``reference_topk`` on every row.

Inputs: ``logits`` fp32 [rows, width] (row stride free, finite values; ``-0.0`` and ``+0.0`` are equal), ``ks`` / ``ke``
int32 [rows] (the row's key window ``[ks, ke)``; only those positions are read as data, the rest of the row is
unspecified: the FP4 scorer stores whole pages).

Output int32 [rows, 2048]: the window-relative positions ``n - ks`` of the 2048 largest logits of the window in ascending
order, ties at the 2048th value going to the smaller position (AITER's ``top_k_per_row_prefill(stable=True)`` order with
``ks`` subtracted); a window of fewer than 2048 keys lists all of them, and the remaining entries are -1.

The kernel matches this bit for bit (it is exact: no sampling error survives into the result), and two runs match.
AITER's ``stable=False`` path (DCP, where each rank selects its own rows) returns the same set in an unspecified order
and breaks ties arbitrarily, so the same set up to ties at the 2048th value.
"""

import torch

TOPK = 2048


def reference_topk(logits: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor, k: int = TOPK) -> torch.Tensor:
    rows = logits.shape[0]
    out = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
    ks_l, ke_l = ks.tolist(), ke.tolist()
    for r in range(rows):
        lo, hi = ks_l[r], ke_l[r]
        n = hi - lo
        if n <= 0:
            continue
        if n <= k:
            out[r, :n] = torch.arange(n, dtype=torch.int32, device=logits.device)
            continue
        window = logits[r, lo:hi] + 0.0  # -0.0 -> +0.0
        order = torch.sort(window, descending=True, stable=True).indices[:k]
        out[r] = torch.sort(order).values.to(torch.int32)
    return out
