# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The host's launch plan from the shapes alone: ``host.nvfp4_sparse_decode``'s glue around ``_use_tail``,
``_tuned_splits``, ``decode_splits`` and ``tail_schedule``. Tests pick one input per compiled variant with it;
``test_compile.py`` checks it against the keys the host compiles."""

import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host

CUS = 256  # MI355X compute units (the plan's cost model reads them)
XCDS = 8  # MI355X XCDs (``device.initialize_xcd_topology``)

# The served configurations per GPU: (KV heads, compute type, decode_query_len, requests). 1 KV head = 16 query heads
# (TP4), 2 KV heads = 32 query heads (TP2); each computes in bf16 or exact fp16. decode_query_len 1 = no draft, 4 =
# DSpark with 3 drafts, 8 = DSpark with 7 drafts (TP4 up to 256 requests = the 2048-token FULL decode graphs).
SERVED = {
    "kv1_bf16_ql1": (1, torch.bfloat16, 1, 256),
    "kv1_bf16_ql4": (1, torch.bfloat16, 4, 128),
    "kv1_bf16_ql8": (1, torch.bfloat16, 8, 256),
    "kv1_f16_ql1": (1, torch.float16, 1, 256),
    "kv1_f16_ql4": (1, torch.float16, 4, 128),
    "kv1_f16_ql8": (1, torch.float16, 8, 256),
    "kv2_bf16_ql1": (2, torch.bfloat16, 1, 128),
    "kv2_bf16_ql4": (2, torch.bfloat16, 4, 128),
    "kv2_bf16_ql8": (2, torch.bfloat16, 8, 128),
    "kv2_f16_ql1": (2, torch.float16, 1, 128),
    "kv2_f16_ql4": (2, torch.float16, 4, 128),
    "kv2_f16_ql8": (2, torch.float16, 8, 128),
}


def plan(reqs, ql, hkv, compute, quant, xcd_count=XCDS, topk=16, cus=CUS):
    """``host._get_kernel``'s key ``(splits, topk, xcds, quant, exact_fp16, tail)`` for an automatic launch
    (``splits`` and ``tail_splits`` None); the q8 variants append their flags (and hkv 2 at 2 KV heads)."""
    exact_fp16 = compute == torch.float16
    rows = reqs * ql
    xcds = xcd_count if xcd_count and xcd_count > 1 else 0
    tail = host._use_tail(exact_fp16, quant, ql, hkv, topk, xcds, reqs, cus)
    q8 = hkv in host.Q8_KV_HEADS and ql == 8 and topk == 16
    row_xcd = (not tail and q8 and hkv in host.ROW_XCD_MAX_ROWS and xcds == 8
               and rows <= host.ROW_XCD_MAX_ROWS[hkv])
    if not tail and exact_fp16 and hkv > host.PLAIN_GRID_MIN_KV and ql > 1 and rows <= 4 * ql:
        xcds = 0
    row_spread = (not tail and not row_xcd and xcds > 0 and 1 < ql < 8
                  and rows <= host.ROW_SPREAD_MAX_ROWS.get(hkv, 0))
    if tail:
        one = 1 if reqs in host._TAIL_ONE_SPLIT.get(ql, ()) else None
        splits = host.tail_schedule(rows * hkv, topk, cus, xcds, one)[2]
    else:
        group = xcds if row_xcd or row_spread else xcds * ql
        grid_x = -(-rows // group) * group if xcds else rows
        wg_rows = grid_x * hkv
        if wg_rows > host.MAX_SPLIT_ROWS:
            splits = 1
        elif row_xcd:
            splits = 4
        else:
            tuned = host._tuned_splits(exact_fp16, quant, ql, hkv, topk, reqs, xcds)
            splits = tuned if tuned is not None else host.decode_splits(rows * hkv, topk, cus)
    key = (splits, topk, xcds, quant, exact_fp16, tail)
    q8 = q8 and xcds == 8
    hk = (2,) if q8 and hkv == 2 else ()
    if row_xcd:
        return (*key, False, True, *((False, False) + hk if hk else ()))
    if not tail and not exact_fp16 and q8 and splits in (1, 2, 4) and grid_x * hkv * splits <= cus:
        return (*key, False, False, True, *((False,) + hk if hk else ()))
    if not tail and not exact_fp16 and q8 and splits in (2, 4) and grid_x * hkv <= 192:
        return (*key, False, False, False, True, *hk)
    if row_spread:
        return (*key, False, False, False, False, 1, True)
    return key


def variants(config, quant):
    """{key: first request count} over the configuration's served range."""
    hkv, compute, ql, max_reqs = SERVED[config]
    first = {}
    for reqs in range(1, max_reqs + 1):
        first.setdefault(plan(reqs, ql, hkv, compute, quant), reqs)
    return first
