# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract: the Kimi MoE router, GEMM through top-16, as vLLM's AITER path computes it.

logits = fp32 sum of the bf16 products x [M, 7168] @ W [896, 7168]^T (fp64 here, fp32 in the kernel: its split-K
sum order differs, so logits agree to ~1e-6 relative). Then AITER's ``biased_grouped_topk`` with one group
(``topk_reg_kernel``): s = rcp(1 + exp2(-log2(e) * logit)) in fp32, the choice is s + e_score_correction_bias, the 16
largest choices win, ordered by choice descending and, for equal choices, by AITER's tie rank (the lane priority of its
DPP tree, then the element position of the float4 order: ``tie_rank``); weights = s[ids] * (scale / sum) with the 16-term
sum taken in slot order, and 0 where the sum is not positive.

``top16_reference`` is the same arithmetic in torch (exact reciprocal where AITER's v_rcp_f32 is within 1 ulp): equal
ids and weights to 2 ulp. The kernel itself is tested bit for bit against ``aiter.biased_grouped_topk``. Finite inputs.
"""
import torch

E, H, TOPK = 896, 7168, 16
_LOG2E = 1.44269504088896340736


def tie_rank(expert_id: torch.Tensor) -> torch.Tensor:
    """AITER's ``legacy_topk_tie_rank``: lower wins among equal choices."""
    vec4 = expert_id >> 2
    lane = vec4 & 63
    low2 = (lane & 3) ^ (((lane >> 2) & 1) * 3)
    bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2
    lane_rank = ((~lane) & 0x38) | bit2 | low2
    return (lane_rank << 5) | (((vec4 >> 6) << 2) | (expert_id & 3))


def logits_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.double() @ w.double().t()).float()


def sigmoid_reference(logits: torch.Tensor) -> torch.Tensor:
    y = (logits.double() * -_LOG2E).float()
    return 1.0 / (1.0 + torch.exp2(y))


def top16_reference(logits: torch.Tensor, bias: torch.Tensor, routed_scaling_factor: float = 1.0):
    """(weights fp32 [M, 16], ids int32 [M, 16]) from fp32 logits [M, 896] and the fp32 correction bias [896]."""
    sig = sigmoid_reference(logits)
    bits = (sig + bias).view(torch.int32).long()
    score_key = (bits ^ ((bits >> 31) | -(2**31))) & 0xFFFFFFFF
    ranks = tie_rank(torch.arange(E, device=logits.device))
    order_key = (score_key << 11) | (2047 - ranks)
    ids = torch.topk(order_key, TOPK, dim=-1, sorted=True)[1]
    picked = sig.gather(1, ids)
    total = picked[:, 0].clone()
    for k in range(1, TOPK):
        total = total + picked[:, k]
    scale = torch.tensor(routed_scaling_factor, dtype=torch.float32, device=logits.device)
    weights = torch.where((total > 0)[:, None], picked * (scale / total)[:, None], torch.zeros_like(picked))
    return weights, ids.to(torch.int32)


def router_sigmoid_bias_top16_e896_h7168_reference(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
                                                   routed_scaling_factor: float = 1.0):
    """(weights, ids, logits) for bf16 x [M, 7168], bf16 w [896, 7168], fp32 bias [896]."""
    logits = logits_reference(x, w)
    weights, ids = top16_reference(logits, bias, routed_scaling_factor)
    return weights, ids, logits


def moe_sort_reference(topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    """The BM=16 sort of [M, 16] top-k ids / weights in AITER's auxiliary layout, rows of one expert in ascending (token,
    slot) order: (sorted_token_ids token | slot << 24, sorted_weights, sorted_expert_ids per 16 rows, num_valid_ids
    [padded rows, M], m_indices, reverse_sorted [M * 16]). Padding rows: token id M, weight 0, m_indices M. Rows past
    ``num_valid_ids[0]`` stay padding rows; ``sorted_expert_ids`` entries past ``num_valid_ids[0] / 16`` are 0."""
    m = topk_ids.shape[0]
    routes = m * TOPK
    cap = (routes + min(E, routes) * 15 + 15) // 16 * 16
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).long()
    counts = torch.bincount(flat, minlength=E)
    padded = (counts + 15) // 16 * 16
    first_row = torch.cumsum(padded, 0) - padded
    first_pair = torch.cumsum(counts, 0) - counts
    order = torch.argsort(flat, stable=True)  # pairs by expert, then (token, slot)
    expert_of = flat[order]
    row = torch.empty(routes, dtype=torch.long, device=dev)
    row[order] = first_row[expert_of] + (torch.arange(routes, device=dev) - first_pair[expert_of])
    tok = torch.arange(routes, device=dev) // TOPK
    slot = torch.arange(routes, device=dev) % TOPK
    sti = torch.full((cap,), m, dtype=torch.int32, device=dev)
    sti[row] = (tok | (slot << 24)).to(torch.int32)
    sw = torch.zeros(cap, dtype=torch.float32, device=dev)
    sw[row] = topk_weights.reshape(-1)
    mi = torch.full((cap,), m, dtype=torch.int32, device=dev)
    mi[row] = tok.to(torch.int32)
    sei = torch.zeros(cap // 16, dtype=torch.int32, device=dev)
    blocks = torch.repeat_interleave(torch.arange(E, device=dev), padded // 16)
    sei[:blocks.numel()] = blocks.to(torch.int32)
    nv = torch.tensor([int(padded.sum()), m], dtype=torch.int32, device=dev)
    return sti, sw, sei, nv, mi, row.to(torch.int32)
