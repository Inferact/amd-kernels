# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Reference of the DSpark draft dense MLA decode over an fp8 latent cache (this rank's round-robin shard).

Inputs are the vLLM tensors of ``ROCM_AITER_MLA`` with DCP: q ``[B * qlen, 96, 576]`` fp8 e4m3 (scaled by ``q_scale``),
the flat latent cache ``kv [N, 576]`` fp8 (the first 512 columns are also V; both scaled by ``kv_scale``), request ``b``'s
local tokens ``kv_indices[kv_indptr[b] : kv_indptr[b + 1]]`` (shard position j holds global position ``j * W + rank``),
``g_kv_indptr`` the global lengths of the requests (the query block included). Query row t of a request sees the local
tokens ``j`` with ``j * W + rank <= G - qlen + t``.

Contract: ``s = (q * q_scale) . (k * kv_scale) * sm_scale`` in float64, ``P = softmax(s)`` in float64, ``o = P (v * kv_scale)``
rounded once to bf16, ``lse = ln sum exp(s)`` in float32 (-inf and ``o = 0`` for a row with no visible token). The kernel
rounds P to fp8 e4m3 before the PV product (the instruction's operand type, as AITER's ``mla_a8w8`` does), so ``o`` matches
to cosine >= 0.9995 per (row, head) and ``lse`` to 1e-4 absolute; two runs match bit for bit.
"""

import torch

HEADS = 96
KV_LORA_RANK = 512
QK_DIM = 576


def fp8_dense_mla_decode_ref(
    q, kv, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale, sm_scale, qlen, cp_world=8, cp_rank=0
):
    n_req = kv_indptr.numel() - 1
    rows = q.shape[0]
    out = torch.zeros(rows, HEADS, KV_LORA_RANK, dtype=torch.bfloat16, device=q.device)
    lse = torch.full((rows, HEADS), float("-inf"), dtype=torch.float32, device=q.device)
    qs = float(q_scale)
    ks = float(kv_scale)
    ip = kv_indptr.tolist()
    gip = g_kv_indptr.tolist()
    for b in range(n_req):
        n = ip[b + 1] - ip[b]
        if n <= 0:
            continue
        idx = kv_indices[ip[b] : ip[b + 1]].long()
        k = kv[idx].double() * ks
        glen = gip[b + 1] - gip[b]
        for t in range(qlen):
            num = glen - qlen + t - cp_rank
            nv = min(n, num // cp_world + 1) if num >= 0 else 0
            if nv <= 0:
                continue
            qt = q[b * qlen + t].double() * qs
            s = (qt @ k[:nv].T) * sm_scale
            m = s.max(dim=-1, keepdim=True).values
            e = torch.exp(s - m)
            den = e.sum(dim=-1, keepdim=True)
            p = e / den
            o = p @ k[:nv, :KV_LORA_RANK]
            out[b * qlen + t] = o.to(torch.bfloat16)
            lse[b * qlen + t] = (m + torch.log(den)).squeeze(-1).float()
    return out, lse
