# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One call's arguments of ``markov_gumbel_fused`` (vocab 163840, rank 512): mixed temperatures, an invalid row, the
cache row mapping."""

import torch

from amd_kernels.sampling.markov_gumbel_common import markov_scratch_dwords

V, RANK = 163840, 512


def make(n, k, seed, weights=None):
    gen = torch.Generator().manual_seed(seed)
    if weights is None:
        weights = [(torch.randn(V, RANK, generator=gen) * 0.05).bfloat16().cuda() for _ in range(2)]
    slots = n + 3
    st = torch.randperm(slots, generator=gen)[:n]
    temp = torch.ones(slots)
    temp[st] = torch.tensor([(0.0, 0.7, 1.0, 1.3)[i % 4] for i in range(n)])
    idx = st.view(n, 1).repeat(1, k)
    if n > 1:
        idx[n // 2] = -1
    rm = torch.full((slots,), -1, dtype=torch.int32)
    rm[torch.randperm(slots, generator=gen)[: slots - 1]] = torch.arange(slots - 1, dtype=torch.int32)
    return {
        "base": (torch.randn(n, k, V, generator=gen) * 3).bfloat16().cuda(), "w1": weights[0], "w2": weights[1],
        "input_ids": torch.randint(0, V, (n + 5,), generator=gen, dtype=torch.int32).cuda(),
        "anchor_idx": torch.randperm(n + 5, generator=gen)[:n].cuda(),
        "draft_tokens": torch.zeros(n + 2, k, dtype=torch.int64, device="cuda"), "idx_mapping": idx.cuda(),
        "sample_pos": torch.randint(1, 200000, (n, k), generator=gen).cuda(), "temperature": temp.cuda(),
        "seeds": torch.randint(0, 2**62, (slots,), generator=gen).cuda(),
        "cache": torch.full((slots - 1, k + 1, V), 7.0, dtype=torch.bfloat16, device="cuda"),
        "scratch": torch.zeros(markov_scratch_dwords(weights[0], n), dtype=torch.int32, device="cuda"),
        "row_mapping": rm.cuda(),
    }


def clone(d):
    return {key: t if key in ("w1", "w2") else t.clone() for key, t in d.items()}
