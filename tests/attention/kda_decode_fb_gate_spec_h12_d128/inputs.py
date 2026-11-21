# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Served-shape inputs of the KDA spec-verify decode (tests and benchmark)."""

import torch

NH, D, LP = 12, 128, 1536
PROJ_COLS = 6288
SPEC_TOKENS = 6  # num_speculative_tokens 5 + the verified token


def make_inputs(rows, tokens=SPEC_TOKENS, seed=0, device="cuda", conv_dim_first=False, norm_f32=False, pad_seqs=0,
                slot_pitch=0):
    """``rows`` tokens as ceil(rows / tokens) sequences (the last one shorter), plus ``pad_seqs`` padding sequences
    (slot 0 everywhere). Returns the entry's keyword arguments."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=gen, device=device) * scale).to(dtype)

    lens = [tokens] * (rows // tokens) + ([rows % tokens] if rows % tokens else [])
    n_real = len(lens)
    n = n_real + pad_seqs
    cu = torch.tensor([0] + lens + [0] * pad_seqs, device=device).cumsum(0).to(torch.int32)
    slots = n * tokens + 8
    proj = rnd(rows, PROJ_COLS)
    cs_shape = (slots, 3 * LP, tokens + 3)
    cs = rnd(*cs_shape)
    if not conv_dim_first:
        cs = cs.transpose(1, 2).contiguous().transpose(1, 2)
    pitch = slot_pitch or NH * D * D
    state_store = torch.randn((slots * pitch + NH * D * D,), generator=gen, device=device) * 0.1
    state = torch.as_strided(state_store, (slots, NH, D, D), (pitch, D * D, D, 1))
    perm = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))[: n * tokens] + 1
    idx = perm.view(n, tokens).to(device).to(torch.int32)
    idx[n_real:] = 0
    acc = torch.randint(1, tokens + 1, (n,), generator=torch.Generator().manual_seed(seed + 1)).to(device).to(torch.int32)
    nw = (rnd(D, scale=0.5).abs() + 0.5).float() if norm_f32 else rnd(D, scale=0.5).abs() + 0.5
    return dict(
        proj=proj, f_b_weight=rnd(LP, D, scale=0.15), A_log=rnd(NH, scale=0.5, dtype=torch.float32),
        dt_bias=rnd(LP, scale=0.5, dtype=torch.float32), lower_bound=-5.0,
        conv_weight=rnd(3, 4, LP, scale=0.5, dtype=torch.float32), conv_state=cs, state=state, cu_seqlens=cu,
        state_indices=idx, num_accepted=acc, norm_weight=nw, norm_eps=1e-5,
        out=torch.full((rows, NH, D), float("nan"), device=device, dtype=torch.bfloat16),
    )


def clone_inputs(a):
    b = {k: v.clone() if torch.is_tensor(v) else v for k, v in a.items()}
    b["state"] = torch.empty_strided(a["state"].shape, a["state"].stride(), dtype=a["state"].dtype, device=a["state"].device)
    b["state"].copy_(a["state"])
    b["conv_state"] = torch.empty_strided(a["conv_state"].shape, a["conv_state"].stride(), dtype=a["conv_state"].dtype,
                                          device=a["conv_state"].device)
    b["conv_state"].copy_(a["conv_state"])
    return b
