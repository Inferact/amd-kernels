# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Served-shape inputs of one KDA prefill layer: ragged sequences, a state cache with initial states, a conv cache."""

import torch

from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

ROW = 6288  # in_proj row: q k v (4608) | gate (1536) | f_a (128) | beta (12), padded to a multiple of 16 -> 6288


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        import pytest

        pytest.skip("needs gfx950")


def make_case(seq_lens, seed=0, slots=None, init=None, dev="cuda", lower_bound=-5.0, dt_shift=0.0):
    """A layer's inputs for sequences of ``seq_lens`` tokens; ``init[n]`` False starts sequence n from zero states; a negative
    ``dt_shift`` slows the gate decay (long memory: every chunk transfer matters)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s, sc=1.0: (torch.randn(*s, device=dev, generator=g) * sc)
    n = len(seq_lens)
    T = sum(seq_lens)
    slots = slots or n + 2
    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int32, device=dev)
    proj = r(T, ROW).bfloat16()
    mixed = proj[:, :3 * R.LP]
    gate = proj[:, 3 * R.LP:4 * R.LP].view(T, R.NH, R.D)
    beta_logit = proj[:, 4 * R.LP + R.D:4 * R.LP + R.D + R.NH]
    raw_g = r(T, R.NH, R.D, sc=1.0).bfloat16()
    perm = torch.randperm(slots, device=dev, generator=g)[:n].to(torch.int32)
    init = [True] * n if init is None else init
    return dict(
        mixed=mixed, raw_g=raw_g, beta_logit=beta_logit, gate=gate,
        A_log=(r(R.NH, sc=0.5)).float(), dt_bias=(r(R.LP, sc=0.1) + dt_shift).float(), lower_bound=lower_bound,
        conv_weight=(r(3, 4, R.LP, sc=0.4)).float(),
        conv_state=r(slots, 3 * R.LP, 3).bfloat16(),
        state_cache=(r(slots, R.NH, R.D, R.D, sc=0.05)).float(),
        state_indices=perm, has_initial_state=torch.tensor(init, device=dev), cu_seqlens=cu,
        norm_weight=(1 + r(R.D, sc=0.1)).float(), norm_eps=1e-6,
    )
