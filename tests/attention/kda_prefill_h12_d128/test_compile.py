# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after the first calls (root README, rule 4): the kernels are compiled once per process (the grid, the group
count and the sequence lengths are runtime arguments); the compile keys of a call are the path (one launch / staged), the norm weight
dtype, the checkpoint export and, on the staged path, whether the sequences are grouped (the walk of grouped chains hosts the fold)."""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.attention.kda_prefill_h12_d128 import chunk_table, prefill
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

from .inputs import make_case, require_gfx950

SERVED = [([4096], 0), ([300, 64, 77, 1], 0), ([130], 0), ([1500, 700, 64, 3000], 0), ([10240], 0), ([2048], 0), ([3000, 64], 0), ([700] * 4, 0)]


def call(c, w_f32, ck, impl=None):
    cu = c["cu_seqlens"]
    t = int(cu[-1])
    table, offs = R.chunk_table(cu)
    ci = torch.tensor([[n, k] for n, k, _, _ in table], dtype=torch.int32, device="cuda")
    co = torch.tensor(offs, dtype=torch.int64, device="cuda")
    out = torch.zeros(t, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    w = c["norm_weight"] if w_f32 else c["norm_weight"].bfloat16()
    extra = {}
    if ck:
        extra = dict(checkpoint_state=torch.zeros(len(cu) + 1, R.NH, R.D, R.D, device="cuda"),
                     checkpoint_offsets=torch.zeros(len(cu) - 1, dtype=torch.int32, device="cuda"))
    prefill(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["conv_weight"], c["conv_state"].clone(),
            c["state_cache"].clone(), c["state_indices"], c["has_initial_state"], cu, ci, co, w, c["norm_eps"], out, c["lower_bound"],
            table=chunk_table(cu, ci, c["state_indices"], c["has_initial_state"]),
            chunk_counts=[b - a for a, b in zip(offs[:-1], offs[1:])], impl=impl, **extra)
    return out


@torch.inference_mode()
@pytest.mark.parametrize("w_f32,ck", [(True, False), (False, False), (True, True)])
def test_no_compile_after_warmup(w_f32, ck, monkeypatch):
    require_gfx950()
    call(make_case([2048]), w_f32, ck, "v1")  # grouped chains
    call(make_case([64]), w_f32, ck, "v1")  # whole chains
    call(make_case([2048]), w_f32, ck, "mega")
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile

    def counting(fn, *args, **kwargs):
        compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    for seq, _ in SERVED:
        call(make_case(seq), w_f32, ck)
        assert not compiled, f"{seq} compiled {compiled}"
