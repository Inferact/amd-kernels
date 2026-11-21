# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The gated RMSNorm against reference.gated_norm_reference."""

import pytest
import torch
from amd_kernels.attention.kda_prefill_h12_d128 import host
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

from .inputs import ROW, require_gfx950


@pytest.mark.parametrize("tokens", [1, 7, 192, 1000, 4096, 10240])
@pytest.mark.parametrize("w_f32", [True, False])
@pytest.mark.parametrize("in_place", [False, True])
@torch.inference_mode()
def test_gated_norm(tokens, w_f32, in_place):
    require_gfx950()
    g = torch.Generator(device="cuda").manual_seed(tokens)
    proj = torch.randn(tokens, ROW, device="cuda", generator=g).bfloat16()
    gate = proj[:, 3 * R.LP:4 * R.LP].view(tokens, R.NH, R.D)
    o = (3 * torch.randn(tokens, R.NH, R.D, device="cuda", generator=g)).bfloat16()
    w = 1 + 0.1 * torch.randn(R.D, device="cuda", generator=g)
    w = w if w_f32 else w.bfloat16()
    ref = R.gated_norm_reference(o, gate, w, 1e-5)
    out = o.clone() if in_place else torch.empty_like(o)
    host.gated_norm(out if in_place else o, gate, w, 1e-5, out)
    assert (out.float() - ref.float()).abs().max() <= 2 ** -7 * ref.float().abs().max()
    again = o.clone() if in_place else torch.empty_like(o)
    host.gated_norm(again if in_place else o, gate, w, 1e-5, again)
    assert torch.equal(out, again)
