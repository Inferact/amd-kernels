# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Random inputs of one K1 launch in the served formats (TP8 shapes of the Kimi-K3 KDA layer)."""

import pytest
import torch

from amd_kernels.attention.kda_mono_decode_k1_h12_d128_ranks8 import reference as R

bf = torch.bfloat16
NPROJ = 6288


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx950 (MI355X) required")


def randn(g, *shape, scale=1.0, dtype=bf):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def make(S, L, nb, *, nacc=1, dim_first=True, seed=0, null_requests=0):
    """Inputs of S rows (S / L requests of L tokens). The last ``null_requests`` requests are graph padding: every
    state slot 0 (the null slot), one accepted token."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nreq = S // L
    slots = S + 4
    sl = 3 + (L - 1)
    w = {
        "prefix": randn(g, S, R.HIDDEN, scale=2.0),
        "delta": randn(g, S, R.HIDDEN, scale=0.5),
        "blocks": randn(g, S, nb + 2, R.HIDDEN, scale=2.0),
        "ares_nw": (1 + randn(g, R.HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": randn(g, R.HIDDEN, scale=0.05),
        "in_nw": (1 + randn(g, R.HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "w_in": randn(g, NPROJ, R.HIDDEN, scale=0.02),
        "w_fb": randn(g, R.PROJ, R.HD, scale=0.08),
        "conv_w": randn(g, 3 * R.PROJ, 4, scale=0.4, dtype=torch.float32),
        "a_log": randn(g, R.NH, scale=0.5, dtype=torch.float32),
        "dt_bias": randn(g, R.PROJ, scale=0.5, dtype=torch.float32),
        "on_w": (1 + randn(g, R.HD, scale=0.1, dtype=torch.float32)).to(bf),
    }
    w["w_in"][NPROJ - 4 :].zero_()
    pad = randn(g, slots, R.NH * R.HD * R.HD + 24576, scale=0.05, dtype=torch.float32)
    w["rstate"] = pad[:, : R.NH * R.HD * R.HD].view(slots, R.NH, R.HD, R.HD)
    conv = randn(g, slots, 3 * R.PROJ * sl + 4096)[:, : 3 * R.PROJ * sl]
    w["conv_state"] = conv.view(slots, 3 * R.PROJ, sl) if dim_first else conv.view(slots, sl, 3 * R.PROJ).transpose(-1, -2)
    w["st_idx"] = torch.arange(3, 3 + S, dtype=torch.int32, device="cuda").view(nreq, L)
    w["num_acc"] = torch.tensor([1 + (nacc - 1 + r) % L for r in range(nreq)], dtype=torch.int32, device="cuda")
    if null_requests:
        w["st_idx"][nreq - null_requests :] = 0
        w["num_acc"][nreq - null_requests :] = 1
    return w


def clone_state(w):
    return {k: (v.clone() if k in ("prefix", "blocks", "conv_state", "rstate") else v) for k, v in w.items()}


def reference(w, L, nb, write_idx):
    s = clone_state(w)
    core = R.kda_decode_k1(
        s["prefix"], s["delta"], s["blocks"], s["ares_nw"], s["ares_qk"], s["in_nw"], s["w_in"], s["w_fb"],
        s["conv_w"], s["conv_state"], s["a_log"], s["dt_bias"], s["on_w"], s["rstate"], s["st_idx"], s["num_acc"],
        qlen=L, nblocks=nb, write_idx=write_idx,
    )  # fmt: skip
    return s, core


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
