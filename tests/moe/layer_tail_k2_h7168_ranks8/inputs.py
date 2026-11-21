# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Random inputs for the layer tail in the checkpoint's formats at TP8 shapes (for the tests and the benchmark)."""

import torch

from amd_kernels.moe.layer_tail_k2_h7168_ranks8.reference import E, HIDDEN, LAT, RI, SI, UP_N

bf = torch.bfloat16
PROJ = 1536


def randn(g, *shape, scale=1.0, dtype=bf):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def u8(g, *shape, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=g, device=g.device).to(torch.uint8)


def moe_weights(rank, device):
    """Natural layouts (for ``reference.moe``) and the kernel's shuffled ones, same values."""
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
    from aiter.utility.fp4_utils import e8m0_shuffle

    gc = torch.Generator(device=device).manual_seed(7)
    gr = torch.Generator(device=device).manual_seed(100 + rank)
    w_up = randn(gc, HIDDEN, LAT, scale=0.02)
    nat = {
        "w_gate": randn(gc, E, HIDDEN, scale=0.02),
        "bias": randn(gc, E, scale=0.02, dtype=torch.float32),
        "w_ld": randn(gc, LAT, HIDDEN, scale=0.02),
        "ln_w": (1 + randn(gc, LAT, scale=0.1, dtype=torch.float32)).to(bf),
        "w_up": w_up[rank * UP_N : (rank + 1) * UP_N].contiguous(),
        "w_sgu": randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": randn(gr, HIDDEN, SI, scale=0.03),
        "w13": u8(gr, E, 2 * RI, LAT // 2),
        "w2": u8(gr, E, LAT, RI // 2),
        "w13s": u8(gr, E, 2 * RI, LAT // 32, lo=118, hi=123),
        "w2s": u8(gr, E, LAT, RI // 32, lo=118, hi=123),
    }
    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    dev = {
        k: nat[k]
        for k in ("w_gate", "bias", "w_ld", "ln_w", "w_sgu", "w_sd")
    }
    dev["w_up"] = nat["w_up"]
    dev["w13"] = shuffle_weight_a16w4(nat["w13"].view(fp4), 16, False)
    dev["w2"] = shuffle_weight_a16w4(nat["w2"].view(fp4), 16, False)
    dev["w13s"] = shuffle_scale_a16w4(nat["w13s"].view(e8).view(-1, LAT // 32), E, False)
    dev["w2s"] = e8m0_shuffle(nat["w2s"].view(e8).view(-1, RI // 32))
    return nat, dev


def layer_inputs(rank, device, rows, seed=0, nblocks=4):
    gc = torch.Generator(device=device).manual_seed(11 + 1000 * seed)
    gr = torch.Generator(device=device).manual_seed(200 + rank + 1000 * seed)
    return {
        "core": randn(gr, rows, PROJ),
        "w_o": randn(gr, HIDDEN, PROJ, scale=0.02),
        "prefix": randn(gc, rows, HIDDEN, scale=2.0),
        "blocks": randn(gc, rows, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": randn(gc, HIDDEN, scale=0.05),
        "in_nw": (1 + randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
    }
