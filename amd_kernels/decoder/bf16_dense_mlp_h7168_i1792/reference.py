# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of one rank's dense MLP shard of the drafter (TP8: hidden 7168, intermediate 14336 / 8 = 1792).

``gate_up`` [3584, 7168] bf16 = [gate rows 0..1791 ; up rows 1792..3583] (the checkpoint's merged column-parallel
weight), ``down`` [7168, 1792] bf16. ``mlp_reference(x, gate_up, down)``: ``act = bf16(silu(x gate^T) * (x up^T))``
from the fp64 products (the intermediate is rounded once to bf16, as the stock ``silu_and_mul`` output is, but gate and
up are not rounded to bf16 first), ``out = bf16(act down^T)`` from an fp64 product: the rank's partial result, to be
all-reduced by the caller. The kernels accumulate in fp32 (K ascending within a split, splits summed in split order) and
evaluate silu in fp32: per-row cosine >= 0.99999 and relative L2 <= 1e-3 against the reference, deterministic bits.
"""
import torch

HIDDEN = 7168
INTER = 1792


def mlp_reference(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(act [rows, 1792] bf16, out [rows, 7168] bf16)."""
    g = x.double() @ gate_up[:INTER].double().t()
    u = x.double() @ gate_up[INTER:].double().t()
    act = (torch.nn.functional.silu(g) * u).bfloat16()
    return act, (act.double() @ down.double().t()).bfloat16()
