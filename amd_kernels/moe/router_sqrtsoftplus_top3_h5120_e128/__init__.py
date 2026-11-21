# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Router of a 128-expert MoE layer: bf16 gate GEMM, sqrt-softplus score, biased top-3, in FlyDSL (gfx950)."""

from .host import router_topk, shuffle_gate, warmup_router_topk
from .reference import logits_reference, router_topk_reference, score_reference, topk_reference

__all__ = ["logits_reference", "router_topk", "router_topk_reference", "score_reference", "shuffle_gate",
           "topk_reference", "warmup_router_topk"]
