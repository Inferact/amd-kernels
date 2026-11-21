# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MQA decode attention over per-tile e4m3 KV rows on gfx950 (FlyDSL): 64 query heads, 1 KV head, head dim 512
(448 + 64 RoPE dims), a sliding-window list and a top-k list per query row, an attention sink, the inverse-RoPE MXFP8
output of the o_proj's grouped GEMM. The contract is ``reference.py``."""

from .host import sparse_mla_decode, supports_sparse_mla_decode

__all__ = ["sparse_mla_decode", "supports_sparse_mla_decode"]
