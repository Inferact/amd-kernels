# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DCP8 LSE combine of packed partials (16 heads per rank, value dim 192), the contract of the kernel.

Every rank r holds the send buffer ``send_r [8, T, 48, 66]`` bf16 of its partial attention over its KV shard: entry
``[d, t, 3 h + p]`` is part p (64 values) of head ``16 d + h`` of row t, followed by the fp32 base-2 LSE of the head as
two bf16 words (low half first). Rank d keeps heads ``16 d .. 16 d + 15``: it receives ``send_s[d]`` from every source s
and writes ``out [T, 16, 192]`` bf16 (``[T, 48, 64]`` as sub-heads).

Per sub-head, with ``l_s`` the LSE of source s (NaN and +inf read as -inf): ``m = max_s l_s`` (0 when all are -inf),
``g = log2(sum_s 2^(l_s - m)) + m``, ``w_s = 2^(l_s - g)`` (a NaN weight is 0) and ``out = bf16(sum_s w_s x_s)``,
summed in source order in fp32 with a zero weight dropping its value (NaN partials of empty shards never propagate).
Tolerance: max error <= 2^-7 of the max |out| against this reference; bit for bit against vLLM's
``all_to_all_single`` + ``_dcp_a2a_unpack_combine`` (the same exp2 / log2 and unfused products); two runs bit for bit.
"""

import math

import torch

WORLD = 8
HPR = 16
V = 192
D = 64
PARTS = V // D
SUB = HPR * PARTS
PACK = D + 2


def unpack(send):
    """x fp32 [8, T, 48, 64] and lse fp32 [8, T, 48] of a send buffer [8, T, 48, 66] bf16."""
    x = send[..., :D].float()
    lse = send[..., D:].contiguous().view(torch.float32)[..., 0]
    return x, lse


def combine_reference(sends, rank):
    """out bf16 [T, 16, 192] of ``rank`` from the 8 ranks' send buffers ``sends`` (each [8, T, 48, 66] bf16)."""
    x, lse = unpack(torch.stack([s[rank] for s in sends]))
    lse = torch.where(lse.isnan() | (lse == math.inf), -math.inf, lse)
    m = lse.amax(0)
    m = torch.where(m == -math.inf, torch.zeros_like(m), m)
    g = torch.log2(torch.exp2(lse - m).sum(0)) + m
    w = torch.exp2(lse - g)
    w = torch.where(w.isnan(), torch.zeros_like(w), w)
    acc = torch.zeros_like(x[0])
    for s in range(WORLD):
        acc = acc + torch.where(w[s][..., None] == 0, torch.zeros_like(x[s]), x[s] * w[s][..., None])
    t = acc.shape[0]
    return acc.to(torch.bfloat16).view(t, HPR, V)
