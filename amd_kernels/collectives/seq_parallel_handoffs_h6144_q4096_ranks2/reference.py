# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of the two-rank sequence-parallel hand-offs (root README, rule 3).

``pack_reference(x)``: the MXFP8 pack of bf16 ``[M, 4096]`` rows into uint8 ``[M, 4224]``: per 32-value block
``e = clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0, 254)`` (E8M0), ``code = e4m3fn(x * 2^(127 - e))``, round to
nearest even; a row is its 4096 codes, then its 128 scale bytes. One exception, the library quantizer's
(``gemm/mxfp8_common/mxfp8_quant.py``): a block whose maximum is below 2^-119 (byte 0, a denormal scale) is
converted with 2^-126, i.e. its codes are those of ``x * 2^126``. Bit for bit, no tolerance; it equals vLLM's
Triton ``sp_pack.mxfp8_quant_pack`` everywhere but in such blocks (activations never hold a block of nonzero
values below 2^-119; an all-zero block packs to zero codes either way).

The phases (``handoffs.SeqParallelHandoffs``) launch other entries' kernels on this rank's rows and move their bytes
with ``pull_collectives_ranks2`` (or ``sdma_push_ranks2``): every phase output is bit for bit those kernels' outputs
on the rows the collective gives the rank (``pull_collectives_ranks2/reference.py``: gathers concatenate the ranks'
shards in rank order, the all-to-all unpack is ``unpack_reference`` with o_proj_full's scale shuffle, the
reduce-scatters add in fp32 in rank order from +0). Their numerics are those entries' contracts: ``gemm/mxfp8_common``
(o_proj_full), ``norm/gemma_add_rmsnorm_h6144`` (the bf16 post-attention norm of a MoE layer),
``moe/mxfp4_h6144_i1536_e128_top4`` (``prepare_mxfp4_moe``; the gathered MoE input is its routed payload, the
routed-only ``MoeInput``; ``shared_expert_full``, the ``shared_mxfp8_ref`` contract), and the fused norm collectives of
``pull_collectives_ranks2``.

The shared expert of a MoE layer is local to the owner of its rows: rank p computes the full shared expert ``h`` of
its own rows from their prepared MXFP8 operands and adds it into its own rows' slot sum in the MoE tail, in this fixed
order for token row i of rank o(i): on rank p, ``P_p(i) = bf16((((s0 + s1) + s2) + s3) [+ h(i) if p == o(i)])`` in
fp32 over rank p's bf16 slots, then ``y(i) = bf16((+0 + P_0(i)) + P_1(i))`` on o(i) (``slots4_reduce_scatter_reference``
with the owner-form shared), then the residual add, norm and quant. The result does not depend on when the shared
expert's side stream ran (its join is an event on its launch).
"""

import torch

D = 4096
E4M3_MAX = 448.0


def pack_reference(x: torch.Tensor) -> torch.Tensor:
    """bf16 ``[M, 4096]`` -> uint8 ``[M, 4224]`` (module docstring)."""
    m, d = x.shape
    assert d == D and x.dtype == torch.bfloat16
    blocks = x.float().reshape(m, d // 32, 32)
    amax = blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    e = (amax / E4M3_MAX).log2().ceil().add(127).clamp(0, 254)
    # 2^(127 - e), except 2^126 for byte 0 (the kernel's conversion scale)
    rescale = torch.exp2(127 - e.clamp_min(1))
    codes = (blocks * rescale.unsqueeze(-1)).reshape(m, d).to(torch.float8_e4m3fn).view(torch.uint8)
    return torch.cat([codes, e.to(torch.uint8)], dim=1)
