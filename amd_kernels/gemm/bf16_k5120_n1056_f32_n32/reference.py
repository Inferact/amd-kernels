# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch contract of the fused fp32 / bf16 GEMM.

``y = x @ w^T`` with x ``[M, 5120]`` bf16 and ``w = [wa; wb]`` bf16 (``[NA + NB, 5120]``): the first NA columns are
returned in fp32, the last NB (the indexer's weights_proj, 32) rounded to bf16 (RNE). The sum of a column is the exact
real sum rounded once to fp32 in this contract; the kernel accumulates the products (exact in fp32) in fp32 in four K
quarters and adds the quarters in order, so it differs from the contract by fp32 rounding (relative 1e-6 of the
column's L1 norm; cosine similarity of every output above 0.99999, bf16 outputs within one bf16 ulp). Two runs of the
kernel are bit-identical.
"""

import torch


def gemm_ref(x: torch.Tensor, w: torch.Tensor, na: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (fp32 ``[M, na]``, bf16 ``[M, w.shape[0] - na]``)."""
    y = (x.double() @ w.double().T).float()
    return y[:, :na], y[:, na:].bfloat16()
