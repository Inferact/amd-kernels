# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract: y = x @ w^T for bf16 x [M, 5120] and the bf16 lm_head weight w [129280, 5120].

The product of the bf16 operands in fp64, rounded once to bf16. The kernel accumulates in fp32 in
a fixed order, so it differs from the reference (and from hipBLASLt) only in the sum order: per row
cosine >= 0.99999 and relative L2 <= 1e-3, deterministic bits. NaN / inf propagate as in an fp32
accumulation (a NaN in a row of x makes that output row NaN).
"""
import torch

N, K = 129280, 5120


def bf16_gemm_n129280_k5120_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.double() @ w.double().t()).bfloat16()
