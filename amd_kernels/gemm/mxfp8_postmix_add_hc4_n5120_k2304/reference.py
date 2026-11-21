# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch contract of the shared-expert down GEMM with the routed add and the mHC post-mix epilogue
(``host.postmix_gemm``, ``host.postmix_gates_gemm``).

``y = bf16(bf16(A @ W^T) + routed)`` for pre-quantized ``A`` (E4M3 codes [M, 2304], row-major E8M0 bytes [M, 72]), a weight
of E4M3 codes [5120, 2304] with E8M0 scales [5120, 72] and the routed experts' bf16 output [M, 5120]; both
dequantized exactly, fp32 sums. Then the seam's post-mix
(``norm/mhc_post_pre_rmsnorm_h5120/reference.py``)::

    R'[m, j] = bf16(p[m, j] * y[m] + sum_h c[m, h, j] * R[m, h])

The kernel rounds the GEMM sums to bf16 (the shared output a separate GEMM stores), adds the routed output in fp32 and
rounds to bf16 again, then adds in fp32 with the seam's own fma order
(``p * y``, then ``fma(c[h][j], R[h], .)`` for h = 0 ... 3), so its new residual equals the output of the separate
GEMM followed by the seam's post-mix bit for bit; against the torch reference below (a different GEMM accumulation order,
one bf16 rounding of y) the per-row cosine is >= ``COS_MIN``.

``postmix_gates_gemm`` also returns, for the next seam (``norm/mhc_gate_finish_rmsnorm_h5120``), the collapse
``x1 = bf16(sum_j q_j R'_j)`` (``q`` the carried pre gate, fma chain in stream order as the seam does) and records
``[M, NB * KW, 32]`` whose sums over the records are the gate projection ``flatten(R') @ fn^T`` (slots 0 - 23, bf16
hi + lo of ``fn`` with fp32 accumulation), the sum of squares of R' (slot 24) and of x1 (slot 25); ``mean`` is the mean
of the four new streams. ``partial_sums_reference`` gives those sums.
"""

import torch

N = 5120
K = 2304
BLOCK = 32
COS_MIN = 0.99999


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[R, K] -> E4M3 codes [R, K] and E8M0 exponents [R, K / 32], row-major (the vLLM Triton quantizer's bytes)."""
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = (amax / 448.0).log2().ceil().add(127).clamp(0, 254)
    codes = (blocks * torch.exp2(127 - exponent).unsqueeze(-1)).reshape(rows, k)
    return codes.to(torch.float8_e4m3fn), exponent.to(torch.uint8)


def dequantize_mxfp8(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    rows, k = codes.shape
    scale = torch.exp2(exponents.float() - 127).unsqueeze(-1)
    return (codes.float().reshape(rows, k // BLOCK, BLOCK) * scale).reshape(rows, k)


def post_mix(y: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor, comb_mix: torch.Tensor) -> torch.Tensor:
    """New residual bf16 [M, 4, H] from y bf16 [M, H]."""
    m, ns, h = residual.shape
    p = post_mix.reshape(m, ns, 1).float()
    c = comb_mix.reshape(m, ns, ns).float()
    return (p * y.float().unsqueeze(1) + torch.einsum("mhj,mhk->mjk", c, residual.float())).bfloat16()


def postmix_gemm_reference(codes, scales, weight_codes, weight_scales, residual, routed, post_mix_, comb_mix):
    a = dequantize_mxfp8(codes, scales)
    shared = (a @ dequantize_mxfp8(weight_codes, weight_scales).T).to(torch.bfloat16)
    y = (shared.float() + routed.float()).bfloat16()
    return post_mix(y, residual, post_mix_, comb_mix)


def collapse_reference(residual_new: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """x1 = bf16(sum_j q_j R'_j) bf16 [M, H]."""
    return torch.einsum("mj,mjh->mh", pre_mix.reshape(-1, 4).float(), residual_new.float()).bfloat16()


def partial_sums_reference(residual_new: torch.Tensor, x1: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """[M, 26] fp32: the 24 gate-projection sums, the sums of squares of R' and of x1."""
    flat = residual_new.reshape(residual_new.shape[0], -1).float()
    return torch.cat([flat @ fn.t(), flat.square().sum(1, keepdim=True), x1.float().square().sum(1, keepdim=True)], 1)
