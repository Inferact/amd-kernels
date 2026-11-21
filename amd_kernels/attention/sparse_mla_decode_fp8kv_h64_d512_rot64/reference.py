# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``sparse_mla_decode`` (root README, rule 3): torch only.

Inputs. ``q [T, 64, 512]`` e4m3 with one per-tensor fp32 ``q_scale``; two paged e4m3 caches
``[blocks, block, 512]`` (the sliding window's ``main`` and the compressed rows' ``extra``; one
per-tensor fp32 ``kv_scale``, a row is K and V: 448 nope + 64 rope dims, the whole row is V);
per query row t the ragged slot lists ``main_indices[main_indptr[t]:main_indptr[t + 1]]`` and
``extra_indices[...]`` (slot s is row ``(s // block) * block_stride + s % block`` of its cache;
causality inside a speculative window and the window / top-k membership are the lists, so a row
attends exactly its listed keys, both lists as one set); per-head fp32 ``sink`` (``exp(sink)`` joins
the softmax denominator as a key with no value); ``positions [T]`` and ``cos_sin [P, 64]`` fp32
(cos of 32 pairs, then sin).

``attention_reference``: ``S = q k^T * q_scale * kv_scale * scale`` and the softmax over the
listed keys and the sink in fp64 from the exact e4m3 values, ``O = P V * kv_scale``, rounded to
bf16 (RNE). A row with no key gives 0.

``epilogue_reference``: the bf16 rows get the inverse GPT-J RoPE on dims 448..511 (pairs
``(2i, 2i + 1)``: ``fma(even, cos, odd * sin)``, ``fma(odd, cos, -(even * sin))``, fp32, cos / sin of
``positions[t]``), then MXFP8 per 32 consecutive dims of a head: ``amax`` floored at fp32 ``tiny``,
``e = clamp(ceil(log2(amax / 448)) + 127, 0, 254)`` (the division in fp32), codes
``e4m3_rne(x * 2^(127 - e))``, scale byte ``e``; ``codes [T, 32768]``, ``scales [T, 1024]``
row-major, the layout of the wo_a input.

Tolerances (``check_close``): cosine of the dequantized output to the reference per row batch >=
``COS_MIN``; with identical pre-quant values the codes and scales are bit-equal. The kernel is
deterministic: the same inputs give the same bits, whatever the split count.
"""

import torch

NUM_HEADS = 64
D = 512
ROT = 64
FP8_MAX = 448.0
TINY_F32 = torch.finfo(torch.float32).tiny
COS_MIN = 0.9999


def _rows(cache: torch.Tensor, indices: torch.Tensor, block_stride: int | None) -> torch.Tensor:
    block = cache.shape[1]
    stride = block if block_stride is None else block_stride
    idx = indices.long()
    row = (idx // block) * stride + idx % block
    flat = cache.view(torch.float8_e4m3fn).as_strided((cache.shape[0] * stride, D), (D, 1))
    return flat[row]


def attention_reference(
    q, q_scale, main_cache, main_indptr, main_indices, extra_cache, extra_indptr, extra_indices,
    kv_scale, sink, scale, main_block_stride=None, extra_block_stride=None,
) -> torch.Tensor:  # fmt: skip
    """fp64 attention of every row, bf16 ``[T, 64, 512]`` (before the inverse RoPE)."""
    t_rows = q.shape[0]
    out = torch.zeros((t_rows, NUM_HEADS, D), dtype=torch.float64, device=q.device)
    qf = q.view(torch.float8_e4m3fn).double() * float(q_scale)
    mp, xp = main_indptr.tolist(), None if extra_indptr is None else extra_indptr.tolist()
    sinkd = sink.double()
    for t in range(t_rows):
        parts = [_rows(main_cache, main_indices[mp[t] : mp[t + 1]], main_block_stride)]
        if extra_cache is not None:
            parts.append(_rows(extra_cache, extra_indices[xp[t] : xp[t + 1]], extra_block_stride))
        kv = torch.cat(parts).double() * float(kv_scale)
        s = torch.einsum("hd,kd->hk", qf[t], kv) * scale
        s = torch.cat([s, sinkd[:, None]], dim=1)
        p = torch.softmax(s, dim=1)[:, :-1]
        out[t] = p @ kv
    return out.bfloat16()


def mxfp8_quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[rows, K]`` -> (e4m3 codes ``[rows, K]`` uint8, e8m0 scales ``[rows, K // 32]`` uint8)."""
    rows, k = x.shape
    v = x.float().view(rows, k // 32, 32)
    amax = v.abs().amax(-1).clamp_min(TINY_F32)
    e = torch.ceil(torch.log2((amax / FP8_MAX).double())) + 127.0
    e = e.clamp(0.0, 254.0)
    rescale = torch.exp2(127.0 - e).float()
    codes = (v * rescale[..., None]).to(torch.float8_e4m3fn)
    return codes.view(torch.uint8).view(rows, k), e.to(torch.uint8)


def inverse_rope(x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """fp32 ``[T, 64, 512]`` of the bf16 values with the trailing 64 dims rotated back: ``fma(even, cos, odd * sin)``
    and ``fma(odd, cos, -(even * sin))``, the product in the addend rounded to fp32."""
    cs = cos_sin[positions.long()]
    cos, sin = cs[:, None, : ROT // 2], cs[:, None, ROT // 2 :]
    y = x.float().clone()
    even, odd = y[..., D - ROT :: 2].clone(), y[..., D - ROT + 1 :: 2].clone()
    new_even = even.double() * cos.double() + (odd * sin).double()
    new_odd = odd.double() * cos.double() - (even * sin).double()
    y[..., D - ROT :: 2] = new_even.float()
    y[..., D - ROT + 1 :: 2] = new_odd.float()
    return y


def epilogue_reference(out_bf16: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    """(codes ``[T, 32768]`` uint8, scales ``[T, 1024]`` uint8) of bf16 ``[T, 64, 512]``."""
    t_rows = out_bf16.shape[0]
    y = inverse_rope(out_bf16, positions, cos_sin).view(t_rows, NUM_HEADS * D)
    return mxfp8_quant_reference(y)


def dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 ``[T, 32768]`` of an MXFP8 output."""
    t_rows = codes.shape[0]
    v = codes.view(torch.float8_e4m3fn).float().view(t_rows, -1, 32)
    return (v * torch.exp2(scales.float() - 127.0)[..., None]).view(t_rows, -1)


def sparse_mla_decode_reference(*args, positions, cos_sin, **kwargs):
    """The whole op: ``attention_reference`` then ``epilogue_reference``."""
    return epilogue_reference(attention_reference(*args, **kwargs), positions, cos_sin)


def check_close(got_codes, got_scales, ref_attention, positions, cos_sin, name: str) -> float:
    """Cosine of the kernel's dequantized rows to the reference's inverse-RoPE rows (``COS_MIN``)."""
    ref = inverse_rope(ref_attention, positions, cos_sin).flatten(1).double()
    got = dequant(got_codes, got_scales).double()
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=1).min().item()
    assert cos >= COS_MIN, (name, cos)
    return cos
