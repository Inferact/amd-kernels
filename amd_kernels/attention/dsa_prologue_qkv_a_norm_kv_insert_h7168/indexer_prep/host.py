# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache and launch wrapper of the DSA indexer decode prep."""

import functools
import struct

import torch

from amd_kernels.warm_up import entry

from ..common import on_gfx950
from ..index_key import PAGE_BYTES
from . import kernel as _kernel


def _run_compiled(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(topk: int, keys: bool):
    return _kernel.compile_indexer_prep(topk, keys)


def supports_indexer_prep(q: torch.Tensor, cache: torch.Tensor, topk_out: torch.Tensor) -> bool:
    return (
        on_gfx950()
        and q.dtype == torch.bfloat16
        and q.dim() == 3
        and q.shape[1:] == (_kernel.HEADS, _kernel.D)
        and q.stride(2) == 1
        and q.stride(1) == _kernel.D
        and cache.dtype == torch.uint8
        and cache.dim() == 3
        and cache.shape[1:] == (64, 68)
        and cache.stride(2) == 1
        and cache.stride(1) == 68
        and cache.stride(0) >= PAGE_BYTES
        and cache.stride(0) % 16 == 0
        and topk_out.dtype == torch.int32
        and topk_out.dim() == 2
        and topk_out.stride(1) == 1
        and topk_out.shape[1] % 1024 == 0
    )


def _indexer_prep(
    q: torch.Tensor,  # [n, 32, 128] bf16 (wq_index output), unit head stride 128
    k: torch.Tensor | None,  # [m, 128] bf16 (wk_index output, before the LayerNorm); None: queries only
    w: torch.Tensor | None,  # [n, 32] bf16 (wv_index output)
    ln_weight: torch.Tensor | None,  # [128] bf16
    ln_bias: torch.Tensor | None,  # [128] bf16
    ln_eps: float,
    slot_mapping: torch.Tensor | None,  # int64 [<= m], < 0 skips the key
    cache: torch.Tensor,  # uint8 [pages, 64, 68] indexer pages
    topk_out: torch.Tensor,  # int32 [>= n, topk], set to -1
    weight_scale: float,
    q_payload: torch.Tensor,  # uint8 [n, 32, 64]
    q_scales: torch.Tensor,  # uint8 [n, 1, 4, 16, 4]
    weights: torch.Tensor | None,  # float32 [n, 32]
) -> None:
    """Query MXFP4 and top-k reset; with ``k``: the key LayerNorm + MXFP4 insert of the first len(slot_mapping) keys
    and the head weights (k=None when ``qkv_a_norm_kv_insert`` already did both)."""
    n = q.shape[0]
    n_out = topk_out.shape[0]
    grid = n + n_out
    if grid == 0:
        return
    assert supports_indexer_prep(q, cache, topk_out)
    keys = k is not None
    assert q_payload.shape == (n, _kernel.HEADS, 64) and q_payload.dtype == torch.uint8 and q_payload.is_contiguous()
    assert q_scales.shape == (n, 1, 4, 16, 4) and q_scales.dtype == torch.uint8 and q_scales.is_contiguous()
    assert q.stride(0) % 8 == 0
    if keys:
        assert k.dtype == w.dtype == ln_weight.dtype == ln_bias.dtype == torch.bfloat16
        assert k.shape[1] == _kernel.D and k.stride(1) == 1 and k.stride(0) % 2 == 0
        assert w.shape == (n, _kernel.HEADS) and w.stride(1) == 1
        assert ln_weight.is_contiguous() and ln_bias.is_contiguous() and ln_weight.numel() == _kernel.D
        assert slot_mapping.dtype == torch.int64 and slot_mapping.is_contiguous()
        assert weights.shape == (n, _kernel.HEADS) and weights.dtype == torch.float32 and weights.is_contiguous()
        n_slots = min(k.shape[0], slot_mapping.shape[0])
        ptrs = (k.data_ptr(), w.data_ptr(), ln_weight.data_ptr(), ln_bias.data_ptr(), slot_mapping.data_ptr())
        strides = (k.stride(0), w.stride(0))
        wo_ptr = weights.data_ptr()
    else:
        n_slots, ptrs, strides, wo_ptr = 0, (0, 0, 0, 0, 0), (0, 0), 0
    _run_compiled(
        _get_kernel(topk_out.shape[1], keys),
        q.data_ptr(),
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ptrs[4],
        cache.data_ptr(),
        q_payload.data_ptr(),
        q_scales.data_ptr(),
        wo_ptr,
        topk_out.data_ptr(),
        n,
        n_out,
        n_slots,
        q.stride(0),
        strides[0],
        strides[1],
        topk_out.stride(0),
        struct.unpack("<i", struct.pack("<f", ln_eps))[0],
        struct.unpack("<i", struct.pack("<f", weight_scale))[0],
        cache.stride(0),
        grid,
        torch.cuda.current_stream(),
    )


def _mode(q, k, w, ln_weight, ln_bias, ln_eps, slot_mapping, cache, topk_out, weight_scale, q_payload, q_scales, weights):
    """What a call fixes about its kernel (``amd_kernels.warm_up``; the row count does not)."""
    return (topk_out.shape[1], k is not None)


def _inputs(mode, n, q, k, w, ln_weight, ln_bias, ln_eps, slot_mapping, cache, topk_out, weight_scale, q_payload,
            q_scales, weights):
    """One row (zeros; slot -1 stores no key)."""
    dev = q.device
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=dev)  # noqa: E731
    keys = mode[1]
    return (
        zeros((n, _kernel.HEADS, _kernel.D), torch.bfloat16),
        zeros((n, _kernel.D), torch.bfloat16) if keys else None,
        zeros((n, _kernel.HEADS), torch.bfloat16) if keys else None,
        ln_weight,
        ln_bias,
        ln_eps,
        torch.full((n,), -1, dtype=torch.int64, device=dev) if keys else None,
        cache,
        torch.empty((n, mode[0]), dtype=torch.int32, device=dev),
        weight_scale,
        zeros((n, _kernel.HEADS, 64), torch.uint8),
        zeros((n, 1, 4, 16, 4), torch.uint8),
        zeros((n, _kernel.HEADS), torch.float32) if keys else None,
    ), {}


indexer_prep = entry(
    "attention/dsa_prologue_qkv_a_norm_kv_insert_h7168:indexer_prep", mode=_mode, sizes=lambda mode: (1,), inputs=_inputs
)(_indexer_prep)
