# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the fused query RoPE / e4m3 quant and KV RoPE / e4m3 insert."""

import functools

import torch

from amd_kernels.warm_up import entry

from . import kernel as _kernel


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(block_size: int):
    return _kernel.compile_rope_fp8_kv_insert(block_size)


@functools.cache
def _on_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def supports_rope_fp8_kv_insert(q: torch.Tensor, kv_cache: torch.Tensor, cos_sin_cache: torch.Tensor) -> bool:
    return (
        _on_gfx950()
        and q.dtype == torch.bfloat16
        and tuple(q.shape[1:]) == (_kernel.H, _kernel.D)
        and kv_cache.dtype in (torch.float8_e4m3fn, torch.uint8)
        and kv_cache.dim() == 3
        and kv_cache.shape[2] == _kernel.D
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.shape[1] == _kernel.ROT
    )


def _rope_fp8_kv_insert(
    q: torch.Tensor,  # [n, 64, 512] bf16, contiguous
    kv: torch.Tensor,  # [n, 512] bf16, contiguous
    q_fp8: torch.Tensor,  # [n, 64, 512] e4m3, written
    kv_cache: torch.Tensor,  # [pages, block_size, 512] e4m3
    slot_mapping: torch.Tensor,  # [>= n] int64, < 0 skips
    positions: torch.Tensor,  # [n] int64
    cos_sin_cache: torch.Tensor,  # [rows, 64] fp32
    kv_scale: torch.Tensor,  # fp32 scalar
    q_scale_inv: torch.Tensor,  # fp32 scalar
    n_insert: int | None = None,
) -> None:
    """q_fp8 = e4m3(RoPE(q) * q_scale_inv); the first ``n_insert`` KV rows (None: all) with a slot go into the cache
    as e4m3(RoPE(kv) / kv_scale)."""
    n = q.shape[0]
    if n == 0:
        return
    assert q.is_contiguous() and kv.is_contiguous() and q_fp8.is_contiguous()
    assert tuple(kv.shape) == (n, _kernel.D) and positions.dtype == torch.int64 and positions.is_contiguous()
    assert slot_mapping.dtype == torch.int64 and slot_mapping.is_contiguous() and slot_mapping.shape[0] >= n
    assert cos_sin_cache.is_contiguous() and kv_scale.dtype == q_scale_inv.dtype == torch.float32
    assert kv_cache.stride(2) == 1 and kv_cache.stride(1) == _kernel.D
    _run_compiled(
        _get_kernel(kv_cache.shape[1]),
        q.data_ptr(),
        kv.data_ptr(),
        q_fp8.data_ptr(),
        kv_cache.data_ptr(),
        slot_mapping.data_ptr(),
        positions.data_ptr(),
        cos_sin_cache.data_ptr(),
        kv_scale.data_ptr(),
        q_scale_inv.data_ptr(),
        n,
        cos_sin_cache.shape[0],
        kv_cache.stride(0) * kv_cache.element_size(),
        n if n_insert is None else n_insert,
        torch.cuda.current_stream(q.device),
    )


def _mode(q, kv, q_fp8, kv_cache, *args, **kwargs):
    """What a call fixes about its kernel (``amd_kernels.warm_up``; the row count does not)."""
    return (kv_cache.shape[1],)


def _inputs(mode, n, q, kv, q_fp8, kv_cache, slot_mapping, positions, cos_sin_cache, kv_scale, q_scale_inv,
            n_insert=None):
    """``n`` rows of zeros with slot -1 (writes no cache byte)."""
    dev = q.device
    return (
        torch.zeros((n, *q.shape[1:]), dtype=q.dtype, device=dev),
        torch.zeros((n, kv.shape[1]), dtype=kv.dtype, device=dev),
        torch.empty((n, *q.shape[1:]), dtype=torch.float8_e4m3fn, device=dev),
        kv_cache,
        torch.full((n,), -1, dtype=torch.int64, device=dev),
        torch.zeros(n, dtype=torch.int64, device=dev),
        cos_sin_cache,
        kv_scale,
        q_scale_inv,
    ), {}


rope_fp8_kv_insert = entry(
    "attention/rope_fp8_kv_insert_h64_d512_rot64", mode=_mode, sizes=lambda mode: (8,), inputs=_inputs
)(_rope_fp8_kv_insert)
