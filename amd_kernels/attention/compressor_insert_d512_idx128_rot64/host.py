# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the compressor tail and indexer key store."""

import functools
import struct

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
def _get_kernel(ratio: int, ring_rows: int, main_rows: int, idx_rows: int, cs_idx_bf16: bool):
    return _kernel.compile_compressor_insert(ratio, ring_rows, main_rows, idx_rows, cs_idx_bf16)


@functools.cache
def _on_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def shuffle_wk(wk: torch.Tensor) -> torch.Tensor:
    """The load-time layout of ``wk [128, 512]`` bf16 that ``compressor_insert`` takes: ``[column tile, K step, lane, 8]``
    with lane = 16 g + l and element ``wk[16 tile + l, 32 step + 8 g + e]``, so a wave's fragment load is 1 KB in a row.
    Bit-exact (a permutation); the caller replaces the weight once, before graph capture."""
    assert tuple(wk.shape) == (_kernel.KN, _kernel.D) and wk.dtype == torch.bfloat16
    return wk.reshape(8, 16, 16, 4, 8).permute(0, 2, 3, 1, 4).contiguous().reshape(wk.shape)


def _f32_bits(x: float) -> int:
    return struct.unpack("<i", struct.pack("<f", x))[0]


def supports_compressor_insert(raw, ratio, main_cache, idx_cache, wk, cos_sin_main, cos_sin_idx) -> bool:
    """gfx950; fp32 raw rows of ``512 * ratio`` columns, e4m3 main pages ``[pages, rows, 512]``, u8 indexer pages of
    whole 68-byte tokens (the first 64 value bytes then 4 scale bytes of the page's tokens), wk ``[128, 512]`` bf16 in ``shuffle_wk``'s layout."""
    return (
        _on_gfx950()
        and ratio in (1, 2)
        and raw.dtype == torch.float32
        and raw.dim() == 2
        and raw.shape[1] == _kernel.D * ratio
        and raw.stride(1) == 1
        and main_cache.dim() == 3
        and main_cache.shape[2] == _kernel.D
        and main_cache.stride(2) == 1
        and main_cache.stride(1) == _kernel.D
        and main_cache.dtype in (torch.float8_e4m3fn, torch.uint8)
        and idx_cache.dtype == torch.uint8
        and idx_cache[0].numel() % 68 == 0
        and (idx_cache[0].numel() // 68) % 32 == 0
        and tuple(wk.shape) == (_kernel.KN, _kernel.D)
        and wk.dtype == torch.bfloat16
        and wk.is_contiguous()
        and cos_sin_main.dtype == torch.float32
        and cos_sin_main.shape[1] == _kernel.ROT
        and cos_sin_idx.dtype in (torch.float32, torch.bfloat16)
        and cos_sin_idx.shape[1] == _kernel.ROT
    )


def _compressor_insert(
    raw: torch.Tensor,  # [n, 512 * ratio] fp32, unit column stride
    positions: torch.Tensor,  # [n] int64
    ring: torch.Tensor | None,  # [blocks, rows, 1024] fp32 (ratio 2), updated
    ring_slots: torch.Tensor,  # [n] int64 (ratio 1: the main-cache slots)
    query_start_loc: torch.Tensor | None,  # [reqs + 1] int32 (ratio 2)
    token_to_req: torch.Tensor | None,  # [n] int32 (ratio 2)
    norm_weight: torch.Tensor,  # [512] bf16
    eps: float,
    main_cache: torch.Tensor,  # [pages, rows, 512] e4m3, updated
    main_slots: torch.Tensor,  # [n] int64
    cos_sin_main: torch.Tensor,  # [rows, 64] fp32
    kv_scale: torch.Tensor,  # fp32 scalar
    wk: torch.Tensor,  # [128, 512] bf16, shuffle_wk's layout
    k_norm_weight: torch.Tensor,  # [128] bf16
    k_eps: float,
    idx_cache: torch.Tensor,  # u8 indexer pages, updated
    idx_slots: torch.Tensor,  # [n] int64
    cos_sin_idx: torch.Tensor,  # [rows, 64] fp32 or bf16
    ratio: int,
) -> None:
    """Pools the compressor rows of the group-closing tokens into latents, inserts them into the main cache (e4m3) and,
    through ``wk``, the indexer cache (MXFP4), and saves the ring (ratio 2); one launch."""
    n = raw.shape[0]
    if n == 0:
        return
    for t in (positions, ring_slots, main_slots, idx_slots):
        assert t.dtype == torch.int64 and t.is_contiguous()
    assert norm_weight.is_contiguous() and k_norm_weight.is_contiguous() and cos_sin_main.is_contiguous()
    assert cos_sin_idx.is_contiguous() and kv_scale.dtype == torch.float32
    pages = idx_cache.view(idx_cache.shape[0], -1)
    entries = pages.shape[1] // 68
    if ratio == 2:
        assert ring.dtype == torch.float32 and ring.shape[2] == 2 * _kernel.D and ring.stride(2) == 1
        assert query_start_loc.dtype == token_to_req.dtype == torch.int32
        ring_ptr, rows, blk, row = ring.data_ptr(), ring.shape[1], ring.stride(0), ring.stride(1)
        qsl_ptr, t2r_ptr = query_start_loc.data_ptr(), token_to_req.data_ptr()
    else:
        ring_ptr = qsl_ptr = t2r_ptr = raw.data_ptr()
        rows = blk = row = 1
    _run_compiled(
        _get_kernel(ratio, rows, main_cache.shape[1], entries, cos_sin_idx.dtype == torch.bfloat16),
        raw.data_ptr(),
        positions.data_ptr(),
        ring_ptr,
        ring_slots.data_ptr(),
        qsl_ptr,
        t2r_ptr,
        norm_weight.data_ptr(),
        main_cache.data_ptr(),
        main_slots.data_ptr(),
        cos_sin_main.data_ptr(),
        kv_scale.data_ptr(),
        wk.data_ptr(),
        k_norm_weight.data_ptr(),
        pages.data_ptr(),
        idx_slots.data_ptr(),
        cos_sin_idx.data_ptr(),
        n,
        cos_sin_main.shape[0],
        cos_sin_idx.shape[0],
        raw.stride(0),
        blk,
        row,
        main_cache.stride(0) * main_cache.element_size(),
        pages.stride(0),
        _f32_bits(eps),
        _f32_bits(k_eps),
        torch.cuda.current_stream(raw.device),
    )


def _mode(raw, positions, ring, ring_slots, query_start_loc, token_to_req, norm_weight, eps, main_cache, main_slots,
          cos_sin_main, kv_scale, wk, k_norm_weight, k_eps, idx_cache, idx_slots, cos_sin_idx, ratio):
    """What a call fixes about its kernel (``amd_kernels.warm_up``; the row count does not)."""
    pages = idx_cache.view(idx_cache.shape[0], -1)
    return (ratio, ring.shape[1] if ratio == 2 else 1, main_cache.shape[1], pages.shape[1] // 68,
            cos_sin_idx.dtype == torch.bfloat16)


def _inputs(mode, n, raw, positions, ring, ring_slots, query_start_loc, token_to_req, norm_weight, eps, main_cache,
            main_slots, cos_sin_main, kv_scale, wk, k_norm_weight, k_eps, idx_cache, idx_slots, cos_sin_idx, ratio):
    """``n`` zero rows with every slot -1 (writes nothing)."""
    dev = raw.device
    neg = torch.full((n,), -1, dtype=torch.int64, device=dev)
    return (
        torch.zeros((n, raw.shape[1]), dtype=raw.dtype, device=dev),
        torch.zeros(n, dtype=torch.int64, device=dev),
        ring,
        neg,
        torch.zeros(2, dtype=torch.int32, device=dev) if ratio == 2 else None,
        torch.zeros(n, dtype=torch.int32, device=dev) if ratio == 2 else None,
        norm_weight,
        eps,
        main_cache,
        neg,
        cos_sin_main,
        kv_scale,
        wk,
        k_norm_weight,
        k_eps,
        idx_cache,
        neg,
        cos_sin_idx,
        ratio,
    ), {}


compressor_insert = entry("attention/compressor_insert_d512_idx128_rot64", mode=_mode, sizes=lambda mode: (16,),
                          inputs=_inputs)(_compressor_insert)
