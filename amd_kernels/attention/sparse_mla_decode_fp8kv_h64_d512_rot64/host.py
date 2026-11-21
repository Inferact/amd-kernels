# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the sparse MQA decode attention."""

import functools

import numpy as np
import torch

from amd_kernels.warm_up import entry

from . import kernel as _kernel
from . import kernel_hg as _kernel_hg

LOG2E = 1.4426950408889634
# Query-row counts a served step reaches (6 rows per request at 1-128 requests, and the 1..6-row mixes)
MAX_ROWS = 1024


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(p_terms: int, mxfp8: bool, has_extra: bool, splitk: bool):
    return _kernel.compile_sparse_mla_decode(_kernel.STAGES, p_terms, mxfp8, has_extra, splitk)


@functools.cache
def _get_kernel_hg(hg: int, p_terms: int, mxfp8: bool, has_extra: bool):
    return _kernel_hg.compile_sparse_mla_decode_hg(hg, p_terms, mxfp8, has_extra)


@functools.cache
def _on_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


@functools.cache
def _split_buffers(device: torch.device):
    """The split partials (graph capture records no allocation) and the zero tickets (the last split of a row
    resets its own)."""
    part = torch.empty(_kernel.MAX_UNITS * _kernel.UNIT_B, dtype=torch.uint8, device=device)
    return part, torch.zeros(MAX_ROWS, dtype=torch.int32, device=device)


def _cache_args(cache: torch.Tensor) -> tuple[int, int]:
    """(log2 block, block stride in rows) of an e4m3 cache [blocks, block, 512] (or [rows, 512])."""
    if cache.dim() == 2:
        return 0, 1
    block = cache.shape[1]
    assert block & (block - 1) == 0, f"block {block}"
    assert cache.stride(1) == _kernel.D and cache.stride(2) == 1 and cache.stride(0) % _kernel.D == 0
    return block.bit_length() - 1, cache.stride(0) // _kernel.D


def supports_sparse_mla_decode(q: torch.Tensor, main_cache: torch.Tensor, num_heads: int | None = None) -> bool:
    """The kernel's case: gfx950, 64 query heads of 512 dims, e4m3 q and caches of 512-byte rows."""
    return (
        _on_gfx950()
        and q.dim() == 3
        and tuple(q.shape[1:]) == (_kernel.H, _kernel.D)
        and q.dtype in (torch.float8_e4m3fn, torch.uint8)
        and main_cache.dtype in (torch.float8_e4m3fn, torch.uint8)
        and main_cache.shape[-1] == _kernel.D
    )


def plan_hg(rows: int) -> int:
    """Heads per workgroup: the fewest whose workgroups (rows x 64 / hg) fit one wave of the 256 CUs (measured: 16 up
    to 64 rows, 32 up to 128; 64 heads per workgroup run each row's tiles at the L2 -> LDS gather rate)."""
    return 16 if rows <= 64 else 32 if rows <= 128 else 64


def plan_wgs(units_x: int) -> int:
    """Workgroups per XCD: the fewest that keep the makespan of the units, which then run several in a row."""
    rounds = -(-units_x // 32)
    return -(-units_x // rounds)


def plan_splits(rows: int, splits: int | None = None) -> int:
    """Key-range workgroups per row: none unless asked (the head groups need no combine, which costs 5-6 us)."""
    return 1 if splits is None else max(1, min(splits, _kernel.MAX_SPLITS, _kernel.MAX_UNITS // max(rows, 1)))


def _sparse_mla_decode(
    q: torch.Tensor,  # [T, 64, 512] e4m3, contiguous
    main_cache: torch.Tensor,  # [blocks, block, 512] e4m3: the sliding window's rows
    main_indptr: torch.Tensor,  # [T + 1] int32
    main_indices: torch.Tensor,  # int32 slot ids
    extra_cache: torch.Tensor | None,  # the top-k rows' cache, or None (no extra list)
    extra_indptr: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    attn_sink: torch.Tensor,  # [>= 64] fp32
    softmax_scale: float,
    q_scale: torch.Tensor,  # [1] fp32
    kv_scale: torch.Tensor,  # [1] fp32
    positions: torch.Tensor,  # [T] int32 or int64
    cos_sin: torch.Tensor,  # [P, 64] fp32
    out_codes: torch.Tensor | None,  # [T, 32768] e4m3 (or uint8)
    out_scales: torch.Tensor | None,  # [T, 1024] uint8 (or e8m0)
    out_bf16: torch.Tensor | None = None,  # [T, 64, 512] bf16: instead of the MXFP8 output, no RoPE
    *,
    splits: int | None = None,
    p_terms: int = 2,
    hg: int | None = None,
    wgs: int | None = None,
) -> None:  # hg, wgs: heads per workgroup and workgroups per XCD (None: ``plan_hg``, ``plan_wgs``)
    """Attention of every query row over its main list and extra list as one set of keys (``reference.py``), then
    the inverse RoPE of the last 64 dims at ``positions`` and the MXFP8 quantization (e4m3 codes, e8m0 scales per
    32 dims, row-major) into ``out_codes`` / ``out_scales``. A row with no key gets zeros. At most ``MAX_KEYS``
    keys per row (both lists)."""
    rows = q.shape[0]
    if rows == 0:
        return
    assert q.is_contiguous() and q.dtype in (torch.float8_e4m3fn, torch.uint8)
    mxfp8 = out_bf16 is None
    if mxfp8:
        assert out_codes.is_contiguous() and out_scales.is_contiguous()
        assert out_codes.dtype in (torch.float8_e4m3fn, torch.uint8) and out_scales.dtype in (torch.uint8, torch.float8_e8m0fnu)
        assert out_codes.shape == (rows, _kernel.H * _kernel.D) and out_scales.shape == (rows, _kernel.H * _kernel.D // 32)
        out_ptr, osc_ptr = out_codes.data_ptr(), out_scales.data_ptr()
    else:
        assert out_bf16.is_contiguous() and out_bf16.dtype == torch.bfloat16 and out_bf16.shape == q.shape
        out_ptr, osc_ptr = out_bf16.data_ptr(), 0
    for t in (main_indptr, main_indices):
        assert t.dtype == torch.int32 and t.is_contiguous()
    assert positions.dtype in (torch.int32, torch.int64) and positions.is_contiguous() and positions.shape[0] >= rows
    has_extra = extra_cache is not None
    m_lg, m_st = _cache_args(main_cache)
    if has_extra:
        x_lg, x_st = _cache_args(extra_cache)
        xc, xip, xid = extra_cache.data_ptr(), extra_indptr.data_ptr(), extra_indices.data_ptr()
    else:
        x_lg = x_st = xc = xip = xid = 0
    assert cos_sin.dtype == torch.float32 and cos_sin.shape[1] == _kernel.ROPE and cos_sin.stride(1) == 1
    assert attn_sink.dtype == torch.float32 and attn_sink.numel() >= _kernel.H
    S = plan_splits(rows, splits)
    if S == 1:
        hg = plan_hg(rows) if hg is None else hg
        nhg = _kernel.H // hg
        units_x = -(-rows * nhg // 8)
        wgs_x = plan_wgs(units_x) if wgs is None else min(wgs, units_x)
        _run_compiled(
            _get_kernel_hg(hg, p_terms, mxfp8, has_extra),
            q.data_ptr(), main_cache.data_ptr(), xc, main_indptr.data_ptr(), main_indices.data_ptr(), xip, xid,
            attn_sink.data_ptr(), positions.data_ptr(), cos_sin.data_ptr(), out_ptr, osc_ptr, q_scale.data_ptr(),
            kv_scale.data_ptr(), rows, units_x, wgs_x, m_lg, m_st, x_lg, x_st, cos_sin.stride(0), positions.element_size(),
            float(np.float32(softmax_scale) * np.float32(LOG2E)), 8 * wgs_x, torch.cuda.current_stream(q.device),
        )  # fmt: skip
        return
    part, cnt = _split_buffers(q.device)
    rows_x = -(-rows // 8)
    _run_compiled(
        _get_kernel(p_terms, mxfp8, has_extra, S > 1),
        q.data_ptr(),
        main_cache.data_ptr(),
        xc,
        main_indptr.data_ptr(),
        main_indices.data_ptr(),
        xip,
        xid,
        attn_sink.data_ptr(),
        positions.data_ptr(),
        cos_sin.data_ptr(),
        out_ptr,
        osc_ptr,
        q_scale.data_ptr(),
        kv_scale.data_ptr(),
        part.data_ptr(),
        cnt.data_ptr(),
        rows,
        rows_x,
        S,
        m_lg,
        m_st,
        x_lg,
        x_st,
        cos_sin.stride(0),
        positions.element_size(),
        float(np.float32(softmax_scale) * np.float32(LOG2E)),
        8 * rows_x * S,
        torch.cuda.current_stream(q.device),
    )


def _mode(q, main_cache, main_indptr, main_indices, extra_cache, extra_indptr, extra_indices, attn_sink,
          softmax_scale, q_scale, kv_scale, positions, cos_sin, out_codes, out_scales, out_bf16=None, *, splits=None,
          p_terms=2, hg=None, wgs=None):
    """What a call fixes about its kernels besides the row count (``amd_kernels.warm_up``)."""
    return (out_bf16 is None, extra_cache is not None, p_terms, main_cache.dim(), _cache_args(main_cache),
            None if extra_cache is None else _cache_args(extra_cache), splits)


def _inputs(mode, n, q, main_cache, main_indptr, main_indices, extra_cache, extra_indptr, extra_indices, attn_sink,
            softmax_scale, q_scale, kv_scale, positions, cos_sin, out_codes, out_scales, out_bf16=None, *,
            splits=None, p_terms=2, hg=None, wgs=None):
    """``n`` rows with empty key lists (no row reads a key)."""
    dev = q.device
    zi = torch.zeros(n + 1, dtype=torch.int32, device=dev)
    args = (
        torch.zeros((n,) + tuple(q.shape[1:]), dtype=q.dtype, device=dev), main_cache, zi,
        torch.zeros(1, dtype=torch.int32, device=dev), extra_cache, None if extra_cache is None else zi,
        None if extra_cache is None else torch.zeros(1, dtype=torch.int32, device=dev), attn_sink, softmax_scale,
        q_scale, kv_scale, torch.zeros(n, dtype=positions.dtype, device=dev), cos_sin,
        None if out_codes is None else torch.zeros((n,) + tuple(out_codes.shape[1:]), dtype=out_codes.dtype, device=dev),
        None if out_scales is None else torch.zeros((n,) + tuple(out_scales.shape[1:]), dtype=out_scales.dtype, device=dev),
        None if out_bf16 is None else torch.zeros((n,) + tuple(out_bf16.shape[1:]), dtype=out_bf16.dtype, device=dev),
    )  # fmt: skip
    return args, {"splits": splits, "p_terms": p_terms, "hg": hg, "wgs": wgs}


def _sizes(mode):
    return (1, 65, 129) if mode[-1] is None else (1, 2)


sparse_mla_decode = entry("attention/sparse_mla_decode_fp8kv_h64_d512_rot64", mode=_mode, sizes=_sizes,
                          inputs=_inputs)(_sparse_mla_decode)
