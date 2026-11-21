# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry and support check of the DCP sparse MLA decode (see ``kernel_row.py`` and ``kernel_tile.py``)."""

import functools

import torch
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from ._primitives import HEADS, KV_LORA_RANK, QK_ROPE_HEAD_DIM, RECORD_BYTES
from .kernel_row import compile_decode_row
from .kernel_tile import compile_decode_tile
from .reference import Q_RECORD_BYTES

LOG2E = 1.4426950408889634
D_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM
_MAX_SLOTS = 1 << 29

_ROW_MAX = 42  # rows up to this run the whole-row kernel, larger counts the tile-loop kernel


def _kernel(rows):
    return "row" if rows <= _ROW_MAX else "tile"


@functools.cache
def _launcher(kind, pre_quantized):
    return (compile_decode_row if kind == "row" else compile_decode_tile)(pre_quantized)


@functools.lru_cache(maxsize=16)
def _is_gfx950(device_index):
    try:
        return torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0] == "gfx950"
    except Exception:  # noqa: BLE001
        return False


def _kv_rows(kv):
    """The cache as a [N, 656] uint8 view of flat records, or None."""
    if kv.element_size() != 1 or kv.shape[-1] != RECORD_BYTES or kv.stride(-1) != 1:
        return None
    if kv.dim() > 2 and kv.shape[-2] != 1 and kv.stride(-2) != RECORD_BYTES:
        return None
    rows = kv.reshape(-1, RECORD_BYTES)
    if rows.shape[0] > 1 and rows.stride(0) != RECORD_BYTES:
        return None
    return rows if rows.dtype == torch.uint8 else rows.view(torch.uint8)


def _supports_q(q, q8):
    if q8 is not None:
        if not (torch.is_tensor(q8) and q8.is_cuda and q8.dim() == 3 and q8.dtype == torch.uint8):
            return None
        rows, heads, d = q8.shape
        ok = heads == HEADS and d == Q_RECORD_BYTES and q8.stride(2) == 1
        ok = ok and q8.stride(0) % 16 == 0 and q8.stride(1) % 16 == 0 and q8.data_ptr() % 16 == 0
        return (q8, rows) if ok else None
    if not (torch.is_tensor(q) and q.is_cuda and q.dim() == 3 and q.dtype == torch.bfloat16):
        return None
    rows, heads, d = q.shape
    if heads != HEADS or d != D_QK or q.stride(2) != 1 or q.stride(0) % 2 or q.stride(1) % 2 or q.data_ptr() % 4:
        return None
    return q, rows


def supports_fp8_sparse_mla_decode(q, kv_cache, indptr, slots, out=None, q8=None) -> bool:
    """The kernel's case: gfx950, bf16 q [rows, 96, 576] (or the pre-quantized ``q8``), fp8_ds_mla records of 656 B at a
    flat pitch, int32 CSR."""
    got = _supports_q(q, q8)
    if got is None:
        return False
    qt, rows = got
    device_index = qt.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    if not _is_gfx950(device_index):
        return False
    if not torch.is_tensor(kv_cache) or kv_cache.device != qt.device:
        return False
    kv = _kv_rows(kv_cache)
    if kv is None or kv.data_ptr() % 16:
        return False
    for t in (indptr, slots):
        if not torch.is_tensor(t) or t.device != qt.device or t.dim() != 1 or t.dtype != torch.int32:
            return False
        if not t.is_contiguous():
            return False
    if indptr.numel() != rows + 1 or slots.numel() >= _MAX_SLOTS:
        return False
    if out is not None:
        if out.dtype != torch.bfloat16 or out.device != qt.device or out.dim() != 3 or out.shape[:2] != (rows, HEADS):
            return False
        if out.shape[2] < KV_LORA_RANK or out.stride(2) != 1 or out.stride(0) % 2 or out.stride(1) % 2:
            return False
        if out.data_ptr() % 4:
            return False
    return True


def _mode(q, kv_cache, indptr, slots, sm_scale, out=None, q8=None):
    return ("pq",) if q8 is not None else ()


def _sizes(mode):
    return [1, _ROW_MAX + 1]


def _inputs(mode, n, q, kv_cache, indptr, slots, sm_scale, out=None, q8=None):
    dev = (q8 if q8 is not None else q).device
    zeros_idx = (torch.zeros(n + 1, dtype=torch.int32, device=dev), torch.zeros(1, dtype=torch.int32, device=dev))
    if q8 is not None:
        return (None, kv_cache, *zeros_idx, sm_scale), {"q8": torch.zeros((n, HEADS, Q_RECORD_BYTES), dtype=torch.uint8, device=dev)}
    return (torch.zeros((n, HEADS, D_QK), dtype=q.dtype, device=dev), kv_cache, *zeros_idx, sm_scale), {}


@entry("attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048", mode=_mode, sizes=_sizes, inputs=_inputs)
def fp8_sparse_mla_decode(q, kv_cache, indptr, slots, sm_scale, out=None, q8=None):
    """``(out, lse)``: this rank's partial latent attention output [rows, 96, 512] bf16 and natural-log LSE [rows, 96]
    f32 for q [rows, 96, 576] bf16 (or the pre-quantized ``q8`` [rows, 96, 1168] uint8 of ``reference.quantize_q``)
    over the CSR slots of an fp8_ds_mla cache (``reference.py``)."""
    assert supports_fp8_sparse_mla_decode(q, kv_cache, indptr, slots, out, q8)
    qt = q8 if q8 is not None else q
    rows = qt.shape[0]
    if out is None:
        out = torch.empty((rows, HEADS, KV_LORA_RANK), dtype=torch.bfloat16, device=qt.device)
    lse = torch.empty((rows, HEADS), dtype=torch.float32, device=qt.device)
    if rows == 0:
        return out, lse
    nnz = slots.numel()
    if nnz == 0:
        slots = torch.zeros(1, dtype=torch.int32, device=qt.device)
    _run_compiled(
        _launcher(_kernel(rows), q8 is not None),
        qt,
        _kv_rows(kv_cache),
        slots,
        indptr,
        out,
        lse,
        rows,
        qt.stride(0),
        qt.stride(1),
        out.stride(0),
        out.stride(1),
        float(sm_scale * LOG2E),
        nnz * 4,
        torch.cuda.current_stream(qt.device),
    )
    return out, lse
