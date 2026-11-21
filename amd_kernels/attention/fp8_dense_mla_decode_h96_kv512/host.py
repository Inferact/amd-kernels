# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry and support check of the DSpark draft dense MLA decode (see ``kernel.py``, ``merge.py``, ``reference.py``)."""

import functools

import torch
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from .kernel import GROUPS, HEADS, KV_LORA_RANK, LOG2E, ROW, XCDS, compile_partials
from .merge import compile_merge

MAX_QLEN = 4
MAX_REQS = 63
NSETS = 85
_ALIGN = 16


@functools.cache
def _partials(qlen, cp_world):
    return compile_partials(qlen, cp_world)


@functools.cache
def _merge(qlen):
    return compile_merge(qlen)


@functools.lru_cache(maxsize=16)
def _is_gfx950(device_index):
    try:
        return torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0] == "gfx950"
    except Exception:  # noqa: BLE001
        return False


_scratch = {}


def _workspace(device, qlen):
    """(partial O bf16, partial (m, l) f32): one slot per KV chunk and request boundary, allocated once per device and
    query length (before graph capture)."""
    key = (device, qlen)
    cur = _scratch.get(key)
    if cur is None:
        slots = NSETS + MAX_REQS
        cur = (
            torch.empty(slots, qlen, HEADS, KV_LORA_RANK, dtype=torch.bfloat16, device=device),
            torch.empty(slots, qlen, HEADS, 2, dtype=torch.float32, device=device),
        )
        _scratch[key] = cur
    return cur


def supports_fp8_dense_mla_decode(q, kv_cache, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale, query_len, out=None):
    """The kernel's case: gfx950, e4m3 q [requests * query_len, 96, 576] (1 <= query_len <= 4, at most 63 requests),
    a flat e4m3 cache of 576-byte rows, int32 CSR token lists, float32 scalar scales on the device."""
    if not (torch.is_tensor(q) and q.is_cuda and q.dim() == 3 and q.dtype == torch.float8_e4m3fn):
        return False
    rows, heads, d = q.shape
    if heads != HEADS or d != ROW or q.stride(2) != 1 or not 1 <= query_len <= MAX_QLEN or rows % query_len:
        return False
    if rows // query_len > MAX_REQS or q.stride(0) % _ALIGN or q.stride(1) % _ALIGN or q.data_ptr() % _ALIGN:
        return False
    device_index = q.device.index if q.device.index is not None else torch.accelerator.current_device_index()
    if not _is_gfx950(device_index):
        return False
    if not torch.is_tensor(kv_cache) or kv_cache.device != q.device or kv_cache.dtype != torch.float8_e4m3fn:
        return False
    if kv_cache.dim() < 2 or kv_cache.shape[-1] != ROW or not kv_cache.is_contiguous() or kv_cache.data_ptr() % _ALIGN:
        return False
    n = rows // query_len
    for t, size in ((kv_indptr, n + 1), (g_kv_indptr, n + 1)):
        if not (torch.is_tensor(t) and t.device == q.device and t.dtype == torch.int32 and t.dim() == 1):
            return False
        if t.numel() != size or not t.is_contiguous():
            return False
    if not (torch.is_tensor(kv_indices) and kv_indices.device == q.device and kv_indices.dtype == torch.int32):
        return False
    if kv_indices.dim() != 1 or not kv_indices.is_contiguous() or kv_indices.numel() == 0:
        return False
    for t in (q_scale, kv_scale):
        if not (torch.is_tensor(t) and t.device == q.device and t.dtype == torch.float32 and t.numel() == 1):
            return False
    if out is not None:
        if out.dtype != torch.bfloat16 or out.device != q.device or out.shape != (rows, HEADS, KV_LORA_RANK):
            return False
        if out.stride(2) != 1 or out.stride(0) % 8 or out.stride(1) % 8 or out.data_ptr() % _ALIGN:
            return False
    return True


def _mode(q, kv_cache, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale, sm_scale, query_len, cp_rank=0, cp_world=8,
          out=None, lse=None):
    return (int(query_len), int(cp_world))


def _sizes(mode):
    return (1, MAX_REQS)


def _inputs(mode, n, q, kv_cache, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale, sm_scale, query_len, cp_rank=0,
            cp_world=8, out=None, lse=None):
    dev = q.device
    zeros = torch.zeros(n + 1, dtype=torch.int32, device=dev)
    qn = torch.zeros((n * query_len, HEADS, ROW), dtype=q.dtype, device=dev)
    return (qn, kv_cache, zeros, torch.zeros(1, dtype=torch.int32, device=dev), zeros.clone(), q_scale, kv_scale, sm_scale,
            query_len, cp_rank, cp_world), {}


@entry("attention/fp8_dense_mla_decode_h96_kv512", mode=_mode, sizes=_sizes, inputs=_inputs)
def fp8_dense_mla_decode(q, kv_cache, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale, sm_scale, query_len, cp_rank=0,
                         cp_world=8, out=None, lse=None):
    """``(out, lse)``: this rank's partial latent attention output [rows, 96, 512] bf16 and the natural-log LSE
    [rows, 96] f32 of ``q`` [requests * query_len, 96, 576] over its round-robin shard of the context (``reference.py``)."""
    assert supports_fp8_dense_mla_decode(q, kv_cache, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale, query_len, out)
    rows = q.shape[0]
    n_req = rows // query_len
    dev = q.device
    if out is None:
        out = torch.empty((rows, HEADS, KV_LORA_RANK), dtype=torch.bfloat16, device=dev)
    if lse is None:
        lse = torch.empty((rows, HEADS), dtype=torch.float32, device=dev)
    if rows == 0:
        return out, lse
    po, pml = _workspace(dev, query_len)
    stream = torch.cuda.current_stream(dev)
    # vLLM's layer._q_scale / _k_scale are 0-d buffers: the compiled launcher takes 1-d tensors (a 0-d one fails its
    # argument fill with "tuple index out of range")
    q_scale, kv_scale = q_scale.reshape(1), kv_scale.reshape(1)
    _run_compiled(
        _partials(query_len, cp_world),
        q,
        kv_cache.view(torch.uint8) if kv_cache.dtype != torch.uint8 else kv_cache,
        kv_indices,
        kv_indptr,
        g_kv_indptr,
        q_scale,
        kv_scale,
        po,
        pml,
        q.stride(0),
        q.stride(1),
        n_req,
        NSETS,
        cp_rank,
        float(sm_scale) * LOG2E,
        XCDS * ((GROUPS * NSETS + XCDS - 1) // XCDS),
        stream,
    )
    _run_compiled(
        _merge(query_len),
        po,
        pml,
        kv_indptr,
        kv_scale,
        out,
        lse,
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        n_req,
        NSETS,
        rows * HEADS // 4,
        stream,
    )
    return out, lse
