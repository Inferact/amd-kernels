# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DSA decode prologue: the q_a / kv_a norms with the fp8_ds_mla KV insert (key (q_lora, kv_lora) from ``q_out`` /
``kv_out``) and the indexer's query prep (key (heads, d) from ``q``)."""

import dataclasses
import functools

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, renamed

_ROUTES = Routes(
    "qkv_a_norm_kv_insert",
    {(1536, 512): "amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.qkv_a_norm_kv_insert"},
    ("q_lora", "kv_lora"),
)
_INDEX_ROUTES = Routes("indexer_prep",
                       {(32, 128): "amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.indexer_prep"},
                       ("heads", "d"))
_INSERT = _ROUTES.op
_PREP = _INDEX_ROUTES.op


def _insert_host(q_out, kv_out):
    return _ROUTES.package((q_out.shape[-1], kv_out.shape[-1]))


def _package_args(pkg, qkv_a, q_norm_weight, kv_norm_weight, eps, q_out, kv_out, kv_cache=None, slot_mapping=None,
                  index=None):
    side = None if index is None else pkg.IndexSide(**index)
    return (qkv_a, q_norm_weight, kv_norm_weight, eps, q_out, kv_out, kv_cache, slot_mapping, side)


def qkv_a_norm_kv_insert(qkv_a, q_norm_weight, kv_norm_weight, eps, q_out, kv_out, kv_cache=None, slot_mapping=None,
                         index: dict | None = None) -> None:
    """q_out / kv_out = the RMSNorms of the qkv_a row; with ``kv_cache`` and ``slot_mapping`` the fp8_ds_mla insert;
    ``index`` (a dict of the package's ``IndexSide`` fields) adds the indexer's key insert and head weights."""
    pkg = _insert_host(q_out, kv_out)
    pkg.qkv_a_norm_kv_insert(*_package_args(pkg, qkv_a, q_norm_weight, kv_norm_weight, eps, q_out, kv_out, kv_cache,
                                             slot_mapping, index))


def qkv_a_norm_kv_insert_shapes() -> frozenset:
    return _ROUTES.shapes()


def _layout(op, what):
    return f"{op}: operand layout ({what})"


def _index_cache_ok(cache, page_bytes):
    return (
        cache.dtype == torch.uint8
        and cache.dim() == 3
        and cache.shape[1:] == (64, 68)
        and cache.stride(2) == 1
        and cache.stride(1) == 68
        and cache.stride(0) >= page_bytes
        and cache.stride(0) % 16 == 0
    )


def _slots_ok(slots):
    return slots.dtype == torch.int64 and slots.dim() == 1 and slots.is_contiguous()


def _bf16_vector(t, n):
    return t.dtype == torch.bfloat16 and t.is_contiguous() and t.numel() == n


def qkv_a_norm_kv_insert_blocker(qkv_a, q_norm_weight, kv_norm_weight, eps, q_out, kv_out, kv_cache=None,
                                 slot_mapping=None, index: dict | None = None) -> str | None:
    op = _INSERT
    if q_out.dim() != 2 or kv_out.dim() != 2:
        return _layout(op, "q_out, kv_out [rows, dim]")
    key = (q_out.shape[1], kv_out.shape[1])
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    pkg = _ROUTES.package(key)
    k = pkg.kernel
    insert = kv_cache is not None and slot_mapping is not None
    if not pkg.supports_qkv_a_norm_kv_insert(qkv_a, kv_cache if insert else None):
        return _layout(op, "supports_qkv_a_norm_kv_insert")
    n = qkv_a.shape[0]
    cols = k.ROW + (k.D + k.HEADS if index is not None else 0)
    if qkv_a.shape[1] < cols:
        return _layout(op, f"qkv_a has {qkv_a.shape[1]} columns, the row needs {cols}")
    if not (_bf16_vector(q_norm_weight, k.Q) and _bf16_vector(kv_norm_weight, k.KV)):
        return _layout(op, "norm weights bf16 contiguous [q_lora], [kv_lora]")
    for t, d in ((q_out, k.Q), (kv_out, k.KV)):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.shape != (n, d):
            return _layout(op, "q_out, kv_out bf16 contiguous [rows of qkv_a, dim]")
    if insert and (kv_cache.shape[0] == 0 or not _slots_ok(slot_mapping)):
        return _layout(op, "non-empty kv_cache, int64 contiguous slot_mapping")
    if index is not None:
        fields = {f.name for f in dataclasses.fields(pkg.IndexSide)}
        if set(index) != fields:
            return _layout(op, f"index keys {sorted(index)}, the side has {sorted(fields)}")
        if not _index_cache_ok(index["cache"], pkg.host.PAGE_BYTES):
            return _layout(op, "indexer cache uint8 [pages, 64, 68], page stride >= PAGE_BYTES, % 16")
        if not _slots_ok(index["slot_mapping"]):
            return _layout(op, "index slot_mapping int64 contiguous")
        w = index["weights"]
        if w.dtype != torch.float32 or not w.is_contiguous() or w.shape != (n, k.HEADS):
            return _layout(op, "index weights fp32 contiguous [rows, heads]")
        if not (_bf16_vector(index["ln_weight"], k.D) and _bf16_vector(index["ln_bias"], k.D)):
            return _layout(op, "index ln_weight / ln_bias bf16 contiguous [d]")
    return None


def _side_inputs(pkg, inputs, mode, n, *args, **kwargs):
    return inputs(mode, n, *_package_args(pkg, *args, **kwargs))


@functools.cache
def _insert_kernel(pkg):
    spec = pkg.qkv_a_norm_kv_insert.entry
    run = functools.partial(qkv_a_norm_kv_insert)
    run.entry = dataclasses.replace(spec, inputs=functools.partial(_side_inputs, pkg, spec.inputs))
    return run


def _outs(qkv_a, q_norm_weight, kv_norm_weight, eps, q_out, kv_out, kv_cache=None, slot_mapping=None, index=None):
    return q_out, kv_out


def qkv_a_norm_kv_insert_kernel(*args, **kwargs):
    """The routed ``amd_kernels.warm_up`` entry for these arguments; it and its ``.entry`` warm-up take the dict
    ``index`` as the op does."""
    return _insert_kernel(_insert_host(*_outs(*args, **kwargs)))


def _prep_host(q):
    key = (q.shape[1], q.shape[2]) if q.dim() == 3 else tuple(q.shape[1:])
    return _INDEX_ROUTES.package(key)


def indexer_prep(q, k, w, ln_weight, ln_bias, ln_eps, slot_mapping, cache, topk_out, weight_scale, q_payload, q_scales,
                 weights) -> None:
    """The indexer query's MXFP4 payload / scales and the top-k reset; with ``k`` also the key LayerNorm + MXFP4
    insert and the head weights."""
    _prep_host(q).indexer_prep(q, k, w, ln_weight, ln_bias, ln_eps, slot_mapping, cache, topk_out, weight_scale,
                               q_payload, q_scales, weights)


def indexer_prep_shapes() -> frozenset:
    return _INDEX_ROUTES.shapes()


def indexer_prep_blocker(q, k, w, ln_weight, ln_bias, ln_eps, slot_mapping, cache, topk_out, weight_scale, q_payload,
                         q_scales, weights) -> str | None:
    op = _PREP
    if q.dim() != 3:
        return _layout(op, "q [rows, heads, d]")
    key = (q.shape[1], q.shape[2])
    reason = _INDEX_ROUTES.blocker(key)
    if reason is not None:
        return reason
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    pkg = _INDEX_ROUTES.package(key)
    heads, d = key
    if not pkg.supports_indexer_prep(q, cache, topk_out):
        return _layout(op, "supports_indexer_prep")
    n = q.shape[0]
    if q.stride(0) % 8:
        return _layout(op, "q row stride % 8")
    if q_payload.dtype != torch.uint8 or not q_payload.is_contiguous() or q_payload.shape != (n, heads, d // 2):
        return _layout(op, "q_payload uint8 contiguous [rows, heads, d / 2]")
    if q_scales.dtype != torch.uint8 or not q_scales.is_contiguous() or q_scales.shape != (n, 1, 4, 16, 4):
        return _layout(op, "q_scales uint8 contiguous [rows, 1, 4, 16, 4]")
    if k is None:
        return None
    if any(t is None for t in (w, ln_weight, ln_bias, slot_mapping, weights)):
        return _layout(op, "k given: w, ln_weight, ln_bias, slot_mapping and weights too")
    if k.dtype != torch.bfloat16 or k.dim() != 2 or k.shape[1] != d or k.stride(1) != 1 or k.stride(0) % 2:
        return _layout(op, "k bf16 [keys, d], unit column stride, even row stride")
    if w.dtype != torch.bfloat16 or w.shape != (n, heads) or w.stride(1) != 1:
        return _layout(op, "w bf16 [rows, heads], unit column stride")
    if not (_bf16_vector(ln_weight, d) and _bf16_vector(ln_bias, d)):
        return _layout(op, "ln_weight / ln_bias bf16 contiguous [d]")
    if not _slots_ok(slot_mapping):
        return _layout(op, "slot_mapping int64 contiguous")
    if weights.dtype != torch.float32 or not weights.is_contiguous() or weights.shape != (n, heads):
        return _layout(op, "weights fp32 contiguous [rows, heads]")
    return None


def indexer_prep_kernel(*args, **kwargs):
    """The routed ``amd_kernels.warm_up`` entry (``.entry``) for these arguments."""
    q = args[0] if args else kwargs["q"]
    return _prep_host(q).indexer_prep


qkv_a_norm_kv_insert_unsupported_reason = renamed(qkv_a_norm_kv_insert_blocker,
                                                  "qkv_a_norm_kv_insert_unsupported_reason")
indexer_prep_unsupported_reason = renamed(indexer_prep_blocker, "indexer_prep_unsupported_reason")
