# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The dense MXFP8 decoder layers around the caller's attention on a TP group (the split input fc, embedding + norm +
quant, the all-reduce + norm tails, the MLP blocks, the context K / V chain): routes the communicator block by
(hidden, world), read from the fc weight [hidden, 5 * hidden] and the group, and a layer's weights by (hidden,
intermediate per rank), read from its down weight, to the shape package (``_ROUTES`` / ``_LAYERS``, imported on first
use). The ops are the block's methods (``fc``, ``forward``, ``store_context_kv``, ``capture``)."""

import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare

_ROUTES = Routes("dense_decoder", {
    (6144, 4): "amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4",
    (6144, 2): "amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2",
}, ("hidden", "world"))
_LAYERS = Routes("dense_decoder_layer", {
    (6144, 3072): "amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4",
    (6144, 6144): "amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2",
}, ("hidden", "intermediate"))
_OP = _ROUTES.op


def _key(fc_weight, world):
    return (fc_weight.shape[0] if fc_weight.dim() == 2 else 0, world)


def _block_class(pkg, world):
    return getattr(pkg, f"DenseDecoderRanks{world}")


def dense_decoder_shapes() -> frozenset[tuple[int, int]]:
    """The (hidden, world) with a block."""
    return _ROUTES.shapes()


def dense_decoder_layer(input_norm, qkv_shape, post_norm, gate_up_w, gate_up_ws, down_w, down_ws):
    """One layer's weights for the block's ``forward`` (bf16 norms, preshuffled MXFP8 gate_up / down with scales, the
    (N, K) of the qkv GEMM): the routed package's ``DenseLayer``."""
    return _LAYERS.package(tuple(down_w.shape)).DenseLayer(input_norm, tuple(qkv_shape), post_norm, gate_up_w,
                                                           gate_up_ws, down_w, down_ws)


def dense_decoder_stack_blocker(world: int, max_rows: int, on_library, fc_weight, embedding, layers) -> str | None:
    """None when the block of ``world`` ranks runs this layer stack: up to ``max_rows`` rows per call, ``on_library``
    per linear whether it runs ``gemm.mxfp8_common`` on the block's codes, the replicated fc weight, this rank's
    embedding shard and every layer's ``dense_decoder_layer`` (a load-time check)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(fc_weight, world)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    reason = _ROUTES.package(key).stack_blocker(world, max_rows, on_library, fc_weight, embedding, layers)
    return None if reason is None else f"{_OP}: {reason}"


def dense_decoder(rank: int, group, device, *, fc_weight, embedding, max_rows: int | None = None):
    """The routed package's block (collective on the group, warmed up); ``max_rows`` defaults to the package's
    bound. The caller may free the replicated ``fc_weight`` afterwards."""
    world = dist.get_world_size(group)
    pkg = _ROUTES.package(_key(fc_weight, world))
    return _block_class(pkg, world)(rank, group, device, fc_weight=fc_weight, embedding=embedding,
                                    max_rows=pkg.MAX_ROWS if max_rows is None else max_rows)


def dense_decoder_blocker(rank: int, group, device, *, fc_weight, embedding, max_rows: int | None = None) -> str | None:
    """None when ``dense_decoder`` with these arguments builds the block, else why not."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    world = dist.get_world_size(group)
    key = _key(fc_weight, world)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    pkg = _ROUTES.package(key)
    h, fc_k = key[0], _ROUTES.module(key, "host").FC_K
    if tuple(fc_weight.shape) != (h, fc_k) or fc_weight.dtype != torch.bfloat16:
        return f"{_OP}: fc weight {tuple(fc_weight.shape)} {fc_weight.dtype} (bf16 [{h}, {fc_k}])"
    e = embedding
    if e.dim() != 2 or e.shape[1] != h or e.dtype != torch.bfloat16 or not e.is_contiguous():
        return f"{_OP}: embedding {tuple(e.shape)} {e.dtype} (contiguous bf16 [V / {world}, {h}])"
    rows = pkg.MAX_ROWS if max_rows is None else max_rows
    if not 1 <= rows <= pkg.MAX_ROWS:
        return f"{_OP}: {rows} rows per call (1-{pkg.MAX_ROWS})"
    if dist.get_rank(group) != rank:
        return f"{_OP}: rank {rank} is not this process's rank {dist.get_rank(group)} in the group"
    return None


def prepare_dense_decoder(rank: int, group, device, *, fc_weight, embedding, max_rows: int | None = None):
    """``dense_decoder(...)``, or the reason it cannot be built."""
    kw = dict(fc_weight=fc_weight, embedding=embedding, max_rows=max_rows)
    return prepare(dense_decoder_blocker(rank, group, device, **kw), lambda: dense_decoder(rank, group, device, **kw))


def dense_decoder_context_kv_blocker(block, proj_on_library: bool, proj_w, rotary_dim: int, neox: bool,
                                     cos_sin_cache, kv_caches, k_scales, v_scales) -> str | None:
    """None when ``block.store_context_kv`` stores these layers' context K / V (the projection on
    ``gemm.mxfp8_common``, the RoPE, the e4m3 caches and their fp32 scales), else why not (a check once the caches
    are bound)."""
    op = "dense_decoder_context_kv"
    key = _ROUTES.owner(type(block))
    if key is None:
        return f"{op}: block {type(block).__qualname__} not built by dense_decoder"
    if not kv_caches:
        return f"{op}: no KV caches"
    reason = _ROUTES.package(key).context_kv_blocker(proj_on_library, proj_w, rotary_dim, neox, cos_sin_cache,
                                                     kv_caches, k_scales, v_scales)
    return None if reason is None else f"{op}: {reason}"
