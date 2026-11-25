# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shape-agnostic vocabulary-sharded embedding lookup: reads (hidden, world size) from the shard and the group and
builds that shape's package's ``VocabGather`` (``_ROUTES``, imported on first use). The caller calls the
returned object (``gather(ids)``, ``warm_up()``, ``close()``); the vocabulary is a runtime argument of every package."""

import sys

import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare, renamed

_ROUTES = Routes("vocab_gather_comm", {
    (6144, 2): "amd_kernels.embedding.vocab_gather_h6144",
    (6144, 4): "amd_kernels.embedding.vocab_gather_h6144",
    (7168, 8): "amd_kernels.embedding.vocab_gather_h7168_v163840_ranks8",
}, ("hidden", "world"))
_PACKAGES = tuple(sorted(set(_ROUTES.hosts.values())))


def _key(weight: torch.Tensor, group) -> tuple[int, int]:
    return (weight.shape[-1] if weight.dim() else 0, dist.get_world_size(group))


def vocab_gather_shapes() -> frozenset[tuple[int, int]]:
    """The (hidden, world size) keys with a package."""
    return _ROUTES.shapes()


def vocab_gather_comm(weight: torch.Tensor, rank: int, group):
    """The package's ``VocabGather`` over this rank's bf16 shard [V / world, hidden]: collective on ``group`` (every
    rank, same order, after the weights load, before graph capture)."""
    return _ROUTES.package(_key(weight, group)).VocabGather(weight, rank, group)


def vocab_gather_comm_blocker(weight: torch.Tensor, rank: int, group) -> str | None:
    """None when ``vocab_gather_comm`` can build a gather for this shard on this rank, else why not (host checks
    only; no collective)."""
    op = _ROUTES.op
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    if not dist.is_initialized():
        return f"{op}: torch.distributed is not initialized"
    key = _key(weight, group)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or not weight.is_contiguous():
        return f"{op}: weight {weight.dtype} {tuple(weight.shape)} (bf16 [vocab shard, {key[0]}], contiguous)"
    if not weight.is_cuda:
        return f"{op}: weight on {weight.device} (a CUDA device)"
    if weight.device.index != torch.cuda.current_device():
        return f"{op}: weight on {weight.device}, current device cuda:{torch.cuda.current_device()}"
    if dist.get_rank(group) != rank:
        return f"{op}: rank {rank}, this process is rank {dist.get_rank(group)} of the group"
    return None


def prepare_vocab_gather_comm(weight: torch.Tensor, rank: int, group):
    """``vocab_gather_comm(...)``, or the reason it cannot be built."""
    return prepare(vocab_gather_comm_blocker(weight, rank, group), lambda: vocab_gather_comm(weight, rank, group))


def vocab_gather_blocker(gather, ids: torch.Tensor) -> str | None:
    """None when ``gather(ids)`` (an object ``vocab_gather_comm`` built) serves these ids, else why not."""
    op = "vocab_gather"
    if not any(type(gather) is getattr(sys.modules.get(name), "VocabGather", None) for name in _PACKAGES):
        return f"{op}: {type(gather).__name__} was not built by vocab_gather_comm"
    if ids.dtype not in (torch.int32, torch.int64) or ids.dim() != 1 or not ids.is_contiguous():
        return f"{op}: ids {ids.dtype} {tuple(ids.shape)} (int32 / int64 [tokens], contiguous)"
    if ids.device != gather.weight.device:
        return f"{op}: ids on {ids.device}, weight on {gather.weight.device}"
    return None


vocab_gather_comm_unsupported_reason = renamed(vocab_gather_comm_blocker, "vocab_gather_comm_unsupported_reason")
vocab_gather_unsupported_reason = renamed(vocab_gather_blocker, "vocab_gather_unsupported_reason")
