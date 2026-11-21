# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shape-agnostic vocabulary-sharded embedding lookup: reads (hidden, world size) from the shard and the group and
builds that shape's package's ``VocabGather`` (``_SHAPE_HOSTS``, imported on first use). The caller calls the
returned object (``gather(ids)``, ``warm_up()``, ``close()``); the vocabulary is a runtime argument of every package."""

import functools
import importlib
import sys

import torch
import torch.distributed as dist

_SHAPE_HOSTS = {
    (6144, 2): "amd_kernels.embedding.vocab_gather_h6144",
    (6144, 4): "amd_kernels.embedding.vocab_gather_h6144",
    (7168, 8): "amd_kernels.embedding.vocab_gather_h7168_v163840_ranks8",
}
_PACKAGES = tuple(sorted(set(_SHAPE_HOSTS.values())))


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable."""
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def _shape_host(hidden: int, world: int):
    name = _SHAPE_HOSTS.get((hidden, world))
    return importlib.import_module(name) if name is not None else None


def _key(weight: torch.Tensor, group) -> tuple[int, int]:
    return (weight.shape[-1] if weight.dim() else 0, dist.get_world_size(group))


def _no_kernel(op: str, key: tuple[int, int]) -> str:
    return f"{op}: no kernel for hidden={key[0]}, world={key[1]} (covered: {sorted(_SHAPE_HOSTS)})"


def vocab_gather_shapes() -> frozenset[tuple[int, int]]:
    """The (hidden, world size) keys with a package."""
    return frozenset(_SHAPE_HOSTS)


def vocab_gather_comm(weight: torch.Tensor, rank: int, group):
    """The package's ``VocabGather`` over this rank's bf16 shard [V / world, hidden]: collective on ``group`` (every
    rank, same order, after the weights load, before graph capture)."""
    key = _key(weight, group)
    pkg = _shape_host(*key)
    if pkg is None:
        raise ValueError(_no_kernel("vocab_gather_comm", key))
    return pkg.VocabGather(weight, rank, group)


def vocab_gather_comm_unsupported_reason(weight: torch.Tensor, rank: int, group) -> str | None:
    """None when ``vocab_gather_comm`` can build a gather for this shard on this rank, else why not (host checks
    only; no collective)."""
    op = "vocab_gather_comm"
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    if not dist.is_initialized():
        return f"{op}: torch.distributed is not initialized"
    key = _key(weight, group)
    if _shape_host(*key) is None:
        return _no_kernel(op, key)
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or not weight.is_contiguous():
        return f"{op}: weight {weight.dtype} {tuple(weight.shape)} (bf16 [vocab shard, {key[0]}], contiguous)"
    if not weight.is_cuda:
        return f"{op}: weight on {weight.device} (a CUDA device)"
    if weight.device.index != torch.cuda.current_device():
        return f"{op}: weight on {weight.device}, current device cuda:{torch.cuda.current_device()}"
    if dist.get_rank(group) != rank:
        return f"{op}: rank {rank}, this process is rank {dist.get_rank(group)} of the group"
    return None


def vocab_gather_unsupported_reason(gather, ids: torch.Tensor) -> str | None:
    """None when ``gather(ids)`` (an object ``vocab_gather_comm`` built) serves these ids, else why not."""
    op = "vocab_gather"
    if not any(type(gather) is getattr(sys.modules.get(name), "VocabGather", None) for name in _PACKAGES):
        return f"{op}: {type(gather).__name__} was not built by vocab_gather_comm"
    if ids.dtype not in (torch.int32, torch.int64) or ids.dim() != 1 or not ids.is_contiguous():
        return f"{op}: ids {ids.dtype} {tuple(ids.shape)} (int32 / int64 [tokens], contiguous)"
    if ids.device != gather.weight.device:
        return f"{op}: ids on {ids.device}, weight on {gather.weight.device}"
    return None
