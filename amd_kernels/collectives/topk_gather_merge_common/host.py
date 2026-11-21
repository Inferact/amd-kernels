# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The DCP score all-gather + global top-k merge + CSR slots, routed by (top-k, world size).

The factory builds the shape package's communicator; the caller calls it (``comm(scores, local_idx, ...)``), whose
reason routes by the communicator's type.
"""

import functools
import importlib

import torch
import torch.distributed as dist

_OP = "topk_gather_merge"
_SHAPE_HOSTS = {(2048, 8): "amd_kernels.collectives.dcp8_topk_gather_merge_top2048"}


@functools.cache
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


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


def _covered() -> str:
    return ", ".join(map(str, sorted(_SHAPE_HOSTS)))


def topk_gather_merge_comm(rank, group, device, *, topk):
    """The collective communicator for ``topk`` over ``group`` (every rank calls it); the caller calls ``.warmup()``
    before capture, the communicator per call and ``.close()`` after its graphs are retired."""
    key = (topk, dist.get_world_size(group))
    host = _shape_host(key)
    if host is None:
        raise ValueError(f"{_OP}_comm: no kernel for topk={key[0]}, world={key[1]} (covered: {_covered()})")
    return host.TopkGatherMerge(rank, group, device)


def topk_gather_merge_comm_unsupported_reason(rank, group, device, *, topk) -> str | None:
    """None when ``topk_gather_merge_comm`` builds a communicator for these arguments, else why not."""
    if not available():
        return f"{_OP}_comm: needs gfx950 with FlyDSL"
    if not dist.is_initialized():
        return f"{_OP}_comm: torch.distributed is not initialized"
    world = dist.get_world_size(group)
    if (topk, world) not in _SHAPE_HOSTS:
        return f"{_OP}_comm: no kernel for topk={topk}, world={world} (covered: {_covered()})"
    if rank != dist.get_rank(group):
        return f"{_OP}_comm: rank {rank} is not this process's rank {dist.get_rank(group)} in the group"
    return None


def topk_gather_merge_shapes() -> frozenset:
    """The covered (top-k, world size)."""
    return frozenset(_SHAPE_HOSTS)


@functools.cache
def _comm_key(cls):
    for key, name in _SHAPE_HOSTS.items():
        if cls.__module__ == name or cls.__module__.startswith(name + "."):
            return key
    return None


def topk_gather_merge_unsupported_reason(comm, scores, local_idx, block_table, out_slots, out_indptr, page_size,
                                         block_stride_rows) -> str | None:
    """None when ``comm(scores, local_idx, ...)`` serves this call, else why not (host checks only)."""
    key = _comm_key(type(comm))
    if key is None:
        return f"{_OP}: {type(comm).__name__} is not a communicator of this entry"
    if comm.closed:
        return f"{_OP}: communicator closed"
    topk = scores.shape[1] if scores.ndim == 2 else None
    if topk != key[0]:
        return f"{_OP}: communicator built for topk={key[0]}, world={key[1]}, call is topk={topk}"
    rows, max_rows = scores.shape[0], _shape_host(key).host.MAX_ROWS
    if not 1 <= rows <= max_rows:
        return f"{_OP}: {rows} rows (the kernel takes 1..{max_rows})"
    if scores.dtype != torch.float32 or scores.stride(1) != 1:
        return f"{_OP}: scores must be fp32 with unit column stride"
    if local_idx.dtype != torch.int32 or local_idx.shape != (rows, topk) or local_idx.stride(1) != 1:
        return f"{_OP}: local_idx must be int32 [rows, {topk}] with unit column stride"
    if (block_table.ndim != 2 or block_table.dtype != torch.int32 or block_table.shape[0] != rows
            or block_table.stride(1) != 1):
        return f"{_OP}: block_table must be int32 [rows, blocks] with unit column stride"
    if (out_slots.dtype != torch.int32 or out_slots.numel() < rows * topk or out_indptr.dtype != torch.int32
            or out_indptr.numel() < rows + 1):
        return f"{_OP}: out_slots must be int32 of >= rows * {topk}, out_indptr int32 of >= rows + 1"
    if any(t.device != comm.device for t in (scores, local_idx, block_table, out_slots, out_indptr)):
        return f"{_OP}: operands must be on the communicator's device {comm.device}"
    return None
