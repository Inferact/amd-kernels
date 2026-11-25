# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The DCP score all-gather + global top-k merge + CSR slots, routed by (top-k, world size).

The factory builds the shape package's communicator; the caller calls it (``comm(scores, local_idx, ...)``), whose
reason routes by the communicator's type.
"""

import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare, renamed

_OP = "topk_gather_merge"
_ROUTES = Routes(f"{_OP}_comm", {(2048, 8): "amd_kernels.collectives.dcp8_topk_gather_merge_top2048"},
                 ("topk", "world"))


def topk_gather_merge_comm(rank, group, device, *, topk):
    """The collective communicator for ``topk`` over ``group`` (every rank calls it); the caller calls ``.warmup()``
    before capture, the communicator per call and ``.close()`` after its graphs are retired."""
    return _ROUTES.package((topk, dist.get_world_size(group))).TopkGatherMerge(rank, group, device)


def topk_gather_merge_comm_blocker(rank, group, device, *, topk) -> str | None:
    """None when ``topk_gather_merge_comm`` builds a communicator for these arguments, else why not."""
    if not routes.available():
        return f"{_OP}_comm: needs gfx950 with FlyDSL"
    if not dist.is_initialized():
        return f"{_OP}_comm: torch.distributed is not initialized"
    reason = _ROUTES.blocker((topk, dist.get_world_size(group)))
    if reason is not None:
        return reason
    if rank != dist.get_rank(group):
        return f"{_OP}_comm: rank {rank} is not this process's rank {dist.get_rank(group)} in the group"
    return None


def prepare_topk_gather_merge_comm(rank, group, device, *, topk):
    """``topk_gather_merge_comm(...)``, or the reason it cannot be built."""
    return prepare(topk_gather_merge_comm_blocker(rank, group, device, topk=topk),
                   lambda: topk_gather_merge_comm(rank, group, device, topk=topk))


def topk_gather_merge_shapes() -> frozenset:
    """The covered (top-k, world size)."""
    return _ROUTES.shapes()


def topk_gather_merge_blocker(comm, scores, local_idx, block_table, out_slots, out_indptr, page_size,
                              block_stride_rows) -> str | None:
    """None when ``comm(scores, local_idx, ...)`` serves this call, else why not (host checks only)."""
    key = _ROUTES.owner(type(comm))
    if key is None:
        return f"{_OP}: {type(comm).__name__} is not a communicator of this entry"
    if comm.closed:
        return f"{_OP}: communicator closed"
    topk = scores.shape[1] if scores.ndim == 2 else None
    if topk != key[0]:
        return f"{_OP}: communicator built for topk={key[0]}, world={key[1]}, call is topk={topk}"
    rows, max_rows = scores.shape[0], _ROUTES.host(key).host.MAX_ROWS
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


topk_gather_merge_comm_unsupported_reason = renamed(topk_gather_merge_comm_blocker,
                                                    "topk_gather_merge_comm_unsupported_reason")
topk_gather_merge_unsupported_reason = renamed(topk_gather_merge_blocker, "topk_gather_merge_unsupported_reason")
