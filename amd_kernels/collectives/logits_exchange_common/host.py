# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""lm_head shard GEMM + exchange of its logits rows: routes by (vocab shard N, hidden K, world) to the shape package's
host (``RowAllToAll``, ``BUFFERS``, ``RECV_ROWS``). K is in the key because the communicator's ``gemm_exchange`` runs
the GEMM of one weight shape. The ops are the communicator's methods (``gemm_exchange``, ``recv``, ``__call__``,
``warmup``, ``close``); the reason of a ``gemm_exchange`` call routes by the communicator."""
import functools
import importlib

import torch
import torch.distributed as dist

from . import available

OP = "logits_gemm_exchange"
_DIMS = ("n_shard", "k", "world")
_SHAPE_HOSTS = {
    (20480, 7168, 8): "amd_kernels.collectives.all_to_all_bf16_n20480_ranks8.host",
}


@functools.cache
def _shape_host(key: tuple):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


@functools.cache
def _comm_key(cls: type):
    return next((key for key in _SHAPE_HOSTS if getattr(_shape_host(key), cls.__name__, None) is cls), None)


def _no_kernel(op: str, key: tuple) -> str:
    dims = ", ".join(f"{d}={v}" for d, v in zip(_DIMS, key))
    return f"{op}: no kernel for {dims} (covered: {sorted(_SHAPE_HOSTS)})"


def logits_gemm_exchange_shapes() -> frozenset[tuple[int, int, int]]:
    """The covered (vocab shard N, K, world) of the lm_head shard [N, K]."""
    return frozenset(_SHAPE_HOSTS)


def logits_exchange_comm(rank: int, group, device, *, n_shard: int, k: int, max_rows: int):
    """The routed package's communicator (collective on the group) with its first-use buffers allocated, so that no
    call allocates (graph capture)."""
    key = (n_shard, k, dist.get_world_size(group))
    host = _shape_host(key)
    if host is None:
        raise ValueError(_no_kernel("logits_exchange_comm", key))
    comm = host.RowAllToAll(rank, group, device, max_rows=max_rows)
    comm._zero_counts()
    comm._local = torch.empty(max_rows, n_shard, dtype=torch.bfloat16, device=comm.counter.device)
    return comm


def logits_exchange_comm_unsupported_reason(rank: int, group, device, *, n_shard: int, k: int,
                                            max_rows: int) -> str | None:
    op = "logits_exchange_comm"
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = (n_shard, k, dist.get_world_size(group))
    host = _shape_host(key)
    if host is None:
        return _no_kernel(op, key)
    if not 1 <= max_rows <= host.RECV_ROWS:
        return f"{op}: max_rows {max_rows} (the kernel takes 1..{host.RECV_ROWS})"
    if dist.get_rank(group) != rank:
        return f"{op}: rank {rank} is not this process's rank {dist.get_rank(group)} in the group"
    return None


def logits_gemm_exchange_unsupported_reason(comm, x: torch.Tensor, w: torch.Tensor, row_indices: torch.Tensor,
                                            counts: torch.Tensor | None = None, buf: int = 0,
                                            serial_above: int | None = None) -> str | None:
    """None if ``comm.gemm_exchange`` with these arguments is served, else why not (host checks only)."""
    if not available():
        return f"{OP}: needs gfx950 with FlyDSL"
    key = _comm_key(type(comm))
    if key is None:
        return f"{OP}: no kernel for communicator {type(comm).__qualname__} (covered: {sorted(_SHAPE_HOSTS)})"
    host = _shape_host(key)
    n_shard, k, world = key
    if tuple(w.shape) != (n_shard, k):
        return f"{OP}: communicator built for {key}, call is {(*w.shape, world)}"
    if comm.closed:
        return f"{OP}: communicator is closed"
    dev = comm.counter.device
    if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.device != dev:
        return f"{OP}: w must be contiguous bf16 on the communicator's device"
    if (x.ndim != 2 or x.shape[1] != k or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != dev
            or x.numel() * x.element_size() > 0xFFFFFFFF):
        return f"{OP}: x must be contiguous bf16 [rows, {k}] (< 4 GiB) on the communicator's device"
    if row_indices.ndim != 1 or row_indices.dtype != torch.int64 or not row_indices.is_contiguous() \
            or row_indices.device != dev:
        return f"{OP}: row_indices must be a contiguous int64 vector on the communicator's device"
    rows = row_indices.numel()
    if not 1 <= rows <= comm.max_rows:
        return f"{OP}: {rows} rows (the communicator takes 1..{comm.max_rows})"
    if counts is not None and (tuple(counts.shape) != (world,) or counts.dtype != torch.int32
                               or not counts.is_contiguous() or counts.device != dev):
        return f"{OP}: counts must be contiguous int32 [{world}] on the communicator's device"
    if not 0 <= buf < host.BUFFERS:
        return f"{OP}: buffer {buf} (the communicator has {host.BUFFERS})"
    return None
