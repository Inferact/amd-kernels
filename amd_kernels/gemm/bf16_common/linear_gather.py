# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""N-sharded linear + column gather + RMSNorm over the ranks of a group: routes by (N shard, K, world) to the shape
package's host (``ProjComm``, ``context_proj_gather_norm``, ``pack_weight``, ``pack_aux``, ``packed_tiles``,
``packed_rows_buffer``, ``MAX_ROWS``, ``TIERS``). The factory reads the world size from the group, the op routes by
its communicator, the load-time helpers by the weight shard. ``w_packed`` is ``linear_gather_pack_weight``'s output:
it keeps the raw shard's shape, so no check can tell the two apart."""
import functools
import importlib

import torch
import torch.distributed as dist

from . import available

OP = "bf16_linear_gather"
_DIMS = ("n_shard", "k", "world")
_SHAPE_HOSTS = {
    (896, 35840, 8): "amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8.host",
}


@functools.cache
def _shape_host(key: tuple):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


@functools.cache
def _comm_key(cls: type):
    return next((key for key in _SHAPE_HOSTS if getattr(_shape_host(key), cls.__name__, None) is cls), None)


@functools.cache
def _weight_key(shape: tuple):
    return next((key for key in _SHAPE_HOSTS if key[:2] == shape), None)


@functools.cache
def _layers(host) -> int:
    return importlib.import_module(host.__package__ + ".pack_rows").LAYERS


def _max_rows(host) -> int:
    return min(host.MAX_ROWS, host.TIERS[-1][0])


def _no_kernel(op: str, key: tuple) -> str:
    dims = ", ".join(f"{d}={v}" for d, v in zip(_DIMS, key))
    return f"{op}: no kernel for {dims} (covered: {sorted(_SHAPE_HOSTS)})"


def _weight_host(op: str, w: torch.Tensor):
    key = _weight_key(tuple(w.shape))
    if key is None:
        raise ValueError(_no_kernel(op, tuple(w.shape)))
    return _shape_host(key)


def bf16_linear_gather_shapes() -> frozenset[tuple[int, int, int]]:
    """The covered (N shard, K, world)."""
    return frozenset(_SHAPE_HOSTS)


def linear_gather_comm(rank: int, group, device, *, n_shard: int, k: int):
    """The routed package's communicator (collective on the group; its ``close()`` is collective too)."""
    key = (n_shard, k, dist.get_world_size(group))
    host = _shape_host(key)
    if host is None:
        raise ValueError(_no_kernel("linear_gather_comm", key))
    return host.ProjComm(rank, group, device)


def linear_gather_comm_unsupported_reason(rank: int, group, device, *, n_shard: int, k: int) -> str | None:
    op = "linear_gather_comm"
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = (n_shard, k, dist.get_world_size(group))
    if _shape_host(key) is None:
        return _no_kernel(op, key)
    if dist.get_rank(group) != rank:
        return f"{op}: rank {rank} is not this process's rank {dist.get_rank(group)} in the group"
    return None


def linear_gather_pack_weight(w: torch.Tensor) -> torch.Tensor:
    """This rank's [N shard, K] bf16 rows of the weight -> the packed layout the op reads (same shape)."""
    return _weight_host("linear_gather_pack_weight", w).pack_weight(w)


def linear_gather_packed_tiles(rows: int, *, w_packed: torch.Tensor) -> int:
    """16-row tiles of the packed activations the op reads for ``rows`` rows."""
    return _weight_host("linear_gather_packed_tiles", w_packed).packed_tiles(rows)


def linear_gather_rows_buffer(rows: int, device, *, w_packed: torch.Tensor) -> torch.Tensor:
    """An uninitialized packed-activation buffer of ``rows`` rows."""
    return _weight_host("linear_gather_rows_buffer", w_packed).packed_rows_buffer(rows, device)


def linear_gather_pack_aux(aux: list[torch.Tensor], out: torch.Tensor, rows: int, *,
                           w_packed: torch.Tensor) -> torch.Tensor:
    """The aux hidden states (their concatenation is the op's K) -> the packed tiles of ``rows`` rows in ``out``."""
    return _weight_host("linear_gather_pack_aux", w_packed).pack_aux(aux, out, rows)


def linear_gather_pack_aux_unsupported_reason(aux: list[torch.Tensor], out: torch.Tensor, rows: int, *,
                                              w_packed: torch.Tensor) -> str | None:
    op = "linear_gather_pack_aux"
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = _weight_key(tuple(w_packed.shape))
    if key is None:
        return _no_kernel(op, tuple(w_packed.shape))
    host = _shape_host(key)
    k, layers, hi = key[1], _layers(host), _max_rows(host)
    if not 1 <= rows <= hi:
        return f"{op}: {rows} rows (the kernel takes 1..{hi})"
    if len(aux) != layers or any(h.ndim != 2 or h.shape[1] != k // layers or h.shape[0] < rows
                                 or h.dtype != torch.bfloat16 or h.stride(1) != 1 or h.stride(0) != aux[0].stride(0)
                                 or h.device != w_packed.device for h in aux):
        return f"{op}: aux must be {layers} bf16 [>= {rows}, {k // layers}] with unit column stride and one row stride"
    need = host.packed_tiles(rows) * (k // 32) * 512
    if out.dtype != torch.bfloat16 or not out.is_contiguous() or out.numel() != need or out.device != w_packed.device:
        return f"{op}: out must be contiguous bf16 of {need} elements (linear_gather_packed_tiles)"
    return None


def bf16_linear_gather(comm, x_packed: torch.Tensor, w_packed: torch.Tensor, norm_weight: torch.Tensor, rows: int,
                       eps: float = 1e-5, out: torch.Tensor | None = None) -> torch.Tensor:
    """``RMSNorm(x @ W^T)`` of the full [rows, N shard * world] projection on every rank (the routed package's
    ``context_proj_gather_norm``: every rank calls with the same ``rows``, in the same order)."""
    key = _comm_key(type(comm))
    if key is None:
        raise ValueError(f"{OP}: no kernel for communicator {type(comm).__qualname__}")
    return _shape_host(key).context_proj_gather_norm(comm, x_packed, w_packed, norm_weight, rows, eps, out)


def bf16_linear_gather_unsupported_reason(comm, x_packed: torch.Tensor, w_packed: torch.Tensor,
                                          norm_weight: torch.Tensor, rows: int, eps: float = 1e-5,
                                          out: torch.Tensor | None = None) -> str | None:
    """None if ``bf16_linear_gather`` serves this call, else why not (host checks only)."""
    if not available():
        return f"{OP}: needs gfx950 with FlyDSL"
    key = _comm_key(type(comm))
    if key is None:
        return f"{OP}: no kernel for communicator {type(comm).__qualname__} (covered: {sorted(_SHAPE_HOSTS)})"
    host = _shape_host(key)
    n_shard, k, world = key
    if tuple(w_packed.shape) != (n_shard, k):
        return f"{OP}: communicator built for {key}, call is {(*w_packed.shape, world)}"
    if not comm.bufs:
        return f"{OP}: communicator is closed"
    dev = comm.device
    if w_packed.dtype != torch.bfloat16 or not w_packed.is_contiguous() or w_packed.device != dev:
        return f"{OP}: w_packed must be contiguous bf16 on the communicator's device"
    hi = _max_rows(host)
    if not 1 <= rows <= hi:
        return f"{OP}: {rows} rows (the kernel takes 1..{hi})"
    tiles = host.packed_tiles(rows)
    if (x_packed.dtype != torch.bfloat16 or not x_packed.is_contiguous() or x_packed.ndim != 4
            or tuple(x_packed.shape[1:]) != (k // 32, 64, 8) or x_packed.shape[0] < tiles or x_packed.device != dev):
        return f"{OP}: x_packed must be contiguous bf16 [>= {tiles}, {k // 32}, 64, 8] (linear_gather_pack_aux)"
    n_full = n_shard * world
    if (norm_weight.dtype != torch.bfloat16 or norm_weight.numel() != n_full or not norm_weight.is_contiguous()
            or norm_weight.device != dev):
        return f"{OP}: norm_weight must be contiguous bf16 [{n_full}] on the communicator's device"
    if out is not None and (tuple(out.shape) != (rows, n_full) or out.dtype != torch.bfloat16
                            or not out.is_contiguous() or out.device != dev):
        return f"{OP}: out must be contiguous bf16 [{rows}, {n_full}] on the communicator's device"
    return None
