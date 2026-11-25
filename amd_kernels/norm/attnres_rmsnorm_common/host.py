# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry of the all-reduce + AttnRes block mix + RMSNorm, routed by (hidden, world size). The communicator is built per
key; the op is its method ``comm.allreduce_attnres_rmsnorm(...)``, which ``allreduce_attnres_rmsnorm_blocker``
checks."""
import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, available, prepare, renamed

_NAMES = ("hidden", "world")
_ROUTES = Routes("allreduce_attnres_rmsnorm",
                 {(7168, 8): "amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8.host"}, _NAMES)
_COMM = Routes("attnres_comm", _ROUTES.hosts, _NAMES)
_OP = _ROUTES.op


def attnres_comm(rank, group, device, *, hidden):
    """The communicator of this rank (collective over ``group``; ``.close()`` after the work completes)."""
    return _COMM.package((hidden, dist.get_world_size(group))).AttnResComm(rank, group, device)


def attnres_comm_blocker(rank, group, device, *, hidden) -> str | None:
    if not routes.available():
        return "attnres_comm: needs gfx950 with FlyDSL"
    return _COMM.blocker((hidden, dist.get_world_size(group)))


def prepare_attnres_comm(rank, group, device, *, hidden):
    """The communicator of this rank, or the reason none serves this key."""
    return prepare(attnres_comm_blocker(rank, group, device, hidden=hidden),
                   lambda: attnres_comm(rank, group, device, hidden=hidden))


def allreduce_attnres_rmsnorm_shapes() -> frozenset:
    return _ROUTES.shapes()


def allreduce_attnres_rmsnorm_blocker(comm, partial, prefix, bank, norm_weight, qk_weight, num_blocks,
                                      block_write_idx=-1, out_norm_weight=None, eps=1e-6, out_norm_eps=1e-6,
                                      reduce=True, out_packed=False) -> str | None:
    """None when ``comm.allreduce_attnres_rmsnorm`` serves these arguments, else why not."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _ROUTES.owner(type(comm))
    if key is None:
        return f"{_OP}: no kernel for a {type(comm).__qualname__} (covered: {sorted(_ROUTES.hosts)})"
    host = _ROUTES.package(key)
    if not comm.bufs:
        return f"{_OP}: communicator is closed"
    h, dev = key[0], comm.device
    if partial.ndim != 2 or partial.shape[1] != h:
        return f"{_OP}: communicator built for {_ROUTES.describe(key)}, call is hidden={partial.shape[-1]}"
    m = partial.shape[0]
    if not 1 <= m <= host.MAX_ROWS:
        return f"{_OP}: {m} rows (the kernel takes 1..{host.MAX_ROWS})"
    if partial.dtype != torch.bfloat16 or not partial.is_contiguous() or partial.device != dev:
        return f"{_OP}: partial must be contiguous bf16 on the communicator device"
    if prefix is not None and (prefix.shape != partial.shape or prefix.dtype != partial.dtype or prefix.stride(1) != 1
                               or prefix.stride(0) % 8 or prefix.device != dev):
        return f"{_OP}: prefix must be bf16 [{m}, {h}] with a row stride divisible by eight"
    if bank is not None:
        if (bank.ndim != 3 or bank.shape[0] != m or bank.shape[1] > host.BANK_ROWS or bank.shape[2] != h
                or bank.dtype != partial.dtype or bank.stride(2) != 1 or bank.stride(1) % 8 or bank.stride(0) % 8
                or bank.device != dev):
            return f"{_OP}: bank must be bf16 [{m}, <={host.BANK_ROWS}, {h}] with strides divisible by eight"
        if not 0 <= num_blocks <= bank.shape[1] or not (block_write_idx < 0 or num_blocks <= block_write_idx
                                                        < bank.shape[1]):
            return f"{_OP}: num_blocks <= block_write_idx < bank rows, or block_write_idx < 0"
    if bank is not None and (norm_weight is None or qk_weight is None):
        return f"{_OP}: a bank needs norm_weight and qk_weight"
    for w in (norm_weight, qk_weight, out_norm_weight):
        if w is not None and (w.numel() != h or w.dtype != partial.dtype or w.stride(-1) != 1 or w.device != dev):
            return f"{_OP}: weights must be contiguous bf16 [{h}]"
    return None


attnres_comm_unsupported_reason = renamed(attnres_comm_blocker, "attnres_comm_unsupported_reason")
allreduce_attnres_rmsnorm_unsupported_reason = renamed(allreduce_attnres_rmsnorm_blocker,
                                                       "allreduce_attnres_rmsnorm_unsupported_reason")
