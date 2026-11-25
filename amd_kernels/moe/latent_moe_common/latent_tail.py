# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry of the latent tail (all-reduce of the routed latent partial, RMSNorm, this rank's up-projection rows added to
the shared output), routed by (up-projection rows, latent width, world size). The communicator is built per key; its
calls route by the communicator."""
import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare, renamed

from . import dispatch

_NAMES = ("up_rows", "latent", "world")  # up_weight [up_rows, latent], world
_ROUTES = Routes("allreduce_latent_rmsnorm_up_proj",
                 {(896, 3584, 8): "amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.host"}, _NAMES)
_SLOTS = Routes("allreduce_slots_rmsnorm_up_proj", _ROUTES.hosts, _NAMES)
_COMM = Routes("latent_tail_comm", _ROUTES.hosts, _NAMES)


def _comm_host(op_routes, cls):
    key = op_routes.owner(cls)
    if key is None:
        raise ValueError(dispatch.not_owned(op_routes, cls))
    return op_routes.package(key)


def latent_tail_comm(rank, group, device, *, up_rows, latent, fused=True):
    """The communicator of this rank (collective over ``group``; ``.close()`` after the work completes)."""
    host = _COMM.package((up_rows, latent, dist.get_world_size(group)))
    return host.LatentTailComm(rank, group, device, fused=fused)


def latent_tail_comm_blocker(rank, group, device, *, up_rows, latent, fused=True) -> str | None:
    if not routes.available():
        return f"{_COMM.op}: needs gfx950 with FlyDSL"
    return _COMM.blocker((up_rows, latent, dist.get_world_size(group)))


def prepare_latent_tail_comm(rank, group, device, *, up_rows, latent, fused=True):
    """The communicator of this rank, or the reason none serves this key."""
    return prepare(latent_tail_comm_blocker(rank, group, device, up_rows=up_rows, latent=latent, fused=fused),
                   lambda: latent_tail_comm(rank, group, device, up_rows=up_rows, latent=latent, fused=fused))


def allreduce_latent_rmsnorm_up_proj(comm, partial, norm_weight, up_weight, shared, out=None, eps=1e-6):
    """``out = shared + rmsnorm(allreduce(partial)) @ up_weight^T`` (``out`` defaults to ``shared``, in place)."""
    return _comm_host(_ROUTES, type(comm)).allreduce_latent_rmsnorm_up_proj(
        comm, partial, norm_weight, up_weight, shared, out, eps)


def allreduce_slots_rmsnorm_up_proj(comm, slots, reverse_sorted, norm_weight, up_weight, shared, out=None, eps=1e-6):
    """The same with the experts' slot sum folded in: ``slots`` the slot buffer (``mxfp4_moe_experts(reduce=False)``),
    ``reverse_sorted`` the sort's row of each (token, slot)."""
    return _comm_host(_SLOTS, type(comm)).allreduce_slots_rmsnorm_up_proj(
        comm, slots, reverse_sorted, norm_weight, up_weight, shared, out, eps)


def allreduce_latent_rmsnorm_up_proj_shapes() -> frozenset:
    return _ROUTES.shapes()


def allreduce_slots_rmsnorm_up_proj_shapes() -> frozenset:
    return _SLOTS.shapes()


def _blocker(op_routes, comm, x, reverse_sorted, norm_weight, up_weight, shared, out):
    op = op_routes.op
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = op_routes.owner(type(comm))
    if key is None:
        return dispatch.not_owned(op_routes, type(comm))
    host = op_routes.package(key)
    if not comm.bufs:
        return f"{op}: communicator is closed"
    n, k = key[0], key[1]
    call = tuple(up_weight.shape)
    if call != (n, k):
        return dispatch.built_for(op_routes, key, call)
    slots = reverse_sorted is not None
    dev = comm.device
    if slots:
        m, hi = reverse_sorted.numel() // host.T.SLOTS, host.T.MAX_SLOT_ROWS
    else:
        m, hi = (x.shape[0] if x.ndim == 2 else 0), host.MAX_ROWS
    if not 1 <= m <= hi:
        return f"{op}: {m} rows (the kernel takes 1..{hi})"
    if x.ndim != 2 or x.shape[1] != k:
        return dispatch.built_for(op_routes, key, (n, x.shape[-1]))
    if slots and (reverse_sorted.dtype != torch.int32 or reverse_sorted.device != dev
                  or not reverse_sorted.is_contiguous() or not x.is_contiguous()
                  or reverse_sorted.numel() != m * host.T.SLOTS):
        return f"{op}: slots must be contiguous, reverse_sorted contiguous int32 [rows * {host.T.SLOTS}]"
    out = shared if out is None else out
    if any(t.dtype != torch.bfloat16 or t.device != dev or t.stride(-1) != 1
           for t in (x, norm_weight, up_weight, shared, out)):
        return f"{op}: operands must be bf16 on the communicator device with unit inner stride"
    if (not slots and x.stride(0) % 8) or shared.stride(0) % 4 or out.stride(0) % 4:
        return f"{op}: row strides must keep 16 B (latent) and 8 B (shard) alignment"
    if tuple(norm_weight.shape) != (k,) or not up_weight.is_contiguous():
        return f"{op}: norm weight [{k}] and contiguous up weight [{n}, {k}] required"
    if tuple(shared.shape) != (m, n) or tuple(out.shape) != (m, n):
        return f"{op}: shared and out must be [{m}, {n}]"
    return None


def allreduce_latent_rmsnorm_up_proj_blocker(comm, partial, norm_weight, up_weight, shared, out=None,
                                             eps=1e-6) -> str | None:
    return _blocker(_ROUTES, comm, partial, None, norm_weight, up_weight, shared, out)


def allreduce_slots_rmsnorm_up_proj_blocker(comm, slots, reverse_sorted, norm_weight, up_weight, shared, out=None,
                                            eps=1e-6) -> str | None:
    if reverse_sorted is None:
        return f"{_SLOTS.op}: needs reverse_sorted"
    return _blocker(_SLOTS, comm, slots, reverse_sorted, norm_weight, up_weight, shared, out)


latent_tail_comm_unsupported_reason = renamed(latent_tail_comm_blocker, "latent_tail_comm_unsupported_reason")
allreduce_latent_rmsnorm_up_proj_unsupported_reason = renamed(allreduce_latent_rmsnorm_up_proj_blocker,
                                                              "allreduce_latent_rmsnorm_up_proj_unsupported_reason")
allreduce_slots_rmsnorm_up_proj_unsupported_reason = renamed(allreduce_slots_rmsnorm_up_proj_blocker,
                                                             "allreduce_slots_rmsnorm_up_proj_unsupported_reason")
