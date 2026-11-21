# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry of the latent tail (all-reduce of the routed latent partial, RMSNorm, this rank's up-projection rows added to
the shared output), routed by (up-projection rows, latent width, world size). The communicator is built per key; its
calls route by the communicator."""
import functools
import importlib

import torch
import torch.distributed as dist

from . import dispatch

# up_weight [up_rows, latent], world
_SHAPE_HOSTS = {(896, 3584, 8): "amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.host"}
_NAMES = ("up_rows", "latent", "world")


@functools.cache
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


@functools.cache
def _comm_host(cls, op):
    key = dispatch.key_of_type(_SHAPE_HOSTS, cls)
    if key is None:
        raise ValueError(dispatch.no_kernel_for_type(op, cls, _SHAPE_HOSTS))
    return _shape_host(key)


def latent_tail_comm(rank, group, device, *, up_rows, latent, fused=True):
    """The communicator of this rank (collective over ``group``; ``.close()`` after the work completes)."""
    key = (up_rows, latent, dist.get_world_size(group))
    host = _shape_host(key)
    if host is None:
        raise ValueError(dispatch.no_kernel("latent_tail_comm", _NAMES, key, _SHAPE_HOSTS))
    return host.LatentTailComm(rank, group, device, fused=fused)


def latent_tail_comm_unsupported_reason(rank, group, device, *, up_rows, latent, fused=True) -> str | None:
    op = "latent_tail_comm"
    if not dispatch.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = (up_rows, latent, dist.get_world_size(group))
    if key not in _SHAPE_HOSTS:
        return dispatch.no_kernel(op, _NAMES, key, _SHAPE_HOSTS)
    return None


def allreduce_latent_rmsnorm_up_proj(comm, partial, norm_weight, up_weight, shared, out=None, eps=1e-6):
    """``out = shared + rmsnorm(allreduce(partial)) @ up_weight^T`` (``out`` defaults to ``shared``, in place)."""
    return _comm_host(type(comm), "allreduce_latent_rmsnorm_up_proj").allreduce_latent_rmsnorm_up_proj(
        comm, partial, norm_weight, up_weight, shared, out, eps)


def allreduce_slots_rmsnorm_up_proj(comm, slots, reverse_sorted, norm_weight, up_weight, shared, out=None, eps=1e-6):
    """The same with the experts' slot sum folded in: ``slots`` the slot buffer (``mxfp4_moe_experts(reduce=False)``),
    ``reverse_sorted`` the sort's row of each (token, slot)."""
    return _comm_host(type(comm), "allreduce_slots_rmsnorm_up_proj").allreduce_slots_rmsnorm_up_proj(
        comm, slots, reverse_sorted, norm_weight, up_weight, shared, out, eps)


def allreduce_latent_rmsnorm_up_proj_shapes() -> frozenset:
    return frozenset(_SHAPE_HOSTS)


def allreduce_slots_rmsnorm_up_proj_shapes() -> frozenset:
    return frozenset(_SHAPE_HOSTS)


def _reason(op, comm, x, reverse_sorted, norm_weight, up_weight, shared, out):
    if not dispatch.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = dispatch.key_of_type(_SHAPE_HOSTS, type(comm))
    if key is None:
        return dispatch.no_kernel_for_type(op, type(comm), _SHAPE_HOSTS)
    host = _shape_host(key)
    if not comm.bufs:
        return f"{op}: communicator is closed"
    n, k = key[0], key[1]
    call = tuple(up_weight.shape)
    if call != (n, k):
        return dispatch.built_for(op, _NAMES, key, call)
    slots = reverse_sorted is not None
    dev = comm.device
    if slots:
        m, hi = reverse_sorted.numel() // host.T.SLOTS, host.T.MAX_SLOT_ROWS
    else:
        m, hi = (x.shape[0] if x.ndim == 2 else 0), host.MAX_ROWS
    if not 1 <= m <= hi:
        return f"{op}: {m} rows (the kernel takes 1..{hi})"
    if x.ndim != 2 or x.shape[1] != k:
        return dispatch.built_for(op, _NAMES, key, (n, x.shape[-1]))
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


def allreduce_latent_rmsnorm_up_proj_unsupported_reason(comm, partial, norm_weight, up_weight, shared, out=None,
                                                        eps=1e-6) -> str | None:
    return _reason("allreduce_latent_rmsnorm_up_proj", comm, partial, None, norm_weight, up_weight, shared, out)


def allreduce_slots_rmsnorm_up_proj_unsupported_reason(comm, slots, reverse_sorted, norm_weight, up_weight, shared,
                                                       out=None, eps=1e-6) -> str | None:
    if reverse_sorted is None:
        return "allreduce_slots_rmsnorm_up_proj: needs reverse_sorted"
    return _reason("allreduce_slots_rmsnorm_up_proj", comm, slots, reverse_sorted, norm_weight, up_weight, shared, out)
