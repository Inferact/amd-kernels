# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry of the all-reduce + AttnRes block mix + RMSNorm, routed by (hidden, world size). The communicator is built per
key; the op is its method ``comm.allreduce_attnres_rmsnorm(...)``, which
``allreduce_attnres_rmsnorm_unsupported_reason`` checks."""
import functools
import importlib

import torch
import torch.distributed as dist

_SHAPE_HOSTS = {(7168, 8): "amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8.host"}  # hidden, world
_NAMES = ("hidden", "world")
_OP = "allreduce_attnres_rmsnorm"


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
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


def _dims(key) -> str:
    return ", ".join(f"{n}={v}" for n, v in zip(_NAMES, key))


def _no_kernel(op, key) -> str:
    return f"{op}: no kernel for {_dims(key)} (covered: {', '.join(str(k) for k in sorted(_SHAPE_HOSTS))})"


def attnres_comm(rank, group, device, *, hidden):
    """The communicator of this rank (collective over ``group``; ``.close()`` after the work completes)."""
    key = (hidden, dist.get_world_size(group))
    host = _shape_host(key)
    if host is None:
        raise ValueError(_no_kernel("attnres_comm", key))
    return host.AttnResComm(rank, group, device)


def attnres_comm_unsupported_reason(rank, group, device, *, hidden) -> str | None:
    if not available():
        return "attnres_comm: needs gfx950 with FlyDSL"
    key = (hidden, dist.get_world_size(group))
    return None if key in _SHAPE_HOSTS else _no_kernel("attnres_comm", key)


def allreduce_attnres_rmsnorm_shapes() -> frozenset:
    return frozenset(_SHAPE_HOSTS)


def allreduce_attnres_rmsnorm_unsupported_reason(comm, partial, prefix, bank, norm_weight, qk_weight, num_blocks,
                                                 block_write_idx=-1, out_norm_weight=None, eps=1e-6,
                                                 out_norm_eps=1e-6, reduce=True, out_packed=False) -> str | None:
    """None when ``comm.allreduce_attnres_rmsnorm`` serves these arguments, else why not."""
    if not available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = next((k for k, name in _SHAPE_HOSTS.items() if type(comm).__module__ == name), None)
    if key is None:
        return (f"{_OP}: no kernel for a {type(comm).__qualname__} "
                f"(covered: {', '.join(str(k) for k in sorted(_SHAPE_HOSTS))})")
    host = _shape_host(key)
    if not comm.bufs:
        return f"{_OP}: communicator is closed"
    h, dev = key[0], comm.device
    if partial.ndim != 2 or partial.shape[1] != h:
        return f"{_OP}: communicator built for {_dims(key)}, call is hidden={partial.shape[-1]}"
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
    for w in (norm_weight, qk_weight, out_norm_weight):
        if w is not None and (w.numel() != h or w.dtype != partial.dtype or w.stride(-1) != 1 or w.device != dev):
            return f"{_OP}: weights must be contiguous bf16 [{h}]"
    return None
