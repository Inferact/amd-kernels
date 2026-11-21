# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DCP query gather (q_b + W_UK + head all-gather, 8 ranks): key (q_lora, q_b rows, heads, world). The communicator
factory takes the dims as keywords and the world from the group; a call routes by its communicator."""

import functools
import importlib

import torch
import torch.distributed as dist

from .routing import available, built_for, no_kernel, owner_key

_SHAPE_HOSTS = {
    (1536, 2304, 96, 8): "amd_kernels.attention.dcp8_q_bproj_wuk_gather_h96_kv512",
}
_KEY = ("q_lora", "q_b_rows", "heads", "world")
_OP = "q_bproj_wuk_gather"
_COMM = "q_gather_comm"


@functools.cache
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


@functools.cache
def _comm_key(cls):
    return owner_key(_SHAPE_HOSTS, cls)


def q_gather_comm(rank, group, device, mode, *, q_lora, q_b_rows, heads):
    """The package's communicator for this shape on the ranks of ``group`` (a collective); ``.close()`` releases it."""
    key = (q_lora, q_b_rows, heads, dist.get_world_size(group))
    pkg = _shape_host(key)
    if pkg is None:
        raise ValueError(no_kernel(_COMM, _KEY, key, _SHAPE_HOSTS))
    return pkg.QueryGatherComm(rank, group, device, mode)


def q_gather_comm_unsupported_reason(rank, group, device, mode, *, q_lora, q_b_rows, heads) -> str | None:
    if not dist.is_initialized():
        return f"{_COMM}: needs an initialized torch.distributed process group"
    key = (q_lora, q_b_rows, heads, dist.get_world_size(group))
    if key not in _SHAPE_HOSTS:
        return no_kernel(_COMM, _KEY, key, _SHAPE_HOSTS)
    if not available():
        return f"{_COMM}: needs gfx950 with FlyDSL"
    modes = sorted(_shape_host(key).comm.GATHER_BYTES)
    if mode not in modes:
        return f"{_COMM}: mode {mode!r} (the package has {modes})"
    return None


def _weight_key(match, what):
    for key in _SHAPE_HOSTS:
        if match(key):
            return key
    raise ValueError(f"{_OP}: no kernel for {what} (covered: {', '.join(str(k) for k in sorted(_SHAPE_HOSTS))})")


def pack_q_b(w):
    """The load-time q_b weight [q_b rows, q_lora] in the kernel's tile order (same shape)."""
    key = _weight_key(lambda k: k[:2] == (w.shape[1], w.shape[0]), f"q_b weight {tuple(w.shape)}")
    return _shape_host(key).pack_q_b(w)


def pack_w_uk(w_uk_t):
    """The load-time W_UK_T of all heads [heads, 128, kv_lora] in the kernel's tile order (dim 0 stays the heads)."""
    def match(k):
        wuk = _shape_host(k).kernel_wuk
        return k[2] == w_uk_t.shape[0] and tuple(w_uk_t.shape[1:]) == (wuk.P, wuk.L)

    return _shape_host(_weight_key(match, f"W_UK_T {tuple(w_uk_t.shape)}")).pack_w_uk(w_uk_t)


def _routed(comm):
    key = _comm_key(type(comm))
    if key is None:
        raise ValueError(_no_comm(comm))
    return _shape_host(key)


def _no_comm(comm):
    covered = ", ".join(str(k) for k in sorted(_SHAPE_HOSTS))
    return f"{_OP}: no kernel for communicator {type(comm).__qualname__} (covered: {covered})"


def q_bproj_wuk_gather(comm, q_a, w_qb, w_uk96, out=None, out_q8=None):
    """The query of all heads on every rank (``out`` bf16 [rows, heads, 576] and / or ``out_q8``, the sparse MLA
    decode's pre-quantized record); returns ``out``, ``out_q8`` or both."""
    return _routed(comm).q_b_allgather_wuk_all_heads(comm, q_a, w_qb, w_uk96, out=out, out_q8=out_q8)


def q_bproj_wuk_gather_shapes() -> frozenset:
    return frozenset(_SHAPE_HOSTS)


def q_bproj_wuk_gather_unsupported_reason(comm, q_a, w_qb, w_uk96, out=None, out_q8=None) -> str | None:
    op = _OP
    key = _comm_key(type(comm))
    if key is None:
        return _no_comm(comm)
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    pkg = _shape_host(key)
    if comm.mode != "q":
        return f"{op}: communicator mode {comm.mode!r} (the op takes 'q')"
    if q_a.dim() != 2 or w_qb.dim() != 2 or w_uk96.dim() < 1:
        return f"{op}: operand layout (q_a [rows, q_lora], w_qb [q_b rows, q_lora], w_uk96 [heads, ...])"
    call = (q_a.shape[1], w_qb.shape[0], w_uk96.shape[0], key[3])
    if call != key:
        return built_for(op, _KEY, key, call)
    m, heads = q_a.shape[0], key[2]
    if not 1 <= m <= pkg.comm.MAX_ROWS:
        return f"{op}: {m} rows (the kernel takes 1..{pkg.comm.MAX_ROWS})"
    if q_a.dtype != torch.bfloat16 or q_a.stride(1) != 1 or q_a.stride(0) % 8:
        return f"{op}: operand layout (q_a bf16, unit column stride, row stride % 8)"
    wuk = pkg.kernel_wuk
    if w_qb.dtype != torch.bfloat16 or w_uk96.dtype != torch.bfloat16 or not w_qb.is_contiguous() \
            or not w_uk96.is_contiguous() or w_uk96.numel() != heads * wuk.P * wuk.L:
        return f"{op}: operand layout (w_qb, w_uk96 bf16 contiguous, as pack_q_b / pack_w_uk return them)"
    if out is not None and (out.dtype != torch.bfloat16 or not out.is_contiguous() or out.shape != (m, heads, wuk.OW)):
        return f"{op}: operand layout (out bf16 contiguous [rows, {heads}, {wuk.OW}])"
    rec = pkg.fp8q.RECORD
    if out_q8 is not None and (out_q8.dtype != torch.uint8 or not out_q8.is_contiguous()
                               or out_q8.shape != (m, heads, rec)):
        return f"{op}: operand layout (out_q8 uint8 contiguous [rows, {heads}, {rec}])"
    return None


def q_bproj_wuk_gather_kernel(*args, **kwargs):
    """The routed ``amd_kernels.warm_up`` entry (``.entry``) for these arguments."""
    return _routed(args[0] if args else kwargs["comm"]).q_b_allgather_wuk_all_heads
