# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DCP LSE combine + W_UV up-projection (8 ranks): key (heads, v_head, kv_lora, world), the dims of ``w_uv_t``
([heads, v_head, kv_lora], ``prepare_w_uv``) and the group's size; not ``attn_out``, whose head dim may be padded.
The communicator factory takes the dims as keywords; a call routes by its communicator."""

import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare, renamed

from .routing import built_for

_KEY = ("heads", "v_head", "kv_lora", "world")
_ROUTES = Routes("lse_combine_vup", {(96, 128, 512, 8): "amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512"},
                 _KEY)
_COMM_ROUTES = Routes("lse_combine_comm", _ROUTES.hosts, _KEY)
_OP = _ROUTES.op
_COMM = _COMM_ROUTES.op


def lse_combine_comm(rank, group, device, *, heads, v_head, kv_lora, max_tokens=None, is_lse_base_on_e=True):
    """The package's communicator for this shape on the ranks of ``group`` (a collective); ``max_tokens`` None: the
    package's ``MAX_TOKENS``."""
    pkg = _COMM_ROUTES.package((heads, v_head, kv_lora, dist.get_world_size(group)))
    if max_tokens is None:
        max_tokens = pkg.host.MAX_TOKENS
    return pkg.DcpVupCombine(rank, group, device, max_tokens=max_tokens, is_lse_base_on_e=is_lse_base_on_e)


def lse_combine_comm_blocker(rank, group, device, *, heads, v_head, kv_lora, max_tokens=None,
                             is_lse_base_on_e=True) -> str | None:
    if not dist.is_initialized():
        return f"{_COMM}: needs an initialized torch.distributed process group"
    key = (heads, v_head, kv_lora, dist.get_world_size(group))
    reason = _COMM_ROUTES.blocker(key)
    if reason is not None:
        return reason
    if not routes.available():
        return f"{_COMM}: needs gfx950 with FlyDSL"
    cap = _COMM_ROUTES.package(key).host.MAX_TOKENS
    if max_tokens is not None and not 1 <= max_tokens <= cap:
        return f"{_COMM}: max_tokens {max_tokens} (the kernel takes 1..{cap})"
    return None


def prepare_lse_combine_comm(rank, group, device, *, heads, v_head, kv_lora, max_tokens=None, is_lse_base_on_e=True):
    """``lse_combine_comm(...)`` (a collective), or the reason it cannot be built."""
    kw = dict(heads=heads, v_head=v_head, kv_lora=kv_lora, max_tokens=max_tokens, is_lse_base_on_e=is_lse_base_on_e)
    return prepare(lse_combine_comm_blocker(rank, group, device, **kw),
                   lambda: lse_combine_comm(rank, group, device, **kw))


def prepare_w_uv(w_uv):
    """The load-time W_UV of all heads [heads, kv_lora, v_head] as the kernel reads it, [heads, v_head, kv_lora]."""
    dims = (w_uv.shape[0], w_uv.shape[2], w_uv.shape[1]) if w_uv.dim() == 3 else None
    key = next((k for k in _ROUTES.hosts if k[:3] == dims), None)
    if key is None:
        raise ValueError(f"{_OP}: no kernel for W_UV {tuple(w_uv.shape)} (covered: {sorted(_ROUTES.hosts)})")
    return _ROUTES.package(key).prepare_w_uv(w_uv)


def _no_comm(comm):
    return f"{_OP}: no kernel for communicator {type(comm).__qualname__} (covered: {sorted(_ROUTES.hosts)})"


def _routed(comm):
    key = _ROUTES.owner(type(comm))
    if key is None:
        raise ValueError(_no_comm(comm))
    return _ROUTES.package(key)


def lse_combine_vup(comm, attn_out, lse, w_uv_t, out, seq_lens=None, query_start_loc=None, gate=None):
    """This rank's heads of the LSE-combined, up-projected attention output into ``out`` [rows, heads / world,
    v_head] (times sigmoid(``gate``) when given); returns ``out``."""
    return _routed(comm).dcp8_lse_combine_vup(comm, attn_out, lse, w_uv_t, out, seq_lens, query_start_loc, gate)


def lse_combine_vup_shapes() -> frozenset:
    return _ROUTES.shapes()


def lse_combine_vup_blocker(comm, attn_out, lse, w_uv_t, out, seq_lens=None, query_start_loc=None,
                            gate=None) -> str | None:
    op = _OP
    key = _ROUTES.owner(type(comm))
    if key is None:
        return _no_comm(comm)
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    if w_uv_t.dim() != 3:
        return f"{op}: operand layout (w_uv_t [heads, v_head, kv_lora])"
    call = (*w_uv_t.shape, key[3])
    if call != key:
        return built_for(op, _KEY, key, call)
    heads, v_head, kv_lora, world = key
    local = heads // world
    if attn_out.dim() != 3 or lse.dim() != 2 or out.dim() != 3:
        return f"{op}: operand layout (attn_out [rows, >= heads, kv_lora], lse [rows, >= heads], out 3-D)"
    m = attn_out.shape[0]
    cap = min(_ROUTES.package(key).host.MAX_TOKENS, comm.max_tokens)
    if not 1 <= m <= cap:
        return f"{op}: {m} rows (the kernel takes 1..{cap})"
    if attn_out.dtype != torch.bfloat16 or attn_out.shape[1] < heads or attn_out.shape[2] != kv_lora \
            or attn_out.stride(2) != 1 or attn_out.stride(0) % 2 or attn_out.stride(1) % 2:
        return f"{op}: operand layout (attn_out bf16 [rows, >= {heads}, {kv_lora}], unit column stride, even strides)"
    if lse.dtype != torch.float32 or lse.shape[0] != m or lse.shape[1] < heads:
        return f"{op}: operand layout (lse fp32 [rows, >= {heads}])"
    if w_uv_t.dtype != torch.bfloat16 or not w_uv_t.is_contiguous():
        return f"{op}: operand layout (w_uv_t bf16 contiguous, as prepare_w_uv returns it)"
    if out.dtype != torch.bfloat16 or not out.is_contiguous() or out.shape != (m, local, v_head):
        return f"{op}: operand layout (out bf16 contiguous [rows, {local}, {v_head}])"
    if gate is not None and (gate.dtype != torch.bfloat16 or not gate.is_contiguous()
                             or gate.shape != (m, local * v_head)):
        return f"{op}: operand layout (gate bf16 contiguous [rows, {local * v_head}])"
    if seq_lens is not None and (seq_lens.dtype != torch.int32 or query_start_loc is None
                                 or query_start_loc.dtype != torch.int32):
        return f"{op}: operand layout (seq_lens and query_start_loc int32)"
    return None


def lse_combine_vup_kernel(*args, **kwargs):
    """The routed ``amd_kernels.warm_up`` entry (``.entry``) for these arguments."""
    return _routed(args[0] if args else kwargs["comm"]).dcp8_lse_combine_vup


lse_combine_comm_unsupported_reason = renamed(lse_combine_comm_blocker, "lse_combine_comm_unsupported_reason")
lse_combine_vup_unsupported_reason = renamed(lse_combine_vup_blocker, "lse_combine_vup_unsupported_reason")
