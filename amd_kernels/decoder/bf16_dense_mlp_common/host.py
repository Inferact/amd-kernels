# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dense MLP shard on packed bf16 weights: routes by the down weight's (hidden, inter) to the shape package's host
(``MlpWorkspace``, ``dense_mlp``, ``pack_gate_up``, ``pack_down``, ``packed_tiles``, ``packed_rows_buffer``,
``MAX_ROWS``). The workspace belongs to one key: the reason checks it against the call's weights. The packed weights
keep the raw weights' shapes, so no check can tell them apart: pass ``dense_mlp_pack_*``'s output."""
import functools
import importlib

import torch

from . import available

OP = "bf16_dense_mlp"
_DIMS = ("hidden", "inter")
_SHAPE_HOSTS = {
    (7168, 1792): "amd_kernels.decoder.bf16_dense_mlp_h7168_i1792.host",
}


@functools.cache
def _shape_host(key: tuple):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


@functools.cache
def _ws_key(cls: type):
    return next((key for key in _SHAPE_HOSTS if getattr(_shape_host(key), cls.__name__, None) is cls), None)


def _no_kernel(op: str, key: tuple) -> str:
    dims = ", ".join(f"{d}={v}" for d, v in zip(_DIMS, key)) if len(key) == len(_DIMS) else f"shape={key}"
    return f"{op}: no kernel for {dims} (covered: {sorted(_SHAPE_HOSTS)})"


def _host(op: str, key: tuple):
    host = _shape_host(key)
    if host is None:
        raise ValueError(_no_kernel(op, key))
    return host


def bf16_dense_mlp_shapes() -> frozenset[tuple[int, int]]:
    """The covered (hidden, inter) = the down weight's shape."""
    return frozenset(_SHAPE_HOSTS)


def dense_mlp_workspace(device, *, hidden: int, inter: int):
    """The routed package's per-stream workspace (allocated once, graph safe)."""
    return _host("dense_mlp_workspace", (hidden, inter)).MlpWorkspace(device)


def dense_mlp_workspace_unsupported_reason(device, *, hidden: int, inter: int) -> str | None:
    op = "dense_mlp_workspace"
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    if _shape_host((hidden, inter)) is None:
        return _no_kernel(op, (hidden, inter))
    return None


def dense_mlp_pack_gate_up(w: torch.Tensor) -> torch.Tensor:
    """The merged [2 inter, hidden] gate / up weight (gate rows first) -> the packed layout the op reads."""
    key = (w.shape[1], w.shape[0] // 2) if w.ndim == 2 and w.shape[0] % 2 == 0 else tuple(w.shape)
    return _host("dense_mlp_pack_gate_up", key).pack_gate_up(w)


def dense_mlp_pack_down(w: torch.Tensor) -> torch.Tensor:
    """The [hidden, inter] down weight -> the packed layout the op reads."""
    return _host("dense_mlp_pack_down", tuple(w.shape)).pack_down(w)


def dense_mlp_packed_tiles(rows: int, *, down: torch.Tensor) -> int:
    """16-row tiles of the packed activations the op reads for ``rows`` rows."""
    return _host("dense_mlp_packed_tiles", tuple(down.shape)).packed_tiles(rows)


def dense_mlp_rows_buffer(rows: int, device, *, down: torch.Tensor) -> torch.Tensor:
    """An uninitialized packed-activation buffer of ``rows`` rows."""
    return _host("dense_mlp_rows_buffer", tuple(down.shape)).packed_rows_buffer(rows, device)


def bf16_dense_mlp(ws, x_packed: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, rows: int,
                   out: torch.Tensor | None = None) -> torch.Tensor:
    """This rank's partial MLP output [rows, hidden] bf16 = down(silu(gate) * up) (the routed package's
    ``dense_mlp``; the caller all-reduces it)."""
    host = _shape_host(tuple(down.shape))
    if host is None:
        raise ValueError(_no_kernel(OP, tuple(down.shape)))
    return host.dense_mlp(ws, x_packed, gate_up, down, rows, out)


def bf16_dense_mlp_unsupported_reason(ws, x_packed: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor,
                                      rows: int, out: torch.Tensor | None = None) -> str | None:
    """None if ``bf16_dense_mlp`` serves this call, else why not (host checks only)."""
    if not available():
        return f"{OP}: needs gfx950 with FlyDSL"
    key = tuple(down.shape)
    host = _shape_host(key)
    if host is None:
        return _no_kernel(OP, key)
    ws_key = _ws_key(type(ws))
    if ws_key is None:
        return f"{OP}: no kernel for workspace {type(ws).__qualname__} (covered: {sorted(_SHAPE_HOSTS)})"
    if ws_key != key:
        return f"{OP}: workspace built for {ws_key}, call is {key}"
    hidden, inter = key
    dev = down.device
    if down.dtype != torch.bfloat16 or not down.is_contiguous():
        return f"{OP}: down must be contiguous bf16"
    if (tuple(gate_up.shape) != (2 * inter, hidden) or gate_up.dtype != torch.bfloat16 or not gate_up.is_contiguous()
            or gate_up.device != dev):
        return f"{OP}: gate_up must be contiguous bf16 [{2 * inter}, {hidden}] on down's device"
    if ws.slab.device != dev:
        return f"{OP}: workspace on {ws.slab.device}, weights on {dev}"
    if not 1 <= rows <= host.MAX_ROWS:
        return f"{OP}: {rows} rows (the kernel takes 1..{host.MAX_ROWS})"
    tiles = host.packed_tiles(rows)
    if (x_packed.dtype != torch.bfloat16 or not x_packed.is_contiguous() or x_packed.ndim != 4
            or tuple(x_packed.shape[1:]) != (hidden // 32, 64, 8) or x_packed.shape[0] < tiles
            or x_packed.device != dev):
        return f"{OP}: x_packed must be contiguous bf16 [>= {tiles}, {hidden // 32}, 64, 8] on down's device"
    if out is not None and (tuple(out.shape) != (rows, hidden) or out.dtype != torch.bfloat16
                            or not out.is_contiguous() or out.device != dev):
        return f"{OP}: out must be contiguous bf16 [{rows}, {hidden}] on down's device"
    return None
