# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA decode layer in one persistent launch (AttnRes seam to the gated core output): a workspace is built for a
(hidden, heads, head_dim) key (``_SHAPE_HOSTS``, imported on the first build of its key); a call routes by its
workspace's package."""

import functools
import importlib
import sys

import torch

from . import device as _dev

_SHAPE_HOSTS = {(7168, 12, 128): "amd_kernels.attention.kda_mono_decode_k1_h12_d128_ranks8"}
_OP = "kda_mono_decode"


@functools.cache
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


@functools.cache
def _workspace_key(cls):
    for key, name in _SHAPE_HOSTS.items():
        pkg = sys.modules.get(name)
        if pkg is not None and pkg.Workspace is cls:
            return key
    return None


def _key(prefix, rstate):
    if prefix.dim() != 2 or rstate.dim() != 4:
        return None
    return (prefix.shape[1], rstate.shape[1], rstate.shape[3])


@functools.cache
def _scratch_bytes(pkg, tokens):
    k = pkg.host.K
    return k.scratch_bytes(k.KdaPreBuild(tokens=tokens, qlen=tokens, nblocks=1, delta=True, state_len=k.CONV_W - 1))


def _no_workspace_kernel(hidden, heads, head_dim):
    return (f"kda_mono_workspace: no kernel for hidden={hidden}, heads={heads}, head_dim={head_dim} "
            f"(covered: {sorted(_SHAPE_HOSTS)})")


def kda_mono_decode_shapes() -> frozenset[tuple[int, int, int]]:
    """The (hidden, heads, head_dim) with a kernel."""
    return frozenset(_SHAPE_HOSTS)


def kda_mono_workspace(device, *, hidden, heads, head_dim, max_tokens=None, max_blocks=8):
    """The routed package's ``Workspace`` (``begin_step()`` once a step, ``failed()``); ``max_tokens`` None: the
    package's most rows a launch."""
    pkg = _shape_host((hidden, heads, head_dim))
    if pkg is None:
        raise ValueError(_no_workspace_kernel(hidden, heads, head_dim))
    return pkg.Workspace(device, pkg.host.MAX_S if max_tokens is None else max_tokens, max_blocks)


def kda_mono_workspace_unsupported_reason(device, *, hidden, heads, head_dim, max_tokens=None, max_blocks=8):
    """None when ``kda_mono_workspace`` builds a workspace for these arguments, else why not."""
    op = "kda_mono_workspace"
    if not _dev.available():
        return f"{op}: needs gfx950 with FlyDSL"
    pkg = _shape_host((hidden, heads, head_dim))
    if pkg is None:
        return _no_workspace_kernel(hidden, heads, head_dim)
    hi = pkg.host.MAX_S
    if max_tokens is not None and not 1 <= max_tokens <= hi:
        return f"{op}: {max_tokens} rows (the kernel takes 1..{hi})"
    if not 1 <= max_blocks <= 8:
        return f"{op}: {max_blocks} blocks (the kernel takes 1..8)"
    return None


def kda_mono_decode(ws, layer, prefix, delta, blocks, ares_nw, ares_qk, in_nw, w_in, w_fb, conv_w, conv_state, a_log,
                    dt_bias, on_w, rstate, st_idx, num_acc, *, qlen, nblocks, write_idx=-1, eps=1e-5, out_eps=1e-5,
                    onorm_eps=1e-5, lower_bound=-5.0):
    """One KDA layer of a spec-verify step on ``ws`` (``kda_mono_workspace``); returns ``core`` [S, heads * d]."""
    key = _workspace_key(type(ws))
    if key is None:
        raise ValueError(f"{_OP}: {type(ws).__name__} is not a kda_mono_workspace")
    return _shape_host(key).kda_mono_decode_k1(
        ws, layer, prefix, delta, blocks, ares_nw, ares_qk, in_nw, w_in, w_fb, conv_w, conv_state, a_log, dt_bias,
        on_w, rstate, st_idx, num_acc, qlen=qlen, nblocks=nblocks, write_idx=write_idx, eps=eps, out_eps=out_eps,
        onorm_eps=onorm_eps, lower_bound=lower_bound,
    )  # fmt: skip


def kda_mono_decode_unsupported_reason(ws, layer, prefix, delta, blocks, ares_nw, ares_qk, in_nw, w_in, w_fb, conv_w,
                                       conv_state, a_log, dt_bias, on_w, rstate, st_idx, num_acc, *, qlen, nblocks,
                                       write_idx=-1, eps=1e-5, out_eps=1e-5, onorm_eps=1e-5,
                                       lower_bound=-5.0) -> str | None:
    """None when ``kda_mono_decode`` serves this call, else why not (host checks only)."""
    if not _dev.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(prefix, rstate)
    if key not in _SHAPE_HOSTS:
        return (f"{_OP}: no kernel for prefix {tuple(prefix.shape)}, recurrent state {tuple(rstate.shape)} "
                f"(covered (hidden, heads, head_dim): {sorted(_SHAPE_HOSTS)})")
    ws_key = _workspace_key(type(ws))
    if ws_key is None:
        return f"{_OP}: {type(ws).__name__} is not a kda_mono_workspace"
    if ws_key != key:
        return f"{_OP}: workspace built for {ws_key}, call is {key}"
    pkg = _shape_host(key)
    h = pkg.host
    s, sl = prefix.size(0), conv_state.size(-1)
    if not 1 <= s <= h.MAX_S:
        return f"{_OP}: {s} rows (the kernel takes 1..{h.MAX_S})"
    if qlen < 1 or s % qlen or qlen > sl:
        return f"{_OP}: {s} rows of {qlen}-token requests, conv state length {sl} (rows % qlen == 0, qlen <= length)"
    if ws.scratch.numel() < _scratch_bytes(pkg, s):
        return f"{_OP}: workspace scratch {ws.scratch.numel()} B for {s} rows (built for fewer rows)"
    if not 0 <= layer < h.K.TAG_SLOTS:
        return f"{_OP}: layer slot {layer} (0..{h.K.TAG_SLOTS - 1})"
    bf, f32 = torch.bfloat16, torch.float32
    if prefix.dtype != bf or not prefix.is_contiguous() or (delta is not None and (
            delta.dtype != bf or not delta.is_contiguous() or delta.shape != prefix.shape)):
        return f"{_OP}: prefix {prefix.dtype} / delta (contiguous bf16 [{s}, {h.HIDDEN}])"
    if blocks.dtype != bf or blocks.dim() != 3 or blocks.size(0) != s or blocks.size(2) != h.HIDDEN or (
            blocks.stride(2) != 1 or blocks.stride(0) % 8 or blocks.stride(1) % 8 or blocks.data_ptr() % 16):
        return f"{_OP}: blocks {tuple(blocks.shape)} stride {blocks.stride()} ([{s}, n, {h.HIDDEN}], 16-byte rows)"
    if not 1 <= nblocks <= min(8, blocks.size(1)) or not (write_idx < 0 or nblocks <= write_idx < blocks.size(1)):
        return f"{_OP}: nblocks {nblocks}, write_idx {write_idx} of {blocks.size(1)} blocks (nblocks 1..8)"
    if tuple(w_in.shape) != (h.NPROJ, h.HIDDEN) or w_in.dtype != bf or not w_in.is_contiguous():
        return f"{_OP}: w_in {tuple(w_in.shape)} {w_in.dtype} (contiguous bf16 [{h.NPROJ}, {h.HIDDEN}])"
    if tuple(w_fb.shape) != (h.PROJ, h.HD) or w_fb.dtype != bf or not w_fb.is_contiguous():
        return f"{_OP}: w_fb {tuple(w_fb.shape)} {w_fb.dtype} (contiguous bf16 [{h.PROJ}, {h.HD}])"
    vectors = (("ares_nw", ares_nw, bf, h.HIDDEN), ("ares_qk", ares_qk, bf, h.HIDDEN), ("in_nw", in_nw, bf, h.HIDDEN),
               ("on_w", on_w, bf, h.HD), ("conv_w", conv_w, f32, 3 * h.PROJ * h.K.CONV_W), ("a_log", a_log, f32, h.NH),
               ("dt_bias", dt_bias, f32, h.PROJ))
    for name, x, dt, n in vectors:
        if x.dtype != dt or x.numel() != n or not x.is_contiguous():
            return f"{_OP}: {name} {tuple(x.shape)} {x.dtype} (contiguous {dt} of {n})"
    if conv_state.dim() != 3 or conv_state.dtype != bf or conv_state.size(1) != 3 * h.PROJ:
        return f"{_OP}: conv state {tuple(conv_state.shape)} {conv_state.dtype} (bf16 [slots, {3 * h.PROJ}, length])"
    if rstate.dtype != torch.float32 or rstate.shape[2] != h.HD or rstate.stride()[1:] != (h.HD * h.HD, h.HD, 1):
        return f"{_OP}: recurrent state {tuple(rstate.shape)} {rstate.dtype} stride {rstate.stride()}"
    if st_idx.dtype != torch.int32 or st_idx.dim() != 2 or st_idx.stride(1) != 1 or num_acc.dtype != torch.int32:
        return f"{_OP}: st_idx {st_idx.dtype} / num_acc {num_acc.dtype} (int32, unit column stride)"
    return None
