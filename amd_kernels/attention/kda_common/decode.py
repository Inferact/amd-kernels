# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA spec-verify decode with the f_b gate: routes a call by its recurrent state's (heads, head_dim) to the shape
package (``_SHAPE_HOSTS``, imported on the first call of its key)."""

import functools
import importlib

import torch

from . import device as _dev

_SHAPE_HOSTS = {(12, 128): "amd_kernels.attention.kda_decode_fb_gate_spec_h12_d128"}
_OP = "kda_decode_fb_gate_spec"


@functools.cache
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


def _key(state):
    return (state.shape[1], state.shape[3]) if state.dim() == 4 else None


def _no_kernel(state):
    if state.dim() != 4:
        return f"{_OP}: state {tuple(state.shape)} (fp32 [slots, heads, head_dim, head_dim])"
    return f"{_OP}: no kernel for heads={state.shape[1]}, head_dim={state.shape[3]} (covered: {sorted(_SHAPE_HOSTS)})"


def kda_decode_fb_gate_spec_shapes() -> frozenset[tuple[int, int]]:
    """The (heads, head_dim) with a kernel."""
    return frozenset(_SHAPE_HOSTS)


def kda_decode_fb_gate_spec(proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu_seqlens,
                            state_indices, num_accepted, norm_weight, norm_eps, out, snapshot_token=None,
                            uniform=False):
    """The KDA core of one layer in a spec-verify step (the routed package's ``kda_decode_fb_gate_spec``)."""
    pkg = _shape_host(_key(state))
    if pkg is None:
        raise ValueError(_no_kernel(state))
    pkg.kda_decode_fb_gate_spec(proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state,
                                cu_seqlens, state_indices, num_accepted, norm_weight, norm_eps, out,
                                snapshot_token=snapshot_token, uniform=uniform)


def kda_decode_fb_gate_spec_unsupported_reason(proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state,
                                               state, cu_seqlens, state_indices, num_accepted, norm_weight, norm_eps,
                                               out, snapshot_token=None, uniform=False) -> str | None:
    """None when ``kda_decode_fb_gate_spec`` serves this call, else why not (host checks only)."""
    if not _dev.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    pkg = _shape_host(_key(state))
    if pkg is None:
        return _no_kernel(state)
    if proj.dim() != 2 or state_indices.dim() != 2:
        return f"{_OP}: proj {tuple(proj.shape)} / state indices {tuple(state_indices.shape)} (2-D)"
    max_tokens = pkg.host.MAX_TOKENS
    if not 1 <= state_indices.shape[1] <= max_tokens:
        return f"{_OP}: {state_indices.shape[1]} tokens a sequence (the kernel takes 1..{max_tokens})"
    reason = pkg.blocker(proj, f_b_weight, conv_state, state, state_indices, norm_weight, out)
    if reason is not None:
        return f"{_OP}: {reason}"
    h, d = state.shape[1], state.shape[3]
    if out.shape[0] < proj.shape[0]:
        return f"{_OP}: out {tuple(out.shape)} for {proj.shape[0]} rows"
    n_seq, t = cu_seqlens.numel() - 1, state_indices.shape[1]
    if uniform and proj.shape[0] < n_seq * t:
        return f"{_OP}: uniform with {proj.shape[0]} rows for {n_seq} sequences of {t} tokens"
    if (A_log.numel() != h or dt_bias.numel() != h * d or tuple(conv_weight.shape) != (3, 4, h * d)
            or not A_log.dtype == dt_bias.dtype == conv_weight.dtype == torch.float32
            or not (A_log.is_contiguous() and dt_bias.is_contiguous() and conv_weight.is_contiguous())):
        return (f"{_OP}: A_log {tuple(A_log.shape)} {A_log.dtype}, dt_bias {tuple(dt_bias.shape)} {dt_bias.dtype}, "
                f"conv weight {tuple(conv_weight.shape)} {conv_weight.dtype} (contiguous fp32 [{h}], [{h * d}], "
                f"[3, 4, {h * d}])")
    snap = snapshot_token.dtype if snapshot_token is not None else torch.int32
    if not cu_seqlens.dtype == num_accepted.dtype == snap == torch.int32:
        return f"{_OP}: cu_seqlens, num_accepted, snapshot {cu_seqlens.dtype}, {num_accepted.dtype}, {snap} (int32)"
    return None
