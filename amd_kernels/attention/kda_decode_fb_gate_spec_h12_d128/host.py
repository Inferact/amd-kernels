# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache, launch wrapper and warm-up entry of the KDA spec-verify decode kernel."""

import functools

import torch

from amd_kernels.warm_up import entry

from .common import LOG2E
from .kernel import COL_BETA, NH, D, LP, compile_kda_decode


def _run_compiled(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


MAX_TOKENS = 8
LPR = 8
STORE_CM = 2
WAVES = 8


@functools.cache
def _get_kernel(max_tok, nw_f32, snap, uniform):
    return compile_kda_decode(max_tok, nw_f32, WAVES, LPR, STORE_CM, snap, uniform)


def blocker(proj, f_b_weight, conv_state, state, state_indices, norm_weight, out):
    """Why a call does not fit the kernel (None: it fits): bf16 fused in_proj rows with 16-byte aligned stride, the
    (state_len, dim) conv state layout with 16-byte aligned positions, fp32 states, int32 tables of at most 8 tokens."""
    cs = conv_state
    if proj.dtype != torch.bfloat16 or proj.stride(1) != 1 or proj.stride(0) % 8 or proj.shape[1] < COL_BETA + NH:
        return f"proj {tuple(proj.shape)} {proj.dtype} stride {proj.stride()} (bf16 rows, stride a multiple of 8)"
    if f_b_weight.shape != (LP, D) or f_b_weight.dtype != torch.bfloat16 or not f_b_weight.is_contiguous():
        return f"f_b weight {tuple(f_b_weight.shape)} {f_b_weight.dtype}"
    if state.dtype != torch.float32 or state.shape[1:] != (NH, D, D) or state.stride(1) != D * D:
        return f"state {tuple(state.shape)} {state.dtype} (fp32 [slots, 12, 128, 128])"
    if cs.dim() != 3 or cs.shape[1] != 3 * LP or cs.dtype != torch.bfloat16 or cs.stride(1) != 1:
        return f"conv state {tuple(cs.shape)} strides {cs.stride()} (bf16, dim innermost)"
    if cs.stride(0) % 8 or cs.stride(2) % 8 or cs.shape[2] < state_indices.shape[1] + 2:
        return f"conv state strides {cs.stride()} / length {cs.shape[2]} (multiples of 8, at least tokens + 2)"
    if state_indices.dtype != torch.int32 or state_indices.stride(1) != 1 or not 1 <= state_indices.shape[1] <= MAX_TOKENS:
        return f"state indices {tuple(state_indices.shape)} {state_indices.dtype}"
    if norm_weight.shape != (D,) or not norm_weight.is_contiguous() or norm_weight.dtype not in (torch.bfloat16, torch.float32):
        return f"norm weight {tuple(norm_weight.shape)} {norm_weight.dtype}"
    if out.shape[1:] != (NH, D) or out.dtype != torch.bfloat16 or out.stride(1) != D or out.stride(2) != 1:
        return f"out {tuple(out.shape)} {out.dtype}"
    return None


def _kda_decode_fb_gate_spec(
    proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu_seqlens, state_indices,
    num_accepted, norm_weight, norm_eps, out, snapshot_token=None, uniform=False,
):
    n_seq = cu_seqlens.shape[0] - 1
    if n_seq <= 0 or proj.shape[0] == 0:
        return
    reason = blocker(proj, f_b_weight, conv_state, state, state_indices, norm_weight, out)
    assert reason is None, reason
    assert cu_seqlens.dtype == num_accepted.dtype == torch.int32
    max_tok = state_indices.shape[1]
    ps = proj.stride(0)
    nw_f32 = norm_weight.dtype == torch.float32
    cs = conv_state
    snap = snapshot_token is not None
    rows = proj.shape[0]
    common = (
        proj.data_ptr(), ps, rows * ps * 2, f_b_weight.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(),
        conv_weight.data_ptr(), cs.data_ptr(), cs.stride(0) * cs.element_size(), cs.stride(1), cs.stride(2),
        state.data_ptr(), state.stride(0) * 4, cu_seqlens.data_ptr(), state_indices.data_ptr(),
        state_indices.stride(0), num_accepted.data_ptr(),
        (snapshot_token if snap else num_accepted).data_ptr(), norm_weight.data_ptr(), out.data_ptr(), out.stride(0),
        out.shape[0] * out.stride(0) * 2, float(lower_bound) * LOG2E, float(norm_eps),
    )
    _run_compiled(_get_kernel(max_tok, nw_f32, snap, uniform), *common, n_seq, torch.cuda.current_stream())


def _mode(proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu_seqlens, state_indices,
          num_accepted, norm_weight, norm_eps, out, snapshot_token=None, uniform=False):
    """What a call fixes about its kernel (``amd_kernels.warm_up``): the tokens per sequence and the norm weight
    dtype; the token and sequence counts are runtime arguments."""
    return (state_indices.shape[1], norm_weight.dtype == torch.float32, snapshot_token is not None, uniform,
            conv_state.stride(1) == 1)


def _inputs(mode, n, proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu_seqlens,
            state_indices, num_accepted, norm_weight, norm_eps, out, snapshot_token=None, uniform=False):
    """One sequence of ``n`` tokens (zeros; slot 0 is padding, nothing launches)."""
    dev = proj.device
    x = torch.zeros((n, proj.shape[1]), dtype=proj.dtype, device=dev)
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    idx = torch.zeros((1, state_indices.shape[1]), dtype=torch.int32, device=dev)
    acc = torch.ones(1, dtype=torch.int32, device=dev)
    o = torch.empty((n, NH, D), dtype=torch.bfloat16, device=dev)
    return (x, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu, idx, acc, norm_weight,
            norm_eps, o, acc if mode[2] else None, mode[3]), {}


kda_decode_fb_gate_spec = entry(
    "attention/kda_decode_fb_gate_spec_h12_d128", mode=_mode, sizes=lambda mode: (1,), inputs=_inputs
)(_kda_decode_fb_gate_spec)
