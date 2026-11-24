# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Validate the layout before passing tensor addresses to the gfx950 kernel."""

import importlib.util

import torch

from amd_kernels.warm_up import entry

H, D = 12, 128


def _rows(tensor, count, tail):
    shape = (count, *tail)
    return (
        tensor.shape == shape
        and tensor.dtype == torch.bfloat16
        and tensor.stride()[-1] == 1
        and tensor.stride(0) >= tail.numel()
        and (len(tail) == 1 or tensor.stride(1) == tail[-1])
    )


def unsupported_reason(
    qkv,
    gate,
    beta,
    out_gate,
    alog,
    bias,
    state,
    offsets,
    slots,
    weight,
    correction,
    key_gate,
    out,
    lower_bound,
    eps,
):
    """Check tensor metadata without reading device values or synchronizing the stream."""
    if qkv.ndim != 2 or slots.ndim != 1 or correction.ndim != 4:
        return "expected packed QKV rows, a slot vector and four-dimensional correction records"
    rows, batch = qkv.shape[0], slots.numel()
    capacity, _, window, _ = correction.shape
    if not 1 <= window <= 8 or capacity < batch or rows > batch * window:
        return "expected 1..8 tokens per sequence and sufficient record capacity"
    for tensor, tail in ((qkv, (3 * H * D,)), (gate, (H, D)), (beta, (H,)), (out_gate, (H, D)), (out, (H, D))):
        if not _rows(tensor, rows, torch.Size(tail)):
            return "expected BF16 activation rows with contiguous heads"
    for tensor in (qkv, gate, out_gate, out):
        if tensor.stride(0) % 2 or tensor.data_ptr() % 4:
            return "packed BF16 accesses require dword-aligned rows"
    if (
        state.ndim != 4
        or state.shape[1:] != (H, D, D)
        or state.dtype != torch.float32
        or state.stride()[1:] != (D * D, D, 1)
        or state.stride(0) < H * D * D
    ):
        return "expected FP32 checkpoint slots with dense [12,128,128] heads"
    if (
        offsets.shape != (batch + 1,)
        or offsets.dtype != torch.int32
        or not offsets.is_contiguous()
        or slots.dtype != torch.int32
        or slots.stride(0) < 1
    ):
        return "expected int32 cumulative offsets and checkpoint slots"
    if (
        correction.shape != (capacity, H, window, D)
        or correction.dtype != torch.float32
        or key_gate.shape != (capacity, H, window, 2 * D)
        or key_gate.dtype != torch.bfloat16
        or not (correction.is_contiguous() and key_gate.is_contiguous())
    ):
        return "expected contiguous FP32 correction and BF16 concatenated key/gate records"
    if (
        alog.shape != (H,)
        or alog.dtype != torch.float32
        or bias.numel() != H * D
        or bias.dtype != torch.float32
        or weight.shape != (D,)
        or weight.dtype not in (torch.float32, torch.bfloat16)
        or not (alog.is_contiguous() and bias.is_contiguous() and weight.is_contiguous())
    ):
        return "expected contiguous FP32 decay parameters and a BF16 or FP32 head norm weight"
    if lower_bound is None or not lower_bound < 0 or not eps > 0:
        return "expected a negative safe-gate bound and positive norm epsilon"
    tensors = (qkv, gate, beta, out_gate, alog, bias, state, offsets, slots, weight, correction, key_gate, out)
    if any(t.device != qkv.device for t in tensors):
        return "all tensors must be on the same device"
    if any(t.stride(0) > 0x7FFFFFFF for t in (qkv, gate, beta, out_gate, out, slots)):
        return "row strides must fit int32"
    if not qkv.is_cuda or not getattr(torch.version, "hip", None):
        return "needs ROCm gfx950"
    if "gfx950" not in torch.cuda.get_device_properties(qkv.device).gcnArchName:
        return "needs gfx950"
    if importlib.util.find_spec("flydsl") is None:
        return "needs FlyDSL"
    return None


def _mode(
    qkv,
    gate,
    beta,
    out_gate,
    alog,
    bias,
    state,
    offsets,
    slots,
    weight,
    correction,
    key_gate,
    out,
    lower_bound,
    eps,
):
    return correction.shape[2], weight.dtype


def _inputs(
    mode,
    n,
    qkv,
    gate,
    beta,
    out_gate,
    alog,
    bias,
    state,
    offsets,
    slots,
    weight,
    correction,
    key_gate,
    out,
    lower_bound,
    eps,
):
    dev = qkv.device

    def zeros(*shape):
        return torch.zeros(shape, device=dev, dtype=torch.bfloat16)

    cu = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    idx = torch.zeros(1, dtype=torch.int32, device=dev)
    return (
        zeros(1, 3 * H * D),
        zeros(1, H, D),
        zeros(1, H),
        zeros(1, H, D),
        alog,
        bias,
        state,
        cu,
        idx,
        weight,
        correction,
        key_gate,
        zeros(1, H, D),
        lower_bound,
        eps,
    ), {}


@entry("attention/kda_verify_rmsnorm_h12_d128", mode=_mode, sizes=lambda mode: (1,), inputs=_inputs)
def _run(
    qkv,
    gate,
    beta,
    out_gate,
    alog,
    bias,
    state,
    offsets,
    slots,
    weight,
    correction,
    key_gate,
    out,
    lower_bound,
    eps,
):
    import flydsl.compiler as flyc
    from .kernel import build

    launch = build(correction.shape[2], weight.dtype == torch.float32)
    tensors = (qkv, gate, beta, out_gate, alog, bias, state, offsets, slots, weight, correction, key_gate, out)
    args = (
        *(t.data_ptr() for t in tensors),
        qkv.stride(0),
        gate.stride(0),
        beta.stride(0),
        out_gate.stride(0),
        state.stride(0),
        slots.stride(0),
        out.stride(0),
        float(lower_bound),
        float(eps),
        slots.numel(),
        torch.cuda.current_stream(qkv.device),
    )
    compiled = getattr(launch, "_cf", None)
    if compiled is None:
        launch._cf = flyc.compile(launch, *args)
    else:
        compiled(*args)


def kda_verify_rmsnorm(
    qkv,
    gate,
    beta,
    out_gate,
    alog,
    bias,
    state,
    offsets,
    slots,
    weight,
    correction,
    key_gate,
    out,
    lower_bound,
    eps,
):
    """Write gated outputs and per-token recovery records, leaving recurrent checkpoints read-only."""
    args = (
        qkv,
        gate,
        beta,
        out_gate,
        alog,
        bias,
        state,
        offsets,
        slots,
        weight,
        correction,
        key_gate,
        out,
        lower_bound,
        eps,
    )
    why = unsupported_reason(*args)
    if why is not None:
        raise ValueError(why)
    if qkv.shape[0]:
        with torch.cuda.device(qkv.device):
            _run(*args)
    return out
