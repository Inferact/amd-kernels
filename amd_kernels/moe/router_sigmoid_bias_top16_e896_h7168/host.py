# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the MoE router: the load-time weight layout, K split by rows, the launches, the warm-up."""
import functools
import struct

import torch

from amd_kernels.warm_up import entry

from .aux import WORKSPACE_WORDS, aux_capacity
from .kernel import FLAG_WORDS, MT_MAX, compile_router_e896_h7168, partial_bytes, slab_rows
from .reference import E, H, TOPK
from .topk import compile_router_sigmoid_bias_top16_e896

MAX_M = 16 * MT_MAX
# row tile (16 rows) -> (K splits, waves along the columns, fragments per column wave[, row splits])
# measured (benchmark_router.py, graph, min of 15): up to 48 rows more K splits to use every CU; above, two row splits of
# half the K splits (half the slab bytes; the pair of workgroups sharing the weights sits on one XCD)
CONFIG_BY_MT = {mt: (16, 4, 1) if mt <= 3 else (8, 4, 1, 2) for mt in range(1, MT_MAX + 1)}


def supports(w: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    """Whether ``w`` is the [896, 7168] bf16 contiguous router weight (either layout) and ``bias`` the fp32 [896]
    correction bias (or None)."""
    ok = tuple(w.shape) == (E, H) and w.dtype == torch.bfloat16 and w.is_contiguous()
    return ok and (bias is None or (tuple(bias.shape) == (E,) and bias.dtype == torch.float32))


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """The load-time layout [56 groups, 112 K blocks, 16 rows, 64 K] of the router weight, as [896, 7168] (bit-exact)."""
    assert w.shape == (E, H) and w.dtype == torch.bfloat16
    return w.view(E // 16, 16, H // 64, 64).permute(0, 2, 1, 3).contiguous().view(E, H)


def shuffle_weight_(w: torch.Tensor) -> torch.Tensor:
    """shuffle_weight in place: ``w`` keeps its storage and holds the load-time layout."""
    ws = shuffle_weight(w)
    w.copy_(ws)
    del ws
    return w


def unshuffle_weight(ws: torch.Tensor) -> torch.Tensor:
    assert ws.shape == (E, H) and ws.dtype == torch.bfloat16
    return ws.view(E // 16, H // 64, 16, 64).permute(0, 2, 1, 3).contiguous().view(E, H)


def m_tile_fragments(m: int) -> int:
    return (m + 15) // 16


@functools.cache
def _router_kernel(mt: int, ks: int, shuf: bool, fused: bool, wn: int, nrw: int, aux: bool, ms: int = 1):
    return compile_router_e896_h7168(mt, ks, int(shuf), int(fused), wn, nrw, int(aux), 0, 0, ms)


@functools.cache
def _topk_kernel(ks: int):
    return compile_router_sigmoid_bias_top16_e896(ks)


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


_WORK: dict = {}


def _workspace(device: torch.device, stream: int, mt: int, config: tuple):
    """The partial slabs and the epoch words (zero, never reset: each call advances each by one) of one (device, stream,
    row tile, config). A call writes every word of its slabs, so the words an epoch finds are those of the previous call
    of the same workspace (the slab parity tag)."""
    key = (device.index, stream, mt, config)
    ws = _WORK.get(key)
    if ws is None:
        ks, wn, _, ms = _config(mt, config)
        ws = (torch.zeros(partial_bytes(mt, ks, wn, ms), dtype=torch.uint8, device=device),
              torch.zeros(FLAG_WORDS, dtype=torch.int32, device=device))
        _WORK[key] = ws
    return ws


_AUX_WORK: dict = {}


def _aux_workspace(device: torch.device, stream: int, m: int):
    """The expert ids the sort's token waves exchange (tagged with the call's parity) of one (device, stream, M): a call
    writes the ids of all M tokens, so the words it finds are those of the previous call with the same M."""
    key = (device.index, stream, m)
    ws = _AUX_WORK.get(key)
    if ws is None:
        ws = _AUX_WORK[key] = torch.zeros(WORKSPACE_WORDS, dtype=torch.int32, device=device)
    return ws


def _config(mt: int, config: tuple | None = None) -> tuple:
    """(K splits, column waves, fragments per column wave, row splits) of a row tile."""
    config = config or CONFIG_BY_MT[mt]
    return tuple(config) + (1,) * (4 - len(config))


def _bits(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


def make_aux(m: int, device: torch.device, zero_moe_buf: bool = True) -> dict:
    """Output tensors of the BM=16 MoE sort of ``m`` tokens, in AITER's ``register_presorted_moe`` layout (capacity as
    ``aiter.presorted_moe.aux_sort_capacity``); ``moe_buf`` [m, 3584] bf16 is zeroed by the launch when ``zero_moe_buf``."""
    cap = aux_capacity(m)
    i32 = torch.int32
    return {"sorted_token_ids": torch.empty(cap, dtype=i32, device=device),
            "sorted_weights": torch.empty(cap, dtype=torch.float32, device=device),
            "sorted_expert_ids": torch.empty(cap // 16, dtype=i32, device=device),
            "num_valid_ids": torch.empty(2, dtype=i32, device=device),
            "m_indices": torch.empty(cap, dtype=i32, device=device),
            "reverse_sorted": torch.empty(m * TOPK, dtype=i32, device=device),
            "moe_buf": torch.empty(m, 3584, dtype=torch.bfloat16, device=device) if zero_moe_buf else None}


def _launch(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, scale: float, out: tuple | None,
            shuffled: bool, aux: dict | None = None):
    """One launch: the whole router into ``out`` (and the sort into ``aux``), or (``out`` None) the GEMM partials alone;
    -> (slabs, K splits)."""
    m = x.shape[0]
    assert x.ndim == 2 and x.shape[1] == H and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert supports(w) and x.device == w.device and 1 <= m <= MAX_M
    stream = torch.cuda.current_stream()
    mt = m_tile_fragments(m)
    ks, wn, nrw, ms = _config(mt)
    part, counters = _workspace(x.device, stream.cuda_stream, mt, (ks, wn, nrw, ms))
    ptrs = [0] * 8
    cap = zvecs = 0
    if out is None:
        bias_ptr = ow = oid = row_stride = 0
    else:
        weights, ids = out
        assert weights.stride(0) == ids.stride(0) and weights.stride(1) == ids.stride(1) == 1
        bias_ptr, ow, oid, row_stride = bias.data_ptr(), weights.data_ptr(), ids.data_ptr(), weights.stride(0)
        if aux is not None:
            cap = aux_capacity(m)
            for t, n in ((aux["sorted_token_ids"], cap), (aux["sorted_weights"], cap), (aux["sorted_expert_ids"], cap // 16),
                         (aux["num_valid_ids"], 2), (aux["m_indices"], cap), (aux["reverse_sorted"], m * TOPK)):
                assert t.is_contiguous() and t.numel() >= n and t.device == x.device
            mb = aux.get("moe_buf")
            zvecs = 0 if mb is None else m * 3584 * 2 // 16
            ptrs = [_aux_workspace(x.device, stream.cuda_stream, m).data_ptr(), aux["sorted_token_ids"].data_ptr(), aux["sorted_weights"].data_ptr(),
                    aux["sorted_expert_ids"].data_ptr(), aux["num_valid_ids"].data_ptr(), aux["m_indices"].data_ptr(),
                    aux["reverse_sorted"].data_ptr(), 0 if mb is None else mb.data_ptr()]
    _run(_router_kernel(mt, ks, bool(shuffled), out is not None, wn, nrw, aux is not None, ms), x.data_ptr(), w.data_ptr(),
         part.data_ptr(), counters.data_ptr(), bias_ptr, ow, oid, *ptrs, m, row_stride, _bits(scale), cap, zvecs, stream)
    return part, ks


def _top16(part: torch.Tensor, ks: int, bias: torch.Tensor, routed_scaling_factor: float, out: tuple, m: int):
    weights, ids = out
    assert weights.stride(0) == ids.stride(0) and weights.stride(1) == ids.stride(1) == 1
    _run(_topk_kernel(ks), part.data_ptr(), bias.data_ptr(), weights.data_ptr(), ids.data_ptr(), m,
         weights.stride(0), _bits(routed_scaling_factor), torch.cuda.current_stream())


def router_logits(x: torch.Tensor, w: torch.Tensor, *, shuffled: bool = False) -> torch.Tensor:
    """fp32 logits [M, 896] = the partial slabs of the GEMM summed in slice order (what the top-k stage computes); for
    tests and benchmarks, serving never materialises them."""
    m = x.shape[0]
    if m > MAX_M:
        return torch.cat([router_logits(x[i:i + MAX_M], w, shuffled=shuffled) for i in range(0, m, MAX_M)])
    part, ks = _launch(x, w, None, 1.0, None, shuffled)
    mt = m_tile_fragments(m)
    rows = slab_rows(mt, _config(mt)[1], _config(mt)[3])
    slabs = part.view(torch.int32).view(ks, rows, E)[:, :m] & -2  # the fused launch drops each word's lowest bit (parity tag)
    slabs = slabs.view(torch.float32)
    logits = slabs[0].clone()
    for k in range(1, ks):
        logits += slabs[k]
    return logits


def top16(logits: torch.Tensor, bias: torch.Tensor, routed_scaling_factor: float = 1.0,
          out: tuple | None = None):
    """(weights fp32 [M, 16], ids int32 [M, 16]) from fp32 logits [M, 896]: AITER's biased top-16 (sigmoid + bias,
    renormalised, times the scale)."""
    m = logits.shape[0]
    assert logits.dtype == torch.float32 and logits.shape[1] == E and logits.is_contiguous()
    assert bias.dtype == torch.float32 and bias.shape == (E,) and bias.is_contiguous()
    if out is None:
        out = (torch.empty(m, TOPK, dtype=torch.float32, device=logits.device),
               torch.empty(m, TOPK, dtype=torch.int32, device=logits.device))
    if m:
        _top16(logits, 1, bias, routed_scaling_factor, out, m)
    return out


def _router(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, routed_scaling_factor: float = 1.0,
            out: tuple | None = None, *, shuffled: bool = False, aux: dict | None = None):
    """(topk_weights fp32 [M, 16], topk_ids int32 [M, 16]) of the router: x [M, 7168] @ W^T, then AITER's biased top-16
    (sigmoid, + bias, renormalised, times ``routed_scaling_factor``; slots in descending score order). W = w (row-major
    [896, 7168]) or shuffle_weight's layout. Any M (above 192 rows: launches of 192 rows). One launch, deterministic, no
    atomics into outputs, CUDA-graph safe. One call at a time per stream (the workspace)."""
    m = x.shape[0]
    assert bias.dtype == torch.float32 and bias.shape == (E,) and bias.is_contiguous()
    if out is None:
        out = (torch.empty(m, TOPK, dtype=torch.float32, device=x.device),
               torch.empty(m, TOPK, dtype=torch.int32, device=x.device))
    if m > MAX_M:
        assert aux is None, "the sort covers one launch of at most 192 rows"
        for i in range(0, m, MAX_M):
            _router(x[i:i + MAX_M], w, bias, routed_scaling_factor, (out[0][i:i + MAX_M], out[1][i:i + MAX_M]),
                    shuffled=shuffled)
        return out
    if m:
        _launch(x, w, bias, routed_scaling_factor, out, shuffled, aux)
    return out


def _mode(x, w, bias, routed_scaling_factor=1.0, out=None, *, shuffled=False, aux=None):
    """What a call fixes about its kernels besides M (``amd_kernels.warm_up``): the weight layout, the sort."""
    return (bool(shuffled), aux is not None)


def _inputs(mode, n, x, w, bias, routed_scaling_factor=1.0, out=None, *, shuffled=False, aux=None):
    kw = {"shuffled": shuffled}
    if mode[1]:
        kw["aux"] = make_aux(n, x.device)
    return (torch.zeros(n, H, dtype=torch.bfloat16, device=x.device), w, bias, routed_scaling_factor), kw


router_sigmoid_bias_top16_e896_h7168 = entry(
    "moe/router_sigmoid_bias_top16_e896_h7168", mode=_mode, sizes=lambda mode: [16 * mt for mt in range(1, MT_MAX + 1)],
    inputs=_inputs)(_router)
