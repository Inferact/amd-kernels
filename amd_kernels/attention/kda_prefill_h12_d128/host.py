# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry points of the KDA prefill kernels: ``prefill`` (the one-launch path of mega.py for short batches, the staged launches of
prologue.py / scan.py / norm.py otherwise; contract in reference.py) and the stage launches the tests drive."""

import functools

import torch

from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from . import mega as _mega
from . import norm as _norm
from . import outk as _outk
from . import prologue as _prologue
from . import scan as _scan
from . import walk as _walk
from .reference import NH, RECORD_BYTES, D
from .walk import SREC_BYTES, flags_ints

WALK_WAVES = 4
WALK_NVT = 1
CHAIN_WAVES = 4
CHAIN_NVT = 1
CHAIN_STAGES = 2
WALK_STAGES = 2
MIN_CHUNKS = 4
MIN_GROUP_CHUNKS = 5
MAX_GROUPS = 32
NUM_CUS = 256


@functools.cache
def _walk_launcher(ckpt=False, foldwalk=False):
    return _scan.compile_scan(WALK_WAVES, WALK_NVT, False, WALK_STAGES, ckpt, False, foldwalk)


@functools.cache
def _chain_launcher():
    return _scan.compile_scan(CHAIN_WAVES, CHAIN_NVT, True, CHAIN_STAGES)


@functools.cache
def _norm_launcher(w_f32):
    return _norm.compile_norm(4, 2, w_f32)


SHORT_CHAIN = 20  # chunks: a sequence this short (every sequence) is walked whole
MAX_UNITS = 128  # (pair, group) units of one launch: 128 fill the 32 CUs of each of the 8 XCDs (two slice groups per unit)


def plan_groups(chunk_counts):
    """Chunk groups per sequence for sequences of ``chunk_counts`` chunks (host ints): the most groups whose (pair, group) units
    still fit one round of the CUs, none for short chains (the chain launch and the fold cost more than they save)."""
    if max(chunk_counts) <= SHORT_CHAIN:
        return 1
    best = 1
    for g in range(2, MAX_GROUPS + 1):
        units = NH * sum(min(g, max(1, nt // MIN_CHUNKS)) for nt in chunk_counts)
        if units > MAX_UNITS or -(-max(chunk_counts) // g) < MIN_GROUP_CHUNKS:
            break
        best = g
    return best


def chunk_scan(ws, cu_seqlens, chunk_offsets, state_cache, state_indices, has_initial_state, out, chunk_counts=None, groups=None,
         checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None):
    """Chunk scan over the workspace records ``ws`` [H, Nc, RECORD_BYTES]; updates ``state_cache`` rows in place and writes the
    bf16 outputs of the tokens into ``out`` [T, 12, 128] (row stride ``out.stride(0)``). ``chunk_counts``: the chunks of each
    sequence as host ints (the grouping of long chains follows it; evenly split when ``None``); ``groups`` forces the grouping.
    ``checkpoint_offsets`` [N] int32 (tokens into the sequence, a multiple of 64, 0 = off) snapshot the state after that many
    tokens into ``checkpoint_state`` [rows, 12, 128, 128] fp32, row ``checkpoint_rows[n]`` (``n`` when ``None``; negative: off)."""
    n = cu_seqlens.numel() - 1
    nc = ws.shape[1]
    assert ws.dtype == torch.uint8 and ws.shape[0] == NH and ws.shape[2] == RECORD_BYTES and ws.is_contiguous()
    assert state_cache.dtype == torch.float32 and state_cache.shape[1:] == (NH, D, D) and state_cache.stride(1) == D * D
    assert out.dtype == torch.bfloat16 and out.shape[1:] == (NH, D) and out.stride(1) == D and out.stride(2) == 1
    assert chunk_offsets.dtype == torch.int64 and cu_seqlens.dtype == torch.int32 and state_indices.dtype == torch.int32
    has = has_initial_state.to(torch.uint8) if has_initial_state.dtype != torch.uint8 else has_initial_state
    if groups is None:
        groups = plan_groups(chunk_counts if chunk_counts is not None else [-(-nc // n)] * n)
    g = groups
    dev = ws.device
    if g > 1:
        gb = torch.empty(n * NH, g, D, D, dtype=torch.float32, device=dev)
        gc = torch.empty(n * NH, g, _scan.FRAGS_C, dtype=torch.uint8, device=dev)
        ga = torch.empty(n * NH, g, D, dtype=torch.float32, device=dev)
        sin = torch.empty(n * NH, g, D, D, dtype=torch.float32, device=dev)
    else:
        gb = gc = ga = sin = ws
    stream = torch.cuda.current_stream()
    common = (ws.data_ptr(), nc, state_cache.data_ptr(), state_cache.stride(0) * 4, state_indices.data_ptr(), has.data_ptr(),
              cu_seqlens.data_ptr(), chunk_offsets.data_ptr(), out.data_ptr(), out.stride(0))
    ckpt = checkpoint_state is not None
    if ckpt:
        assert checkpoint_state.dtype == torch.float32 and checkpoint_state.shape[1:] == (NH, D, D) and checkpoint_offsets.dtype == torch.int32
        rows = checkpoint_rows if checkpoint_rows is not None else torch.arange(n, dtype=torch.int32, device=dev)
        ck = (checkpoint_state.data_ptr(), checkpoint_state.stride(0) * 4, checkpoint_offsets.data_ptr(), rows.data_ptr())
    else:
        ck = (ws.data_ptr(), 0, ws.data_ptr(), ws.data_ptr())
    nbfold = ((n * NH + 7) // 8) * 8 * 8 if g > 1 else 0
    tail = (g, MIN_CHUNKS, n * NH, *ck)
    full = (*common, gb.data_ptr(), gc.data_ptr(), ga.data_ptr(), sin.data_ptr(), *tail)
    if g > 1:
        _run_compiled(_chain_launcher(), *full, 0, stream)
    _run_compiled(_walk_launcher(ckpt, nbfold > 0), *full, nbfold, stream)
    return out


def gated_norm(o, gate, norm_weight, eps, out=None):
    """``out = bf16(o * rsqrt(mean(o^2) + eps) * norm_weight * sigmoid(gate))`` per 128-wide row; ``out`` may be ``o``."""
    t = o.shape[0]
    out = o if out is None else out
    assert o.is_contiguous() and out.is_contiguous() and gate.shape == o.shape and gate.stride(2) == 1 and gate.stride(1) == D
    wf32 = norm_weight.dtype == torch.float32
    assert wf32 or norm_weight.dtype == torch.bfloat16
    span = (t - 1) * gate.stride(0) + NH * D
    _run_compiled(_norm_launcher(wf32), o.data_ptr(), gate.data_ptr(), norm_weight.data_ptr(), out.data_ptr(), gate.stride(0),
                  t * NH, span * 2, float(eps), torch.cuda.current_stream())
    return out


MAX_WALK_WGS = 192  # a multiple of 32 and 64: a workgroup keeps its XCD and its slice group across the units it walks
_flag_cache = {}


def _flags(device, chunks):
    """Zeroed hand-over flags of the mega launch (every flag is reset by its consumer, so the buffer is zero again after a call)."""
    need = flags_ints(chunks)
    buf = _flag_cache.get(device)
    if buf is None or buf.numel() < need:
        buf = torch.zeros(max(need, 1 << 21), dtype=torch.int32, device=device)
        _flag_cache[device] = buf
    return buf


@functools.cache
def _mega_launcher(w_f32, ckpt, ts=False):
    return _mega.compile_mega(w_f32, ckpt, ts)


def prefill_mega(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices, has_initial_state,
                 cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound, scale=D ** -0.5, table=None,
                 chunk_counts=None, checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None, walk_wgs=None, timestamps=None):
    """The KDA prefill of one layer in one launch (mega.py): same contract as ``prefill``."""
    n = cu_seqlens.numel() - 1
    nc = chunk_indices.shape[0]
    dev = mixed.device
    assert mixed.dtype == raw_g.dtype == beta_logit.dtype == torch.bfloat16 and mixed.stride(-1) == raw_g.stride(-1) == beta_logit.stride(-1) == 1
    assert mixed.stride(0) % 8 == 0 and conv_state.stride(0) < 2**31 // 2
    assert state_cache.dtype == torch.float32 and state_cache.shape[1:] == (NH, D, D) and state_cache.stride(1) == D * D
    assert out.dtype == torch.bfloat16 and out.shape[1:] == (NH, D) and out.stride(1) == D and out.stride(2) == 1
    assert gate.shape[1:] == (NH, D) and gate.stride(2) == 1 and gate.stride(1) == D
    assert chunk_offsets.dtype == torch.int64 and cu_seqlens.dtype == torch.int32 and state_indices.dtype == torch.int32
    wf32 = norm_weight.dtype == torch.float32
    assert wf32 or norm_weight.dtype == torch.bfloat16
    has = has_initial_state.to(torch.uint8) if has_initial_state.dtype != torch.uint8 else has_initial_state
    if table is None:
        table = _prologue.chunk_table(cu_seqlens, chunk_indices, state_indices, has_initial_state)
    ws = torch.empty(NH, nc, RECORD_BYTES, dtype=torch.uint8, device=dev)
    sb = torch.empty(NH, nc, SREC_BYTES, dtype=torch.uint8, device=dev)
    flags = _flags(dev, nc)
    ckpt = checkpoint_state is not None
    if ckpt:
        assert checkpoint_state.dtype == torch.float32 and checkpoint_state.shape[1:] == (NH, D, D) and checkpoint_offsets.dtype == torch.int32
        rows = checkpoint_rows if checkpoint_rows is not None else torch.arange(n, dtype=torch.int32, device=dev)
        ck = (checkpoint_state.data_ptr(), checkpoint_state.stride(0) * 4, checkpoint_offsets.data_ptr(), rows.data_ptr())
    else:
        ck = (ws.data_ptr(), 0, ws.data_ptr(), ws.data_ptr())
    units = ((n * NH + 7) // 8) * _walk.UNIT
    nwg = min(units, MAX_WALK_WGS if walk_wgs is None else walk_wgs)
    cs = conv_state
    _run_compiled(
        _mega_launcher(wf32, ckpt, timestamps is not None),
        mixed.data_ptr(), mixed.stride(0), mixed.shape[0] * mixed.stride(0) * 2, raw_g.data_ptr(),
        raw_g.stride(-3) if raw_g.dim() > 2 else raw_g.stride(0), beta_logit.data_ptr(), beta_logit.stride(-2), A_log.data_ptr(),
        dt_bias.data_ptr(), conv_weight.data_ptr(), cs.data_ptr(), cs.stride(0), cs.stride(1), cs.stride(2), table.data_ptr(), ws.data_ptr(), nc,
        float(lower_bound), float(scale), state_cache.data_ptr(), state_cache.stride(0) * 4, state_indices.data_ptr(), has.data_ptr(),
        cu_seqlens.data_ptr(), chunk_offsets.data_ptr(), sb.data_ptr(), n * NH, *ck, out.data_ptr(), out.stride(0), gate.data_ptr(),
        gate.stride(0), norm_weight.data_ptr(), float(norm_eps), flags.data_ptr(), nwg,
        timestamps.data_ptr() if timestamps is not None else ws.data_ptr(), torch.cuda.current_stream())
    return out


@functools.cache
def _walk_standalone(ckpt, ts=False):
    return _walk.compile_walk(ckpt, ts)


@functools.cache
def _out_standalone(w_f32):
    return _outk.compile_out(w_f32)


def walk_states(ws, cu_seqlens, chunk_offsets, state_cache, state_indices, has_initial_state, checkpoint_state=None, checkpoint_offsets=None,
                checkpoint_rows=None, timestamps=None):
    """The state walk alone over the workspace records ``ws``: the state records [H, Nc, SREC_BYTES] the output pass reads; updates
    ``state_cache`` (and the checkpoint rows) in place."""
    n = cu_seqlens.numel() - 1
    nc = ws.shape[1]
    dev = ws.device
    has = has_initial_state.to(torch.uint8) if has_initial_state.dtype != torch.uint8 else has_initial_state
    sb = torch.empty(NH, nc, SREC_BYTES, dtype=torch.uint8, device=dev)
    ckpt = checkpoint_state is not None
    if ckpt:
        rows = checkpoint_rows if checkpoint_rows is not None else torch.arange(n, dtype=torch.int32, device=dev)
        ck = (checkpoint_state.data_ptr(), checkpoint_state.stride(0) * 4, checkpoint_offsets.data_ptr(), rows.data_ptr())
    else:
        ck = (ws.data_ptr(), 0, ws.data_ptr(), ws.data_ptr())
    _run_compiled(_walk_standalone(ckpt, timestamps is not None), ws.data_ptr(), nc, state_cache.data_ptr(), state_cache.stride(0) * 4, state_indices.data_ptr(),
                  has.data_ptr(), cu_seqlens.data_ptr(), chunk_offsets.data_ptr(), sb.data_ptr(), n * NH, *ck, timestamps.data_ptr() if timestamps is not None else ws.data_ptr(), torch.cuda.current_stream())
    return sb


def chunk_outputs(ws, sb, table, gate, norm_weight, norm_eps, out):
    """The output pass alone: gated-normed outputs of every chunk from the workspace and state records."""
    nc = ws.shape[1]
    wf32 = norm_weight.dtype == torch.float32
    _run_compiled(_out_standalone(wf32), ws.data_ptr(), nc, sb.data_ptr(), table.data_ptr(), out.data_ptr(), out.stride(0), gate.data_ptr(),
                  gate.stride(0), norm_weight.data_ptr(), float(norm_eps), torch.cuda.current_stream())
    return out


def prefill_v1(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices, has_initial_state,
             cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound, scale=D ** -0.5, table=None,
             chunk_counts=None, checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None):
    """The KDA prefill of one layer: conv + prologue, chunk scan, gated RMSNorm. Writes ``out`` [T, 12, 128] bf16, rolls
    ``conv_state`` and updates ``state_cache`` rows in place. ``table``: ``prologue.chunk_table`` of the step (built once per step
    and shared by the layers; built here when ``None``); ``chunk_counts``: the chunks of each sequence as host ints (sizes the
    groups of long chains); ``checkpoint_*``: see ``chunk_scan``."""
    ws = _prologue.prologue(mixed, raw_g, beta_logit, A_log, dt_bias, lower_bound, scale, conv_weight, conv_state, state_indices,
                            has_initial_state, cu_seqlens, chunk_indices, table=table)
    chunk_scan(ws, cu_seqlens, chunk_offsets, state_cache, state_indices, has_initial_state, out, chunk_counts,
         checkpoint_state=checkpoint_state, checkpoint_offsets=checkpoint_offsets, checkpoint_rows=checkpoint_rows)
    return gated_norm(out, gate, norm_weight, norm_eps, out)


MEGA_MAX_CHAIN = 60  # chunks of the longest sequence the one-launch path serves (its walk is serial: 0.7-0.9 us per chunk)
MEGA_MAX_CHUNKS = 60  # chunks of a batch of up to three sequences; larger batches start too many walk workgroups beside the prologue
MEGA_TINY_CHUNKS = 16  # a batch this small takes the one-launch path whatever its sequence count (up to 8)


def uses_mega(chunk_counts):
    """Whether sequences of ``chunk_counts`` chunks (host ints) take the one-launch path (``prefill_mega``) rather than the staged
    one (``prefill_v1``: prologue, grouped chains, gated norm); the staged path when the counts are not known."""
    if chunk_counts is None:
        return False
    n, nc = len(chunk_counts), sum(chunk_counts)
    return max(chunk_counts) <= MEGA_MAX_CHAIN and (nc <= MEGA_TINY_CHUNKS and n <= 8 or n <= 3 and nc <= MEGA_MAX_CHUNKS)


def _prefill(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices, has_initial_state,
             cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound, scale=D ** -0.5, table=None,
             chunk_counts=None, checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None, impl=None):
    """The KDA prefill of one layer (reference.py): writes ``out`` [T, 12, 128] bf16, rolls ``conv_state`` and updates
    ``state_cache`` rows in place. ``impl``: ``"mega"`` / ``"v1"`` force a path, by default ``uses_mega(chunk_counts)`` chooses."""
    if impl is None:
        impl = "mega" if uses_mega(chunk_counts) else "v1"
    run = prefill_mega if impl == "mega" else prefill_v1
    return run(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices, has_initial_state,
               cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound, scale, table, chunk_counts,
               checkpoint_state, checkpoint_offsets, checkpoint_rows)


def _mode(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices, has_initial_state,
          cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound, scale=D ** -0.5, table=None,
          chunk_counts=None, checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None, impl=None):
    """What a call fixes about its kernels (``amd_kernels.warm_up``): the norm weight dtype and the checkpoint export."""
    return (norm_weight.dtype == torch.float32, checkpoint_state is not None)


def _inputs(mode, n, mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices,
            has_initial_state, cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound, scale=D ** -0.5,
            table=None, chunk_counts=None, checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None, impl=None):
    """One sequence of ``n`` tokens (zeros; the launches are compiled, not run): the staged path with a short chain at 64 tokens
    (whole chains), the one-launch path at 2048, the staged path with grouped chains at 8192."""
    dev = mixed.device
    chunks = -(-n // 64)
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    ci = torch.tensor([[0, k] for k in range(chunks)], dtype=torch.int32, device=dev)
    co = torch.tensor([0, chunks], dtype=torch.int64, device=dev)
    zero = torch.zeros(1, dtype=torch.int32, device=dev)
    rows = mixed.shape[1] if mixed.dim() == 2 else 0
    x = torch.zeros((n, max(rows, 4608)), dtype=mixed.dtype, device=dev)[:, :4608]
    xg = torch.zeros((n, 6288), dtype=gate.dtype, device=dev)
    g = xg[:, 4608:6144].view(n, NH, D)
    args = (x, torch.zeros((n, NH, D), dtype=raw_g.dtype, device=dev), xg[:, 6272:6284], g, A_log, dt_bias, conv_weight, conv_state,
            state_cache, zero, torch.zeros(1, dtype=torch.bool, device=dev), cu, ci, co, norm_weight, norm_eps,
            torch.empty((n, NH, D), dtype=out.dtype, device=dev), lower_bound)
    kwargs = {"chunk_counts": [chunks], "impl": "v1" if n == 64 else None}
    if mode[1]:
        kwargs.update(checkpoint_state=checkpoint_state, checkpoint_offsets=zero, checkpoint_rows=None)
    return args, kwargs


prefill = entry("attention/kda_prefill_h12_d128", mode=_mode, sizes=lambda mode: (64, 2048, 8192), inputs=_inputs)(_prefill)
