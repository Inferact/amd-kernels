# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Workspace, row-count dispatch and entry of the fused Markov draft chain."""

import functools

import torch

from amd_kernels.warm_up import entry

from . import kernel as _kernel
from .reference import NSPEC, R, V

MAX_ROWS = 160  # rows of one launch; more rows run as consecutive launches
NRF_VARIANTS = (1, 2, 3, 4, 6, 8, 10)  # 16-row fragments of a launch's buffers; the row count rounds up to one
_POS_ADD = (1 << 30) - 1  # the draft-noise salt and the draw position sample_pos - 1


def nrf_for(n: int) -> int:
    """The row-fragment variant (of 16 rows) that serves ``n`` <= 160 rows."""
    return next(k for k in NRF_VARIANTS if 16 * k >= n)


@functools.cache
def _chain_exe(nrf: int):
    return _kernel.compile_markov_chain(nrf)


@functools.cache
def _workspace(device: torch.device, nrf: int):
    """Tile winners and the epoch of one variant: every launch rewrites all of its winners and bumps its epoch, so a
    tag only ever meets the entries of the launch before (zeros once; no reset)."""
    winners = torch.zeros(NSPEC * _kernel.NT * 16 * nrf * 2, dtype=torch.int32, device=device)
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    return winners, epoch


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _chain(base, w1, w2, input_ids, anchor_idx, idx_map, sample_pos, temperature, seeds, cache, tokens):
    """The 5 sequential draft steps of ``reference.py`` for every row (request) of the batch, one launch per 160 rows.

    ``base`` bf16 [n * 5, V] (row r * 5 + s: request r, step s; unit column stride, any row stride), ``w1`` / ``w2``
    bf16 contiguous [V, 256] (the Markov embedding and bias head), ``input_ids`` int32 / int64 and ``anchor_idx`` int64
    [n] (step 0's previous token is ``input_ids[anchor_idx[r]]``), ``idx_map`` int32 / int64 [n, 5] and ``sample_pos``
    int64 [n, 5] (unit column stride), ``temperature`` fp32 / ``seeds`` int64 [slots], ``cache`` bf16 [slots, 5, V]
    contiguous (< 2 GiB; rows of the replicated draft-logits cache, written where ``idx_map`` >= 0), ``tokens`` int64
    [>= n, 5] (unit column stride): the draft tokens. Deterministic, graph safe, no host sync. One workspace per
    device and row-count variant: launches on one stream. Needs the GPU's 253 CUs free: the workgroups wait for each
    other.
    """
    n = anchor_idx.shape[0]
    assert base.dtype == torch.bfloat16 and base.shape == (n * NSPEC, V) and base.stride(1) == 1
    assert base.stride(0) % 2 == 0 and base.data_ptr() % 8 == 0
    for w in (w1, w2):
        assert w.dtype == torch.bfloat16 and w.shape == (V, R) and w.is_contiguous()
    assert input_ids.dtype in (torch.int32, torch.int64) and input_ids.is_contiguous()
    assert anchor_idx.dtype == torch.int64 and anchor_idx.is_contiguous()
    assert idx_map.dtype in (torch.int32, torch.int64) and idx_map.shape == (n, NSPEC) and idx_map.stride(1) == 1
    assert sample_pos.dtype == torch.int64 and sample_pos.shape == (n, NSPEC) and sample_pos.stride(1) == 1
    assert temperature.dtype == torch.float32 and temperature.is_contiguous()
    assert seeds.dtype == torch.int64 and seeds.is_contiguous() and seeds.shape == temperature.shape
    assert cache.dtype == torch.bfloat16 and cache.is_contiguous() and cache.shape[1:] == (NSPEC, V)
    assert cache.numel() * 2 < 1 << 31 and base.shape[0] * base.stride(0) * 2 < 1 << 31
    assert tokens.dtype == torch.int64 and tokens.shape[0] >= n and tokens.shape[1] == NSPEC
    assert tokens.stride(1) == 1
    stream = torch.cuda.current_stream()
    ide = input_ids.element_size() // 4
    idxe = idx_map.element_size() // 4
    for r0 in range(0, n, MAX_ROWS):
        m = min(MAX_ROWS, n - r0)
        winners, epoch = _workspace(base.device, nrf_for(m))
        _run_compiled(
            _chain_exe(nrf_for(m)), w2.data_ptr(), w1.data_ptr(), input_ids.data_ptr(), ide,
            anchor_idx.data_ptr() + r0 * 8, tokens.data_ptr() + r0 * tokens.stride(0) * 8, tokens.stride(0),
            base.data_ptr() + r0 * NSPEC * base.stride(0) * 2, base.stride(0), cache.data_ptr(),
            idx_map.data_ptr() + r0 * idx_map.stride(0) * idx_map.element_size(), idx_map.stride(0) * idxe, idxe,
            sample_pos.data_ptr() + r0 * sample_pos.stride(0) * 8, sample_pos.stride(0), temperature.data_ptr(),
            seeds.data_ptr(), winners.data_ptr(), epoch.data_ptr(), m, _POS_ADD, stream)


def _mode(*args):
    """One mode: the variants follow the row count (``amd_kernels.warm_up``)."""
    return ()


def _sizes(mode):
    return tuple(16 * k for k in NRF_VARIANTS)


def _inputs(mode, n, base, w1, w2, input_ids, anchor_idx, idx_map, sample_pos, temperature, seeds, cache, tokens):
    """``n`` requests of inert rows (slot -1) on the caller's weights, ids and buffers."""
    dev = base.device
    args = (torch.zeros(n * NSPEC, V, dtype=torch.bfloat16, device=dev), w1, w2, input_ids,
            torch.zeros(n, dtype=torch.int64, device=dev), torch.full((n, NSPEC), -1, dtype=idx_map.dtype, device=dev),
            torch.zeros(n, NSPEC, dtype=torch.int64, device=dev), temperature, seeds, cache,
            torch.zeros(n, NSPEC, dtype=torch.int64, device=dev))
    return args, {}


markov_chain = entry("sampling/markov_chain_gumbel_bf16_v129280_r256_s5", mode=_mode, sizes=_sizes,
                     inputs=_inputs)(_chain)


def warmup_markov_chain(device: torch.device, idx_dtype: torch.dtype = torch.int64,
                        ids_dtype: torch.dtype = torch.int32) -> None:
    """Compile every row-fragment variant and allocate the workspaces, before graph capture."""
    w = torch.zeros(V, R, dtype=torch.bfloat16, device=device)
    ids = torch.zeros(1, dtype=ids_dtype, device=device)
    cache = torch.zeros(1, NSPEC, V, dtype=torch.bfloat16, device=device)
    temp = torch.zeros(1, dtype=torch.float32, device=device)
    seeds = torch.zeros(1, dtype=torch.int64, device=device)
    for n in _sizes(()):
        markov_chain(torch.zeros(n * NSPEC, V, dtype=torch.bfloat16, device=device), w, w, ids,
                     torch.zeros(n, dtype=torch.int64, device=device),
                     torch.full((n, NSPEC), -1, dtype=idx_dtype, device=device),
                     torch.zeros(n, NSPEC, dtype=torch.int64, device=device), temp, seeds, cache,
                     torch.zeros(n, NSPEC, dtype=torch.int64, device=device))
    torch.cuda.synchronize(device)
