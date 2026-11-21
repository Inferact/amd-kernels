# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + entry function of the fused Markov draft chain."""

import functools

import torch

from amd_kernels import warm_up

from .reference import RANK, V

SCRATCH_DW = 2048 + 48 * 256 * 4
MAX_ROWS = 48
MAX_STEPS = 8
NAME = "sampling/markov_gumbel_fused_v163840_r512"


def row_tiles(n: int) -> int:
    return (n + 15) // 16


def scratch_dwords(n: int) -> int:
    """Size of the int32 scratch (any row count): generation, per-row results, per-XCC rank counters, per-workgroup
    (value, column, tag) records."""
    return SCRATCH_DW


@functools.cache
def _chain_exe(mt: int, k: int):
    from .kernel import compile_markov_chain

    return compile_markov_chain(mt, k)


def _run_compiled(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _mode(base, *args, **kwargs):
    return base.shape[1]


def _sizes(mode):
    return (16, 32, 48)


def _inputs(mode, n, base, w1, w2, input_ids, anchor_idx, draft_tokens, idx_mapping, sample_pos, temperature, seeds,
            cache, scratch, row_mapping=None):
    dev, k = base.device, mode
    zi = lambda *s: torch.zeros(*s, dtype=torch.int64, device=dev)
    return (torch.zeros(n, k, V, dtype=torch.bfloat16, device=dev), w1, w2, input_ids, zi(n), zi(n, k), zi(n, k),
            zi(n, k), temperature, seeds, cache, torch.zeros(scratch_dwords(n), dtype=torch.int32, device=dev),
            None), {}


@warm_up.entry(NAME, mode=_mode, sizes=_sizes, inputs=_inputs)
def markov_gumbel_fused(base, w1, w2, input_ids, anchor_idx, draft_tokens, idx_mapping, sample_pos, temperature,
                        seeds, cache, scratch, row_mapping=None):
    """The Markov loop of ``DSparkSpeculator._sample_sequential`` after the base logits (``reference.py``): all k steps
    in one persistent launch of 256 workgroups, which must be co-resident (one per CU: nothing else may hold CUs of the
    device while it runs).

    ``base`` bf16 [n, k, V] (the base logits of every draft position); ``w1`` / ``w2`` bf16 [V, 512] contiguous;
    ``input_ids`` int32 and ``anchor_idx`` int64 [n] (step 0's previous token is ``input_ids[anchor_idx[r]]``);
    ``draft_tokens`` int64 [>= n, k] (output; step s reads column s - 1); ``idx_mapping`` and ``sample_pos`` int64
    [n, k] (request state index, < 0: invalid row, and the position of the predicted token: the draw is at
    ``sample_pos - 1``); ``temperature`` fp32 and ``seeds`` int64 [slots]; ``cache`` bf16 [rows, >= k, V] draft
    logits cache (column = step); ``scratch`` int32 [``scratch_dwords(n)``], zeroed once, then owned by the kernel (it
    never needs a reset); ``row_mapping`` int32 [slots] (``DraftLogitsCache.row_mapping``) or None. 1 <= n <= 48,
    1 <= k <= 8. Graph safe.
    """
    n, k, v = base.shape
    assert v == V and 1 <= n <= MAX_ROWS and 1 <= k <= MAX_STEPS and base.dtype == torch.bfloat16 and base.stride(2) == 1
    assert base.stride(1) == V
    assert w1.shape == w2.shape == (V, RANK) and w1.dtype == w2.dtype == torch.bfloat16
    assert w1.is_contiguous() and w2.is_contiguous()
    assert input_ids.dtype == torch.int32 and anchor_idx.dtype == torch.int64 and anchor_idx.shape[0] >= n
    assert idx_mapping.dtype == torch.int64 and idx_mapping.stride(1) == 1 and sample_pos.dtype == torch.int64
    assert sample_pos.stride(1) == 1 and draft_tokens.dtype == torch.int64 and draft_tokens.stride(1) == 1
    assert temperature.dtype == torch.float32 and seeds.dtype == torch.int64
    assert cache.dtype == torch.bfloat16 and cache.stride(2) == 1 and cache.shape[1] >= k
    assert cache.numel() * 2 < (1 << 32) and base.numel() * 2 < (1 << 32)
    assert row_mapping is None or row_mapping.dtype == torch.int32
    assert scratch.dtype == torch.int32 and scratch.numel() >= scratch_dwords(n)
    exe = _chain_exe(row_tiles(n), k)
    _run_compiled(
        exe, base.data_ptr(), base.stride(0), w1.data_ptr(), w2.data_ptr(), input_ids.data_ptr(),
        anchor_idx.data_ptr(), draft_tokens.data_ptr(), draft_tokens.stride(0), idx_mapping.data_ptr(),
        idx_mapping.stride(0), sample_pos.data_ptr(), sample_pos.stride(0), temperature.data_ptr(),
        seeds.data_ptr(), cache.data_ptr(), cache.stride(0), cache.stride(1),
        0 if row_mapping is None else row_mapping.data_ptr(), scratch.data_ptr(), n, (1 << 30) - 1,
        torch.cuda.current_stream(),
    )
