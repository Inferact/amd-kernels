# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Workspace and entry of the two-launch rejection sampler."""

import functools

import torch

from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from . import kernel as _kernel
from .reference import NSPEC, ROWS, V


@functools.cache
def _workspace(device: torch.device) -> torch.Tensor:
    """Stats, block partials and ticket counters, one set per device (launches on one stream); zeroed once, every
    launch leaves its counters at zero."""
    return torch.zeros(_kernel.WORKSPACE_BYTES, dtype=torch.uint8, device=device)


@functools.cache
def _rates_placeholder(device: torch.device) -> torch.Tensor:
    return torch.zeros(NSPEC, dtype=torch.float32, device=device)


@functools.cache
def _stats_exe(target_bf16: bool):
    return _kernel.compile_rejection_stats(target_bf16)


@functools.cache
def _resample_exe(synthetic: bool, target_bf16: bool):
    return _kernel.compile_rejection_resample(synthetic, target_bf16)


def _rejection_sample(target_logits, draft_logits, draft_sampled, pos, idx_mapping, temperature, seed,
                      synthetic_rates=None):
    b = idx_mapping.shape[0]
    dev = target_logits.device
    assert target_logits.dtype in (torch.float32, torch.bfloat16) and target_logits.shape == (b * ROWS, V)
    target_bf16 = target_logits.dtype == torch.bfloat16
    assert target_logits.is_contiguous()
    assert draft_logits.dtype == torch.bfloat16 and draft_logits.dim() == 3 and draft_logits.shape[1:] == (NSPEC, V)
    assert draft_logits.is_contiguous() and draft_sampled.dtype == torch.int64 and pos.dtype == torch.int64
    assert idx_mapping.dtype == torch.int32 and temperature.dtype == torch.float32 and seed.dtype == torch.int64
    assert 1 <= b <= _kernel.MAX_REQS
    ws = _workspace(dev)
    sampled = torch.empty(b, ROWS, dtype=torch.int64, device=dev)
    num_sampled = torch.empty(b, dtype=torch.int32, device=dev)
    synthetic = synthetic_rates is not None
    rates = synthetic_rates if synthetic else _rates_placeholder(dev)
    stream = torch.cuda.current_stream()
    _run_compiled(_stats_exe(target_bf16), target_logits.data_ptr(), draft_logits.data_ptr(), idx_mapping.data_ptr(),
                  temperature.data_ptr(), ws.data_ptr(), b * NSPEC, stream)
    _run_compiled(_resample_exe(synthetic, target_bf16), target_logits.data_ptr(), draft_logits.data_ptr(),
                  draft_sampled.data_ptr(), pos.data_ptr(), idx_mapping.data_ptr(), temperature.data_ptr(),
                  seed.data_ptr(), rates.data_ptr(), ws.data_ptr(), sampled.data_ptr(), num_sampled.data_ptr(),
                  b, _kernel.NB * 8 * ((b + 7) // 8), stream)
    return sampled, num_sampled


def _mode(target_logits, draft_logits, draft_sampled, pos, idx_mapping, temperature, seed, synthetic_rates=None):
    """The variant follows the target logits' dtype and whether the acceptance is synthetic."""
    return (synthetic_rates is not None, target_logits.dtype == torch.bfloat16)


def _sizes(mode):
    return (1,)


def _inputs(mode, n, target_logits, draft_logits, draft_sampled, pos, idx_mapping, temperature, seed,
            synthetic_rates=None):
    """One request of zeros (compile-only: nothing launches)."""
    dev = target_logits.device
    z = torch.zeros
    rates = z(NSPEC, dtype=torch.float32, device=dev) if mode[0] else None
    return (z(ROWS, V, dtype=torch.bfloat16 if mode[1] else torch.float32, device=dev), z(1, NSPEC, V, dtype=torch.bfloat16, device=dev),
            z(ROWS, dtype=torch.int64, device=dev), z(ROWS, dtype=torch.int64, device=dev),
            z(1, dtype=torch.int32, device=dev), z(1, dtype=torch.float32, device=dev),
            z(1, dtype=torch.int64, device=dev), rates), {}


rejection_sample = entry("sampling/rejection_standard_v129280_s5", mode=_mode, sizes=_sizes,
                         inputs=_inputs)(_rejection_sample)
rejection_sample.__doc__ = """Standard rejection sampling of 5 draft tokens per request (``reference.py``): ``target_logits``
[B * 6, V], fp32 with the temperature applied or bf16 raw (the lm_head GEMM's output: the kernel divides by the
request's temperature, so no conversion launch sits between them), bf16 ``draft_logits`` [R, 5, V] (the Markov
sampler's cache, pre-temperature), int64
``draft_sampled`` and ``pos`` [B * 6], int32 ``idx_mapping`` [B] (request state, >= 0), fp32 ``temperature`` [R],
int64 ``seed`` [R]; ``synthetic_rates`` fp32 [5] replaces the probability-ratio test by ``u < rate``. Returns int64
``sampled`` [B, 6] (-1 after the resampled token) and int32 ``num_sampled`` [B]. One workspace per device: launches
on one stream. CUDA-graph safe."""


def warmup_rejection_sample(device: torch.device) -> None:
    """Compile every mode (fp32 or bf16 target logits, ratio test or synthetic) and allocate the workspace, before
    graph capture; nothing compiles after it."""
    z = torch.zeros
    for dtype in (torch.float32, torch.bfloat16):
        args = (z(ROWS, V, dtype=dtype, device=device), z(1, NSPEC, V, dtype=torch.bfloat16, device=device),
                z(ROWS, dtype=torch.int64, device=device), z(ROWS, dtype=torch.int64, device=device),
                z(1, dtype=torch.int32, device=device), z(1, dtype=torch.float32, device=device),
                z(1, dtype=torch.int64, device=device))
        rejection_sample(*args)
        rejection_sample(*args, z(NSPEC, dtype=torch.float32, device=device))
    torch.cuda.synchronize(device)
