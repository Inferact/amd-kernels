# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry point of the FP4 prefill scorer: grid sizing and the launch (kernel.py, contract in reference.py)."""

import functools

import torch

from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from . import kernel as _kernel

ROWS_PER_CTA = 8
WAVES = 4
TARGET_CTAS = 1024  # four workgroups per CU: the key window is split until the grid fills the GPU
MIN_PAGES_PER_SPLIT = 16


@functools.cache
def _launcher():
    return _kernel.compile_score_rows(ROWS_PER_CTA, WAVES)


def grid_for(rows: int, max_pages: int) -> tuple[int, int]:
    """(row groups, key-window splits) of a call over ``rows`` rows whose widest window spans ``max_pages`` pages."""
    groups = -(-rows // ROWS_PER_CTA)
    splits = max(1, min(TARGET_CTAS // groups, max_pages // MIN_PAGES_PER_SPLIT))
    return groups, splits


def _mode(*args, **kwargs):
    return ()


def _sizes(mode):
    return (ROWS_PER_CTA,)


def _warm_inputs(mode, n, q_payload, q_scale, weights, key_payload, key_scale, ks, ke, out, weight_scale=1.0):
    dev = q_payload.device
    pages = 1
    return (
        (
            torch.zeros(n, 32, 64, dtype=torch.uint8, device=dev),
            torch.zeros(n, 1, 4, 16, 4, dtype=torch.uint8, device=dev),
            torch.zeros(n, 32, dtype=torch.bfloat16, device=dev),
            torch.zeros(pages, 1, 4, 64, 16, dtype=torch.uint8, device=dev),
            torch.zeros(pages, 1, 4, 64, dtype=torch.uint8, device=dev),
            torch.zeros(n, dtype=torch.int32, device=dev),
            torch.full((n,), 64, dtype=torch.int32, device=dev),
            torch.zeros(n, pages * 64, dtype=torch.float32, device=dev),
        ),
        {"weight_scale": weight_scale},
    )


@entry("indexer.mxfp4_prefill_score_d128_h32", mode=_mode, sizes=_sizes, inputs=_warm_inputs)
def score_rows(
    q_payload: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    key_payload: torch.Tensor,
    key_scale: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    out: torch.Tensor,
    weight_scale: float = 1.0,
) -> torch.Tensor:
    """Write the logits of every row's key window ``[ks, ke)`` into ``out`` ([rows, width] fp32); whole pages are stored, so
    positions of the window's pages outside it are unspecified."""
    rows = q_payload.shape[0]
    pages = key_payload.shape[0]
    assert q_payload.shape[1:] == (32, 64) and q_payload.is_contiguous() and q_scale.is_contiguous()
    assert key_payload.shape[1:] == (1, 4, 64, 16) and key_payload.is_contiguous() and key_scale.is_contiguous()
    assert weights.dtype == torch.bfloat16 and weights.is_contiguous() and weights.shape == (rows, 32)
    assert ks.dtype == ke.dtype == torch.int32 and out.dtype == torch.float32 and out.stride(1) == 1
    assert out.shape[0] == rows and out.shape[1] >= pages * 64 or rows == 0
    assert out.numel() * 4 <= _kernel.MAX_OUT_BYTES, "logit tile above 2 GiB"
    if rows == 0:
        return out
    groups, splits = grid_for(rows, pages)
    _run_compiled(
        _launcher(),
        q_payload.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        key_payload.data_ptr(),
        key_scale.data_ptr(),
        ks.data_ptr(),
        ke.data_ptr(),
        out.data_ptr(),
        out.stride(0),
        rows,
        groups,
        splits,
        pages - 1,
        float(weight_scale),
        groups * splits,
        torch.cuda.current_stream(),
    )
    return out
