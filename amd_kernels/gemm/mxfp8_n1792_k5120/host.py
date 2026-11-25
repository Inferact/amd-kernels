# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entries of the N 1792 x K 5120 MXFP8 GEMM: weight preshuffle, row dispatch, warm-up."""

import torch

from amd_kernels.gemm.mxfp8_common import gemm as _common
from amd_kernels.gemm.mxfp8_decode_prequant import gemm as _decode
from amd_kernels.gemm.mxfp8_decode_prequant import gemm_swiglu_mxfp8 as _decode_swiglu
from amd_kernels.warm_up import entry

from . import kernel as _kernel

N, K = _kernel.N, _kernel.K
KEY = "mxfp8_n1792_k5120"
LIMIT = 10.0


def preshuffle_weight(weight: torch.Tensor, weight_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 codes [N, K] and E8M0 scales, [N, K / 32] or the checkpoint's 32 x 32 blocks [N / 32, K / 32] (expanded
    to one row each), -> the layouts the entries read (load time)."""
    assert tuple(weight.shape) == (N, K) and weight.dtype == torch.float8_e4m3fn and weight_scale.dtype == torch.uint8
    if tuple(weight_scale.shape) == (N // 32, K // 32):
        weight_scale = weight_scale.repeat_interleave(32, dim=0)
    assert tuple(weight_scale.shape) == (N, K // 32)
    return _common.preshuffle_mxfp8_weight(weight, weight_scale)


def serves(m: int, table=None) -> bool:
    """Whether the entry beat the served path at ``m`` rows (``kernel.py``); callers fall back where it did not."""
    r = _kernel.row(_kernel.LINEAR if table is None else table, m)
    return r is not None and r[2]


def _run_rows(table, x, weight, weight_scale):
    m = x.shape[0]
    r = _kernel.row(table, m)
    if r is None:
        return _common.mxfp8_linear(x, weight, weight_scale, torch.bfloat16)
    kind, _, _, cfg = r
    if kind == "fq":
        return _decode(x, None, weight, weight_scale, cfg, key=KEY)
    if kind == "shuf":
        codes, scale = _common.mxfp8_e4m3_quantize_shuffled(x, 16 * cfg.rt)
        return _decode(codes, scale, weight, weight_scale, cfg, layout="shuffled", rows=m, key=KEY)
    return _common.mxfp8_linear(x, weight, weight_scale, torch.bfloat16, config=_kernel.common_config(cfg))


def _mxfp8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.shape[1] == K and x.dtype == torch.bfloat16 and x.is_contiguous()
    return _run_rows(_kernel.LINEAR, x, weight, weight_scale)


def _mode(x, weight, weight_scale):
    return ()


def _sizes(mode):
    return tuple(row[1] for row in _kernel.LINEAR)


def _inputs(mode, n, x, weight, weight_scale):
    return (torch.zeros(n, K, dtype=torch.bfloat16, device=x.device), weight, weight_scale), {}


mxfp8_linear = entry("gemm/mxfp8_n1792_k5120", mode=_mode, sizes=_sizes, inputs=_inputs)(_mxfp8_linear)
mxfp8_linear.__doc__ = """``x @ W^T`` for a bf16 activation ``x`` [M, 5120] and a preshuffled MXFP8 weight (``preshuffle_weight``):
quantized per 1 x 32 block as ``reference.py`` says, bf16 [M, 1792] out. A fresh output per call; CUDA-graph safe
(static launch arguments, no host sync); deterministic. Rows above ``kernel.LINEAR``'s last run the common
dispatcher. Rows whose split-K sum runs in the last workgroup of a tile share one counter arena per device: one
launch at a time."""


def warmup_mxfp8_linear(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    """Compile every row's variants and create the counter arena before graph capture (a first call inside a
    capture does not warm up)."""
    dev = weight.device
    for m in _sizes(()):
        mxfp8_linear(torch.zeros(m, K, dtype=torch.bfloat16, device=dev), weight, weight_scale)
    torch.cuda.synchronize(dev)


# The folder's protocol with ``gemm.mxfp8_common`` (``gemm._ROUTES``): its common-kernel rows, and its own
# entries for the row counts its table covers.
ROWS = {
    (N, K, "bf16"): tuple(
        (rows, _kernel.common_config(cfg)) for kind, rows, _, cfg in _kernel.LINEAR if kind == "common"
    )
}
LAUNCHERS = {}


def select(m: int, out_dtype: torch.dtype, allow_decode: bool):
    return _common.select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)


def linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, out_dtype: torch.dtype):
    """``mxfp8_linear``'s call where ``kernel.LINEAR`` has a row, else None (the common dispatcher)."""
    if out_dtype != torch.bfloat16 or not x.is_contiguous() or _kernel.row(_kernel.LINEAR, x.shape[0]) is None:
        return None
    return mxfp8_linear(x, weight, weight_scale)


def covers(m: int, prequant: bool = False) -> bool:
    """Whether the folder's row for ``m`` bf16 rows beat the served path (no row-major-codes rows)."""
    return not prequant and serves(m)


def warmup(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    warmup_mxfp8_linear(weight, weight_scale)
