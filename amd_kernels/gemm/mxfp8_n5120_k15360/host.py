# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Weight preparation, row-count dispatch and entry of the draft's input projection (N 5120, K 15360)."""

import functools
from dataclasses import dataclass

import torch

from amd_kernels.gemm.mxfp8_common import gemm as _gemm
from amd_kernels.gemm.mxfp8_common.gemm import _Config
from amd_kernels.warm_up import entry

from .reference import K, N, expand_block_scales

MAX_ROWS = 32768
_STAGED = "staged_ab"
# row upper bound -> config (graph-replay sweeps over staged_ab, aiter, decode and 4wave)
DISPATCH = (
    (32, _Config(64, 128, 128, 0, 0, 6, _STAGED, 2, 2)),
    (64, _Config(64, 128, 128, 0, 0, 6, _STAGED, 2, 2)),
    (128, _Config(128, 128, 128, 0, 0, 6, _STAGED, 2, 2)),
    (256, _Config(256, 128, 128, 0, 0, 6, _STAGED, 2, 4)),
    (384, _Config(128, 256, 128, 0, 0, 4, _STAGED, 2, 4)),
    (512, _Config(128, 256, 128, 0, 0, 3, _STAGED, 2, 4)),
    (768, _Config(128, 256, 128, 0, 0, 2, _STAGED, 2, 4)),
    (1024, _Config(256, 256, 128, 0, 0, 3, _STAGED, 2, 4)),
    (MAX_ROWS, _Config(256, 256, 128, 0, 0, 1, _gemm._FOURWAVE_KERNEL, 2, 2)),
)
WARM_UP_ROWS = tuple(bound for bound, _ in DISPATCH[:-1]) + (4096,)


def config_for(rows: int) -> _Config:
    return next(cfg for bound, cfg in DISPATCH if rows <= bound)


@dataclass
class MainProjWeights:
    """The projection's weight in the GEMM's layout: preshuffled E4M3 codes ``[5120, 15360]`` and per-row E8M0 scales."""

    weight: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def from_block_scales(cls, weight: torch.Tensor, block_scale: torch.Tensor) -> "MainProjWeights":
        """From the checkpoint's E4M3 ``weight`` ``[5120, 15360]`` and its 32 x 32 block scales ``[160, 480]``; once,
        before graph capture."""
        assert weight.shape == (N, K) and block_scale.shape == (N // 32, K // 32)
        scale = expand_block_scales(block_scale.view(torch.uint8))
        shuffled, shuffled_scale = _gemm.preshuffle_mxfp8_weight(weight.view(torch.float8_e4m3fn), scale)
        return cls(shuffled, shuffled_scale)


def _main_proj(x: torch.Tensor, w: MainProjWeights) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == K and x.is_contiguous()
    assert 1 <= x.shape[0] <= MAX_ROWS
    return _gemm.mxfp8_linear(x, w.weight, w.scale, torch.bfloat16, config=config_for(x.shape[0]))


def _mode(x, w):
    return ()


def _sizes(mode):
    return WARM_UP_ROWS


def _inputs(mode, rows, x, w):
    return (torch.zeros((rows, K), dtype=x.dtype, device=x.device), w), {}


main_proj = entry("gemm/mxfp8_n5120_k15360", mode=_mode, sizes=_sizes, inputs=_inputs)(_main_proj)
main_proj.__doc__ = """``x @ W^T``: bf16 ``x`` [M, 15360] (1-32768 rows) quantized per 1 x 32 block, MXFP8 weight
(``MainProjWeights``), bf16 [M, 5120] (``reference.py``); a split-K sum is in fixed order, so two runs match bit for
bit. CUDA-graph safe. An ``amd_kernels.warm_up`` entry: the first call compiles every row tier."""


def warmup_main_proj(w: MainProjWeights) -> None:
    """Compile every row tier and the quantizers, before graph capture; nothing compiles after it."""
    for rows in WARM_UP_ROWS:
        main_proj(torch.zeros((rows, K), dtype=torch.bfloat16, device=w.weight.device), w)
    torch.cuda.synchronize(w.weight.device)


# The folder's protocol with ``gemm.mxfp8_common`` (``gemm._SHAPE_HOSTS``): ``mxfp8_linear`` runs ``DISPATCH``'s rows.
ROWS = {(N, K, "bf16"): DISPATCH}
LAUNCHERS = {}
SERVES_FROM = 640  # rows: below, the rows are correct but not faster than the served path (README)


def select(m: int, out_dtype: torch.dtype, allow_decode: bool):
    return _gemm.select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)


def covers(m: int, prequant: bool = False) -> bool:
    return not prequant and SERVES_FROM <= m <= MAX_ROWS
