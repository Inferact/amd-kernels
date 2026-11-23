# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Tuned rows of the N 1792 x K 5120 MXFP8 GEMM: which kernel and tile serve each row count.

A row is (kind, rows <=, serves, config). ``serves``: the row beat the served path (vLLM quantizer + AITER
``gemm_a8w8_blockscale_bpreshuffle``) in the benchmark; callers fall back to it where it is False. Kinds: ``fq``
(``gemm.mxfp8_decode_prequant``: the GEMM quantizes the bf16 activation in registers, one launch; every N-tile
workgroup quantizes the whole activation it reads, so it pays up to M 32-64), ``rm`` (pre-quantized row-major
activation, same kernel), ``shuf`` (a quantizer launch, then the kernel on (16, 16) codes), ``common`` (a row of
``gemm.mxfp8_common``: ``decode`` / ``staged_ab`` / ``aiter`` after a quantizer launch). Every row is the best of a
sweep of tile, K-waves, split-K and weight cache modifier in graph replay over cold weights.
"""

from amd_kernels.gemm.mxfp8_common.gemm import _Config
from amd_kernels.gemm.mxfp8_decode_prequant import Config

N = 1792
K = 5120

LINEAR = (
    ("fq", 16, True, Config(1, 2, 5, 4, 0, None, 1)),
    ("fq", 32, True, Config(1, 2, 2, 2, 1, None, 1)),
    ("common", 64, True, ("decode", 32, 16, 128, 0, 0, 1, 5, 0, None)),
    ("common", 128, True, ("decode", 32, 32, 128, 0, 0, 1, 4, 1, 2)),
    ("common", 192, True, ("decode", 48, 32, 128, 0, 0, 1, 4, 1, 2)),
    ("common", 256, True, ("decode", 64, 32, 128, 0, 0, 1, 4, 1, None)),
    ("common", 384, False, ("decode", 96, 32, 128, 0, 0, 1, 4, 1, 2)),
    ("common", 512, True, ("decode", 128, 32, 128, 0, 0, 1, 4, 1, None)),
    ("common", 768, False, ("decode", 192, 32, 128, 0, 0, 1, 4, 1, 1)),
)


def row(table, m: int):
    """The (kind, rows <=, serves, config) of ``table`` for ``m`` rows, or None above its last row."""
    for entry in table:
        if m <= entry[1]:
            return entry
    return None


def common_config(fields) -> _Config:
    kernel, tile_m, tile_n, tile_k, waves_per_eu, xcd_swizzle, split_k, waves_m, waves_n, w_cm = fields
    return _Config(tile_m, tile_n, tile_k, waves_per_eu, xcd_swizzle, split_k, kernel, waves_m, waves_n, w_cm=w_cm)
