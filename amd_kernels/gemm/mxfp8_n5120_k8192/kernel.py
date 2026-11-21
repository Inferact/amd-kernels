# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Tuned rows of the N 5120 x K 8192 MXFP8 GEMM: which kernel and tile serve each row count.

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

N = 5120
K = 8192

LINEAR = (
    ("fq", 16, False, Config(1, 5, 4, 4, 0, None, 1)),
    ("fq", 32, False, Config(2, 5, 4, 4)),
    ("common", 64, False, ("decode", 64, 80, 128, 0, 0, 4, 2, 0, 1)),
    ("common", 128, False, ("decode", 48, 64, 128, 0, 0, 1, 4, 1, None)),
    ("common", 192, False, ("decode", 96, 80, 128, 0, 0, 2, 2, 0, 1)),
    ("common", 256, True, ("decode", 128, 80, 128, 0, 0, 2, 2, 0, 1)),
    ("common", 384, True, ("decode", 128, 64, 128, 0, 0, 1, 2, 0, 0)),
    ("common", 512, True, ("decode", 128, 80, 128, 0, 0, 1, 4, 0, 0)),
    ("common", 768, False, ("staged_ab", 256, 128, 128, 1, 0, 2, 2, 4, None)),
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
