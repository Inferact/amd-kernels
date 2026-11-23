# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Tuned rows of the N 4096 x K 1280 MXFP8 GEMM: which kernel and tile serve each row count.

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

N = 4096
K = 1280

LINEAR = (
    ("fq", 16, True, Config(1, 1, 5)),
    ("fq", 32, True, Config(1, 2, 5, 1, 0, 2)),
    ("fq", 64, True, Config(1, 4, 5)),
    ("fq", 128, True, Config(2, 4, 5, 1, 0, 2)),
    ("common", 192, False, ("decode", 96, 32, 128, 0, 0, 1, 1, 0, None)),
    ("common", 256, False, ("decode", 64, 64, 128, 0, 0, 1, 1, 0, None)),
    ("common", 384, False, ("decode", 96, 64, 128, 0, 0, 1, 5, 0, 2)),
    ("common", 512, False, ("aiter", 128, 64, 256, 2, 4, 1, 0, 0, None)),
    ("common", 768, False, ("aiter", 64, 128, 256, 2, 4, 1, 0, 0, None)),
)

PREQUANT = (
    ("rm", 16, False, Config(1, 1, 1, 1, 1, 2)),
    ("rm", 32, True, Config(1, 2, 1)),
    ("rm", 64, True, Config(2, 2, 1)),
    ("rm", 128, False, Config(2, 4, 1)),
    ("rm", 192, False, Config(3, 4, 1)),
    ("rm", 256, False, Config(4, 4, 1)),
    ("rm", 384, False, Config(6, 4, 1)),
    ("rm", 512, False, Config(8, 4, 1)),
    ("rm", 768, False, Config(12, 4, 1)),
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
