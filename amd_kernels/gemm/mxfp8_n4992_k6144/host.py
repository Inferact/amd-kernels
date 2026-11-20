# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 GEMM rows and kernels of the QKV projection N 4992 x K 6144 (TP2), registered with the common
dispatcher (``gemm.mxfp8_common.gemm._SHAPE_HOSTS``): every bf16 row of the shape (``ROWS`` only lists the shape), the
deferred (split-K partial) rows of the fused norm / RoPE / insert chain, and the launches of the persistent kernel
(``persistent``), one or two (a row split) per call."""

import functools

import torch

from amd_kernels.gemm.mxfp8_common.gemm import parse_rows, select_from_rows
from amd_kernels.gemm.mxfp8_common.gemm import (
    _DECODE_KERNEL,
    _PLAIN_GRID_MAX,
    _Config,
    _fourwave_grid_max,
    splitk_rows,
)

N, K = 4992, 6144
_PERSISTENT_KERNEL = "persistent_n4992_k6144"
_PERSISTENT_SPLIT_KERNEL = "persistent_split_n4992_k6144"


def _decode(tile_m, tile_n, k_waves, split_k=1, lookahead=(0, 0)):
    return _Config(tile_m, tile_n, 128, 0, 0, split_k, _DECODE_KERNEL, k_waves, 0, *lookahead)


def _persistent(tile_m, tile_n, kernel=_PERSISTENT_KERNEL):
    return _Config(tile_m, tile_n, 128, 1, 1, 1, kernel, 2, 2)


# Decode rows on (16, 16) codes: (largest M, row); sweep: benchmarks/gemm/mxfp8_n4992_k6144/sweep_n4992_k6144.py.
# An explicit lookahead (W steps, A blocks) where the deepest one parks operands in AGPRs (tests/.../test_isa.py)
# or a shallower one measured faster. 49-256: the split-K decode kernel, K split 3 (the kw3 rows' K order, bitwise),
# one m-block to 112 rows, two above: -9..-25% at every multiple of 8 and window start
# (benchmarks/gemm/mxfp8_common/benchmark_decode_splitk.py); with the insert -6..-8% at 49-64 against the deferred
# split-2 chain, +-2% at 33-48.
DECODE_ROWS = (
    (16, _decode(16, 32, 1)),
    (32, _decode(32, 32, 1)),
    (48, _decode(48, 32, 1, lookahead=(4, 9))),
    *splitk_rows(49, 112, 1, 64, 4, 4, 3, 2),
    *splitk_rows(113, 256, 2, 128, 4, 4, 3, 2),
    (320, _decode(160, 48, 4)),
    (384, _decode(192, 48, 3, lookahead=(1, 8))),
    (448, _decode(224, 48, 3, lookahead=(1, 10))),
    (512, _decode(128, 96, 3)),
    (576, _decode(192, 64, 3, lookahead=(1, 8))),
    (624, _decode(208, 64, 3, lookahead=(1, 6))),
    (672, _decode(224, 64, 3)),
    (720, _decode(240, 64, 3)),
    (768, _decode(256, 64, 3, lookahead=(1, 8))),
)
DECODE_MAX = DECODE_ROWS[-1][0]
# Rows from 385 take the grouped XCD order of the decode kernel (an XCD's contiguous eighth of the column-major
# four-m-block groups: W and A shared in its L2): -4.6..-15% at 448-720 rows against the plain order; flat at two
# m-blocks below. With them the decode rows reach 768 (256x64: 30.2 us against the persistent tile's 34.1).
GROUPED_ROWS = frozenset(config for top, config in DECODE_ROWS if top > 384)
# row-major codes at up to this many rows keep the common fallback (AITER split-K) row: 12.6..24.7 us against the
# persistent 128x128 tile's ~33 us (same sweep)
ROW_MAJOR_FALLBACK_MAX = 192

# Deferred rows of the chain (partials into the split-K reduce + norm / RoPE / insert), where they beat the decode
# row + insert; same sweep, --chain.
DEFERRED_ROWS = (
    (16, _decode(16, 48, 1, 2)),
    (32, _decode(32, 48, 1, 2)),
    (48, _decode(48, 64, 1, 3)),
)

# Persistent tiles: microseconds per round of 256 workgroups (K 6144, graph replay, cold weights; same sweep,
# --prefill). A row count takes the tile of the fewest round-microseconds, the larger tile on a tie.
PERSISTENT_ROUND_US = {(256, 192): 50.0, (256, 128): 39.5, (192, 192): 42.0, (192, 128): 34.5, (128, 192): 37.0,
                       (128, 128): 33.0}


def _rounds(m, tile):
    return -(-(-(-m // tile[0]) * (N // tile[1])) // 256)


@functools.cache
def persistent_tile(m: int) -> tuple[int, int]:
    """The persistent kernel's tile for ``m`` rows."""
    return min(PERSISTENT_ROUND_US, key=lambda t: (_rounds(m, t) * PERSISTENT_ROUND_US[t], -t[0] * t[1]))


# A row split (a first launch of whole rounds, the rest with another tile) is taken where the round model predicts
# 5% or more: measured -0.4..-8.7% at 5632-20480 rows, +0.6% (noise) at 2560 (same sweep, --split; every tile sums
# K in the same order, so the split's bytes are the one launch's).
SPLIT_MIN_GAIN = 0.05


@functools.cache
def persistent_plan(m: int) -> tuple[tuple[int, tuple[int, int]], ...]:
    """The persistent launches for ``m`` rows: (first row, tile) each, in launch order."""
    tile = persistent_tile(m)
    best, plan = _rounds(m, tile) * PERSISTENT_ROUND_US[tile] * (1 - SPLIT_MIN_GAIN), ((0, tile),)
    for first in PERSISTENT_ROUND_US:
        for rounds in range(1, _rounds(m, first)):
            rows = 256 * rounds // (N // first[1]) * first[0]
            if not 0 < rows < m:
                continue
            for rest in PERSISTENT_ROUND_US:
                us = rounds * PERSISTENT_ROUND_US[first] + _rounds(m - rows, rest) * PERSISTENT_ROUND_US[rest]
                if us < best:
                    best, plan = us, ((0, first), (rows, rest))
    return plan


def decode_launcher(config: _Config, a_bf16: bool):
    """The decode kernel of a ``GROUPED_ROWS`` row, or None (the common decode kernel)."""
    if a_bf16 or config not in GROUPED_ROWS:
        return None
    from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import compile_mxfp8_gemm_decode

    return compile_mxfp8_gemm_decode(
        N=N, K=K, RT=config.tile_m // 16, NI=config.tile_n // 16, KW=config.waves_m, KS=config.split_k,
        LA=(config.la_w, config.la_a) if config.la_w else None, sched_barriers=bool(config.waves_n), GROUPED=True,
    )


def _decode_row(m):
    for top, config in DECODE_ROWS:
        if m <= top:
            return config
    return None


def _select_own(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    """The shape's row for ``m`` rows (``_select_config``); None keeps the common fallback (fp16 output, row-major
    codes at up to ``ROW_MAJOR_FALLBACK_MAX`` rows)."""
    if out_dtype != torch.bfloat16:
        return None
    if allow_decode and m <= DECODE_MAX:
        return _decode_row(m)
    if m <= ROW_MAJOR_FALLBACK_MAX:
        return None
    plan = persistent_plan(m)
    return _persistent(*plan[0][1], _PERSISTENT_SPLIT_KERNEL if len(plan) > 1 else _PERSISTENT_KERNEL)


def select_deferred(m: int, config: _Config) -> _Config:
    """The chain's split-K row for ``m`` rows of producer codes in ``config``'s tile, else ``config``. A deferred
    row reads a prefix of the producer's codes: ``ceil(m / tile) * tile`` never exceeds the producer's rows."""
    if config.kernel != _DECODE_KERNEL:
        return config
    for top, row in DEFERRED_ROWS:
        if m <= top:
            held = -(-m // config.tile_m) * config.tile_m
            return row if -(-m // row.tile_m) * row.tile_m <= held else config
    return config


@functools.cache
def _persistent_launcher(tiles: tuple, grid_max: int):
    from .persistent import compile_mxfp8_gemm_n4992_k6144

    return compile_mxfp8_gemm_n4992_k6144(tiles, True, grid_max)


def _launch_tile(tile, grid_max, activation, activation_scale, weight, weight_scale, out):
    from amd_kernels.common.tensor_shim import _run_compiled

    m, n = out.shape
    _run_compiled(
        _persistent_launcher((tile[0] // 64, tile[1] // 64), grid_max),
        activation.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
        activation_scale.view(torch.uint8).view(-1), weight_scale.view(torch.uint8).view(-1),
        m, n, -(-m // tile[0]) * (n // tile[1]), torch.cuda.current_stream(),
    )


_compiled_all = []


def _launch_persistent(tile, activation, activation_scale, weight, weight_scale, out):
    m, n = out.shape
    if activation.shape[1] != K or n != N or n % tile[1]:
        raise ValueError(f"persistent N4992 K6144 GEMM got K {activation.shape[1]}, N {n}, tile {tile}")
    # Inside a DBO microbatch more tiles than CUs take the plain grid. Every tile and grid form writes the same
    # bytes; the first launch outside capture runs (compiles) them all, so no later row count or DBO step compiles.
    if not _compiled_all and not torch.cuda.is_current_stream_capturing():
        for other in PERSISTENT_ROUND_US:
            for grid_max in (256, _PLAIN_GRID_MAX):
                _launch_tile(other, grid_max, activation, activation_scale, weight, weight_scale, out)
        _compiled_all.append(True)
    grid_max = _fourwave_grid_max() if -(-m // tile[0]) * (n // tile[1]) > 256 else 256
    _launch_tile(tile, grid_max, activation, activation_scale, weight, weight_scale, out)


def _launch_split(activation, activation_scale, weight, weight_scale, out):
    """``persistent_plan``'s launches over row ranges (row-major codes, scale rows in 32-row groups)."""
    m = out.shape[0]
    plan = persistent_plan(m)
    for i, (first, tile) in enumerate(plan):
        last = plan[i + 1][0] if i + 1 < len(plan) else m
        _launch_persistent(tile, activation[first:last], activation_scale[first:], weight, weight_scale,
                           out[first:last])


LAUNCHERS = {
    _PERSISTENT_KERNEL: lambda config, a, sa, w, sw, out: _launch_persistent((config.tile_m, config.tile_n), a, sa,
                                                                             w, sw, out),
    _PERSISTENT_SPLIT_KERNEL: lambda config, a, sa, w, sw, out: _launch_split(a, sa, w, sw, out),
}


# This shape's rows. ``ROWS`` holds one ``min_m`` 65536 row: it lists the shape in ``gemm.tuned_shapes``;
# ``_select_own`` picks every row.
ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
65536,4992,6144,bf16,aiter,128,128,128,0,0,2,4,1,,,65536
""")


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    """This shape's row for ``m``: its own kernels' windows (``_select_own``), else ``ROWS``."""
    config = _select_own(m, out_dtype, allow_decode)
    if config is not None:
        return config
    return select_from_rows(ROWS, m, 4992, 6144, out_dtype, allow_decode)
