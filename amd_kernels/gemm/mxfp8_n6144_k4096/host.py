# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 GEMM rows and kernels of the o_proj N 6144 x K 4096 (TP2), registered with the common dispatcher
(``gemm.mxfp8_common.gemm._ROUTES``)."""

import functools

import torch

from amd_kernels.gemm.mxfp8_common.gemm import parse_rows, select_from_rows
from amd_kernels.gemm.mxfp8_common.gemm import (
    _DECODE_FQ_KERNEL,
    _DECODE_KERNEL,
    _FOURWAVE_NT_KERNEL,
    _STAGED_KERNEL,
    _Config,
    _fourwave_grid_max,
    splitk_rows,
)
from amd_kernels.gemm.mxfp8_n6144_k8192.host import _prefill_n6144_k8192_192_launcher

_N, _K = 6144, 4096
_TILE_128_KERNEL = "n6144_k4096_128x192"
_ONE_TILE_KERNEL = "n6144_k4096_256x192_one_tile"
_PERSISTENT_KERNEL = "n6144_k4096_256x192"


def _decode(tile_m, tile_n, k_waves, kernel=_DECODE_KERNEL, sched_barriers=0):
    return _Config(tile_m, tile_n, 128, 0, 0, 1, kernel, k_waves, sched_barriers)


def _tile(tile_m, tile_n, kernel):
    return _Config(tile_m, tile_n, 128, 1, 1, 1, kernel, 2, 2)


# (last row, config): the decode rows on (16, 16) codes (a producer's, or the bf16 activation quantized for them);
# swept over tiles 16..160 x 16..96, K-waves 1/2/4/8, split-K 1/2 (benchmarks/gemm/mxfp8_n6144_k4096/sweep_rows.py).
# 49-224: the split-K decode kernel (two m-blocks, K split 2), -3..-13% at every multiple of 8 and window start; +2..+4%
# at 33-48, +0.5..+1% at 240-256 on the layer-39 weights (benchmarks/gemm/mxfp8_common/benchmark_decode_splitk.py,
# benchmarks/gemm/mxfp8_n6144_k4096/benchmark_n6144_k4096.py).
_DECODE_ROWS = (
    (16, _decode(16, 32, 1)),
    (32, _decode(32, 32, 1, sched_barriers=1)),
    (48, _decode(48, 32, 1)),
    *splitk_rows(49, 224, 2, 96, 3, 3, 2, 3),
    (256, _decode(64, 96, 2, sched_barriers=1)),
    (320, _decode(80, 96, 2, sched_barriers=1)),
    (384, _decode(96, 96, 2, sched_barriers=1)),
    (448, _decode(112, 96, 2, sched_barriers=1)),
    (512, _decode(128, 96, 2)),
    (576, _decode(144, 96, 2, sched_barriers=1)),
    (640, _decode(160, 96, 2, sched_barriers=1)),
    (768, _decode(96, 96, 2, sched_barriers=1)),
)
# The two-K-wave 96-column rows with scheduling barriers read owned accumulators in the owner's branch
# (OWNER_READ): -0.5..-3.5 % at 225-448 and 513-768 rows, bitwise equal (sweep_rows.py --family decode).
_OWNER_READ = frozenset({(64, 96), (80, 96), (96, 96), (112, 96), (144, 96), (160, 96)})
# a bf16 activation at few rows: the decode kernel quantizes it in registers (no quantizer launch)
_DECODE_FQ_ROWS = (
    (4, _decode(4, 32, 2, _DECODE_FQ_KERNEL)),
    (8, _decode(8, 32, 2, _DECODE_FQ_KERNEL)),
    (16, _decode(16, 32, 2, _DECODE_FQ_KERNEL)),
    (32, _decode(16, 48, 4, _DECODE_FQ_KERNEL)),
)
# row-major codes (above 768 rows also the bf16 activation's rows, after the quantizer). The 4-wave 256x256 row
# and the persistent 256x192 tile alternate with their wave quantization: the 256x192 tile -1.0..-3.5 % at 3329-4096
# and 5121-6144 rows, +4..18 % elsewhere above 2048 (bf16 in; sweep_rows.py --family prefill).
_PREFILL_ROWS = (
    (32, _Config(32, 32, 256, 0, 0, 4)),
    (128, _Config(64, 64, 256, 0, 0, 2)),
    (384, _Config(128, 64, 256, 2, 4, 1)),
    (640, _Config(128, 128, 128, 2, 4, 1, _STAGED_KERNEL, 2, 4)),
    (1024, _tile(128, 192, _TILE_128_KERNEL)),
    (1280, _tile(256, 128, _FOURWAVE_NT_KERNEL)),
    (2048, _tile(256, 192, _ONE_TILE_KERNEL)),
    (3328, _tile(256, 256, _FOURWAVE_NT_KERNEL)),
    (4096, _tile(256, 192, _PERSISTENT_KERNEL)),
    (5120, _tile(256, 256, _FOURWAVE_NT_KERNEL)),
    (6144, _tile(256, 192, _PERSISTENT_KERNEL)),
)
_LARGE = _tile(256, 256, _FOURWAVE_NT_KERNEL)


def _row(rows, m):
    for last, config in rows:
        if m <= last:
            return config
    return None


def _select_own(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    """This shape's config for ``m`` rows ahead of ``ROWS`` (``_select_config``), or None."""
    if out_dtype != torch.bfloat16:
        return None
    if allow_decode:
        config = _row(_DECODE_FQ_ROWS, m) or _row(_DECODE_ROWS, m)
        if config is not None:
            return config
    return _row(_PREFILL_ROWS, m) or _LARGE


def select_prequant(m: int, out_dtype: torch.dtype) -> _Config | None:
    """The config for producer codes (``_select_prequant_config``): the decode rows without the in-kernel quantizer."""
    if out_dtype != torch.bfloat16:
        return None
    return _row(_DECODE_ROWS, m)


def decode_launcher(config: _Config, a_bf16: bool):
    """The decode kernel of an owner-read row (``_OWNER_READ``), or None (the common decode kernel)."""
    if (
        a_bf16 or config.kernel != _DECODE_KERNEL or (config.tile_m, config.tile_n) not in _OWNER_READ
        or (config.waves_m, config.split_k, config.waves_n) != (2, 1, 1)
        or config.la_w or config.la_a or config.w_cm is not None
    ):
        return None
    from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import compile_mxfp8_gemm_decode

    return compile_mxfp8_gemm_decode(
        N=_N, K=_K, RT=config.tile_m // 16, NI=config.tile_n // 16, KW=2, KS=1,
        OWNER_READ=True,
    )


def _args(activation, activation_scale, weight, weight_scale, out):
    assert activation.shape[1] == _K and out.shape[1] == _N and out.dtype == torch.bfloat16
    return (
        activation.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
        activation_scale.view(torch.uint8).view(-1), weight_scale.view(torch.uint8).view(-1),
    )


@functools.cache
def _tile_128_launcher():
    from .tile_128x192 import compile_mxfp8_gemm_n6144_k4096_128x192
    return compile_mxfp8_gemm_n6144_k4096_128x192()


def _launch_128(activation, activation_scale, weight, weight_scale, out):
    from amd_kernels.common.tensor_shim import _run_compiled
    _run_compiled(
        _tile_128_launcher(), *_args(activation, activation_scale, weight, weight_scale, out),
        activation.shape[0], _N, torch.cuda.current_stream(),
    )


@functools.cache
def _one_tile_launcher():
    from .prefill_256x192_one_tile import compile_mxfp8_gemm_n6144_k4096_256x192_one_tile
    return compile_mxfp8_gemm_n6144_k4096_256x192_one_tile()


def _launch_256(activation, activation_scale, weight, weight_scale, out, *, one_tile):
    from amd_kernels.common.tensor_shim import _run_compiled
    m = out.shape[0]
    n_work = ((m + 255) // 256) * (_N // 192)
    if one_tile:
        if n_work > 256:
            raise ValueError("the one-tile 256x192 GEMM needs at most 256 tiles")
        exe = _one_tile_launcher()
    else:
        # more tiles than CUs: inside a DBO microbatch the plain grid, as the 4-wave rows
        exe = _prefill_n6144_k8192_192_launcher(_K, _fourwave_grid_max() if n_work > 256 else 256)
    _run_compiled(
        exe, *_args(activation, activation_scale, weight, weight_scale, out),
        m, _N, 0, n_work, torch.cuda.current_stream(),
    )


LAUNCHERS = {
    _TILE_128_KERNEL: lambda config, a, sa, w, sw, out: _launch_128(a, sa, w, sw, out),
    _ONE_TILE_KERNEL: lambda config, a, sa, w, sw, out: _launch_256(a, sa, w, sw, out, one_tile=True),
    _PERSISTENT_KERNEL: lambda config, a, sa, w, sw, out: _launch_256(a, sa, w, sw, out, one_tile=False),
}


# This shape's rows. ``ROWS`` holds one ``min_m`` 65536 row: it lists the shape in ``gemm.tuned_shapes``;
# ``_select_own`` picks every row.
ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
65536,6144,4096,bf16,4wave_nt,256,256,128,2,2,1,1,1,,,65536
""")


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    """This shape's row for ``m``: its own kernels' windows (``_select_own``), else ``ROWS``."""
    config = _select_own(m, out_dtype, allow_decode)
    if config is not None:
        return config
    return select_from_rows(ROWS, m, 6144, 4096, out_dtype, allow_decode)
