# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 GEMM rows and kernels of the o_proj N 6144 x K 2048 (TP4), registered with the common dispatcher
(``gemm.mxfp8_common.gemm._ROUTES``): its windows ahead of ``ROWS`` (the 128x192 tile here, the
256x192 persistent tile of ``gemm/mxfp8_n6144_k8192`` at K 2048), the producer rows of pre-quantized codes, and
the decode row whose activation stages through LDS (``decode_astage``)."""

import functools

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, _DECODE_KERNEL, parse_rows, select_from_rows
from amd_kernels.gemm.mxfp8_n6144_k8192.host import _PREFILL_N6144_K8192_192_KERNEL

_N6144_K2048_128_KERNEL = "n6144_k2048_128x192"


def _select_own(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    """The rows ahead of ``ROWS`` (``_select_config``), or None."""
    # The TP4 o_proj at 3329-3968 rows (14-16 m-blocks, the last one at most half full): the
    # persistent 256x192 tile at K 2048, 448-512 tiles in two rounds, instead of the AITER
    # 128x256 row's three (graph -4.5..-12.7 %, eager -3.6..-12.6 %). From
    # 3969 rows both fill their rounds and the eager gain is gone (4032 -1 % overlapping, 4095 0).
    if out_dtype == torch.bfloat16 and 3328 < m <= 3968:
        return _Config(256, 192, 128, 1, 1, 1, _PREFILL_N6144_K8192_192_KERNEL, 2, 2)
    # The TP4 o_proj, the same 128x192 kernel at K 2048: one round of up to 256 tiles
    # instead of 192 256x128 tiles (row-major codes; above 1024 rows a second round loses).
    if out_dtype == torch.bfloat16 and 768 < m <= 1024:
        return _Config(128, 192, 128, 1, 1, 1, _N6144_K2048_128_KERNEL, 2, 2)
    return None


def select_prequant(m: int, out_dtype: torch.dtype) -> _Config | None:
    """A codes input can avoid the K-wave reduction paid by tiny bf16 rows
    (``_select_prequant_config``)."""
    if out_dtype == torch.bfloat16 and 1 <= m <= 32:
        return _Config(16 if m <= 16 else 32, 32, 128, 0, 0, 1, _DECODE_KERNEL, 1, 0)
    return None


def decode_launcher(config: _Config, a_bf16: bool):
    """The decode row whose activation stages through LDS, or None (the common decode kernel)."""
    if (
        not a_bf16 and config.kernel == _DECODE_KERNEL
        and (config.tile_m, config.tile_n, config.waves_m, config.split_k) == (64, 48, 1, 1)
        and not config.la_w and not config.la_a and config.w_cm is None
    ):
        # M97-128 keeps its producer's 64-row code tile and deterministic K order.
        from .decode_astage import compile_mxfp8_gemm_n6144_k2048_astage

        return compile_mxfp8_gemm_n6144_k2048_astage()
    return None


@functools.cache
def _n6144_k2048_128_launcher():
    from .tile_128x192 import compile_mxfp8_gemm_n6144_k2048_128x192
    return compile_mxfp8_gemm_n6144_k2048_128x192()


def _launch_n6144_k2048_128(activation, activation_scale, weight, weight_scale, out):
    from amd_kernels.common.tensor_shim import _run_compiled
    assert activation.shape[1] == 2048 and out.shape[1] == 6144 and out.dtype == torch.bfloat16
    _run_compiled(
        _n6144_k2048_128_launcher(),
        activation.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
        activation_scale.view(torch.uint8).view(-1), weight_scale.view(torch.uint8).view(-1),
        activation.shape[0], 6144, torch.cuda.current_stream(),
    )


LAUNCHERS = {
    _N6144_K2048_128_KERNEL: lambda config, a, sa, w, sw, out: _launch_n6144_k2048_128(a, sa, w, sw, out),
}


# This shape's rows. Measured bf16 rows at 33-48, 65-80, 129-160, 161-192, 257-320, 385-448 and 1281-3328 (graph
# replay, one MI355X).
ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
1,6144,2048,any,aiter,32,32,256,0,0,0,0,4,,
4,6144,2048,any,aiter,32,32,256,0,0,0,0,4,,
16,6144,2048,any,aiter,32,32,256,0,0,0,0,4,,
64,6144,2048,any,aiter,64,128,256,0,0,0,0,1,,
128,6144,2048,any,aiter,64,64,256,0,0,2,4,1,,
192,6144,2048,any,aiter,64,64,256,0,0,2,4,1,,
256,6144,2048,any,aiter,128,64,256,0,0,2,4,1,,
320,6144,2048,any,aiter,64,64,256,0,0,2,4,1,,
384,6144,2048,any,aiter,128,64,256,0,0,2,4,1,,
512,6144,2048,any,aiter,128,64,256,0,0,2,4,1,,
640,6144,2048,any,aiter,128,64,256,0,0,2,4,1,,
768,6144,2048,any,aiter,64,128,256,0,0,2,4,1,,
1024,6144,2048,any,aiter,128,128,128,0,0,2,4,1,,
4096,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
8192,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
8224,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
10240,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
12288,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
14336,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
16384,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
18432,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
20480,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
22528,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
24576,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
26624,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
28672,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
30720,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
32768,6144,2048,any,aiter,128,256,128,0,0,2,4,1,,
8,6144,2048,bf16,decode_fq,8,32,128,2,0,0,0,1,,
16,6144,2048,bf16,decode_fq,16,32,128,2,0,0,0,1,,
32,6144,2048,bf16,decode_fq,16,64,128,2,0,0,0,1,,
48,6144,2048,bf16,decode,48,32,128,1,0,0,0,1,,
64,6144,2048,bf16,decode,32,48,128,1,0,0,0,1,,
80,6144,2048,bf16,decode,80,32,128,1,0,0,0,1,,
96,6144,2048,bf16,decode,96,32,128,1,0,0,0,1,,
128,6144,2048,bf16,decode,64,48,128,1,0,0,0,1,,
160,6144,2048,bf16,decode,80,48,128,4,0,0,0,1,,
192,6144,2048,bf16,decode,96,48,128,4,0,0,0,1,,
256,6144,2048,bf16,decode,64,96,128,2,0,0,0,1,,
320,6144,2048,bf16,decode,80,96,128,4,0,0,0,1,,
384,6144,2048,bf16,decode,96,96,128,2,0,0,0,1,,
448,6144,2048,bf16,decode,112,96,128,2,0,0,0,1,,
512,6144,2048,bf16,decode,128,96,128,2,0,0,0,1,,
640,6144,2048,bf16,staged_ab,128,128,128,2,4,2,4,1,,
768,6144,2048,bf16,decode,96,96,128,1,0,0,0,1,,
1280,6144,2048,bf16,4wave_nt,256,128,128,2,2,1,1,1,,
2048,6144,2048,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
3072,6144,2048,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
3328,6144,2048,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
4096,6144,2048,bf16,aiter,128,256,128,0,0,2,4,1,,
12288,6144,2048,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
16384,6144,2048,bf16,4wave,256,256,128,2,2,1,1,1,,
65536,6144,2048,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
""")


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    """This shape's row for ``m``: its own kernels' windows (``_select_own``), else ``ROWS``."""
    config = _select_own(m, out_dtype, allow_decode)
    if config is not None:
        return config
    return select_from_rows(ROWS, m, 6144, 2048, out_dtype, allow_decode)
