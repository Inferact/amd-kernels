# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 12288 x 6144 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``). Every bf16 row
measured (benchmarks/gemm/mxfp8_common/sweep_rows.py); the split-K decode rows at 49-224 beat the rows they replaced
at every multiple of 8 and window start, 2-4% slower at 225-256
(benchmarks/gemm/mxfp8_common/benchmark_decode_splitk.py)."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 12288, 6144

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
4,12288,6144,bf16,decode_fq,8,32,128,1,0,0,0,1,,
8,12288,6144,bf16,decode_fq,8,48,128,4,0,0,0,1,,
16,12288,6144,bf16,decode_fq,16,48,128,3,0,0,0,1,,
32,12288,6144,bf16,decode,32,48,128,1,0,0,0,1,,
48,12288,6144,bf16,decode,48,48,128,1,0,0,0,1,,
64,12288,6144,bf16,decode_splitk,64,96,128,6,2,0,0,2,3,4,,
80,12288,6144,bf16,decode_splitk,80,96,128,6,2,0,0,2,3,4,,
96,12288,6144,bf16,decode_splitk,96,96,128,6,2,0,0,2,3,4,,
112,12288,6144,bf16,decode_splitk,112,96,128,6,2,0,0,2,3,4,,
128,12288,6144,bf16,decode_splitk,128,96,128,6,2,0,0,2,3,4,,
144,12288,6144,bf16,decode_splitk,144,96,128,6,2,0,0,2,3,4,,
160,12288,6144,bf16,decode_splitk,160,96,128,6,2,0,0,2,3,4,,
176,12288,6144,bf16,decode_splitk,176,96,128,6,2,0,0,2,3,4,,
192,12288,6144,bf16,decode_splitk,192,96,128,6,2,0,0,2,3,4,,
208,12288,6144,bf16,decode_splitk,208,96,128,6,2,0,0,2,3,4,,
224,12288,6144,bf16,decode_splitk,224,96,128,6,2,0,0,2,3,4,,
256,12288,6144,bf16,decode,128,96,128,2,0,0,0,1,,
288,12288,6144,bf16,decode,96,96,128,2,0,0,0,1,,,,0
320,12288,6144,bf16,decode,112,96,128,2,0,0,0,1,,,,0
384,12288,6144,bf16,decode,128,96,128,2,0,0,0,1,,,,0
448,12288,6144,bf16,decode,160,96,128,2,0,0,0,1,,,,0
512,12288,6144,bf16,4wave_nt,256,128,128,2,2,1,1,1,,
640,12288,6144,bf16,decode,160,96,128,2,0,0,0,1,,,,0
768,12288,6144,bf16,decode,128,96,128,1,0,0,0,1,,,,0
65536,12288,6144,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
