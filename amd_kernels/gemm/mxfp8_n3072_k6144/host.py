# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 3072 x 6144 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``). Every bf16 row
measured (benchmarks/gemm/mxfp8_common/sweep_rows.py); the split-K decode rows at 33-256 beat the rows they replaced
at every multiple of 8 and window start (benchmarks/gemm/mxfp8_common/benchmark_decode_splitk.py)."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 3072, 6144

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
4,3072,6144,bf16,decode_fq,4,16,128,2,0,0,0,1,,
8,3072,6144,bf16,decode_fq,8,16,128,3,0,0,0,1,,
16,3072,6144,bf16,decode_fq,16,48,128,1,0,0,0,8,,
32,3072,6144,bf16,decode_fq,32,48,128,2,0,0,0,4,,
48,3072,6144,bf16,decode_splitk,48,48,128,3,3,0,0,4,3,0,,
64,3072,6144,bf16,decode_splitk,64,48,128,3,3,0,0,4,3,0,,
80,3072,6144,bf16,decode_splitk,80,48,128,3,3,0,0,4,3,0,,
96,3072,6144,bf16,decode_splitk,96,48,128,3,3,0,0,4,3,0,,
112,3072,6144,bf16,decode_splitk,112,48,128,3,3,0,0,4,3,0,,
128,3072,6144,bf16,decode_splitk,128,48,128,3,3,0,0,4,3,0,,
144,3072,6144,bf16,decode_splitk,144,48,128,3,3,0,0,4,3,0,,
160,3072,6144,bf16,decode_splitk,160,48,128,3,3,0,0,4,3,0,,
176,3072,6144,bf16,decode_splitk,176,48,128,3,3,0,0,4,3,0,,
192,3072,6144,bf16,decode_splitk,192,48,128,3,3,0,0,4,3,0,,
208,3072,6144,bf16,decode_splitk,208,48,128,3,3,0,0,4,3,0,,
224,3072,6144,bf16,decode_splitk,112,48,128,3,3,0,0,2,4,0,,
256,3072,6144,bf16,decode_splitk,128,48,128,3,3,0,0,2,4,0,,
320,3072,6144,bf16,decode,80,48,128,3,0,0,0,1,,
384,3072,6144,bf16,decode,96,48,128,3,0,0,0,1,,
448,3072,6144,bf16,decode,112,48,128,3,0,0,0,1,,,,0
512,3072,6144,bf16,decode,128,48,128,3,0,0,0,1,,,,0
640,3072,6144,bf16,decode,160,48,128,3,0,0,0,1,,,,0
768,3072,6144,bf16,decode,192,48,128,3,0,0,0,1,,,,0
896,3072,6144,bf16,aiter,128,64,256,0,0,2,4,1,,
1024,3072,6144,bf16,staged_ab,128,256,128,2,4,2,4,2,,
1280,3072,6144,bf16,staged_ab,128,128,128,2,4,2,4,1,,
2560,3072,6144,bf16,4wave_nt,256,128,128,2,2,1,1,1,,
65536,3072,6144,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
