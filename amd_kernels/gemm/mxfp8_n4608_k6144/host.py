# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 4608 x 6144 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``). Every bf16 row
measured (benchmarks/gemm/mxfp8_common/sweep_rows.py); the split-K decode rows at 33-256 beat the rows they replaced
at every multiple of 8 and window start (benchmarks/gemm/mxfp8_common/benchmark_decode_splitk.py)."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 4608, 6144

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
4,4608,6144,bf16,decode_fq,4,32,128,2,0,0,0,1,,
8,4608,6144,bf16,decode_fq,8,32,128,2,0,0,0,1,,
16,4608,6144,bf16,decode_fq,16,64,128,2,0,0,0,3,,
32,4608,6144,bf16,decode,16,32,128,1,0,0,0,1,,,,0
48,4608,6144,bf16,decode_splitk,48,64,128,4,4,0,0,3,2,0,,
64,4608,6144,bf16,decode_splitk,64,64,128,4,4,0,0,3,2,0,,
80,4608,6144,bf16,decode_splitk,80,64,128,4,4,0,0,3,2,0,,
96,4608,6144,bf16,decode_splitk,96,64,128,4,4,0,0,3,2,0,,
112,4608,6144,bf16,decode_splitk,112,64,128,4,4,0,0,3,2,0,,
128,4608,6144,bf16,decode_splitk,64,128,128,4,4,0,0,3,2,0,,
160,4608,6144,bf16,decode_splitk,80,128,128,4,4,0,0,3,2,0,,
192,4608,6144,bf16,decode_splitk,96,128,128,4,4,0,0,3,2,0,,
224,4608,6144,bf16,decode_splitk,112,128,128,4,4,0,0,3,2,0,,
256,4608,6144,bf16,decode_splitk,128,128,128,4,4,0,0,3,2,0,,
288,4608,6144,bf16,decode,48,64,128,1,0,0,0,1,,,,0
384,4608,6144,bf16,decode,64,64,128,1,0,0,0,1,,,,0
448,4608,6144,bf16,decode,80,64,128,2,0,0,0,1,,,,0
512,4608,6144,bf16,decode,64,96,128,2,0,0,0,1,,,,0
640,4608,6144,bf16,decode,128,96,128,2,0,0,0,1,,,,0
768,4608,6144,bf16,decode,128,64,128,2,0,0,0,1,,,,0
896,4608,6144,bf16,decode,112,96,128,2,0,0,0,1,,,,0
1024,4608,6144,bf16,decode,128,96,128,2,0,0,0,1,,,,0
1536,4608,6144,bf16,4wave,256,128,128,2,2,1,1,1,,
65536,4608,6144,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
