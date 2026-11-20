# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 6144 x 6144 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``). Split-K decode at
33-256 beats the decode rows at every row count, its 225-256 row looks 2 W steps ahead (3 spills a VGPR, 1% slower);
K-wave 2 decode rows at 257-320, 385-448 and 641-1024, 4wave_nt 256 x 128 at 1025-1280
(benchmarks/gemm/mxfp8_common/benchmark_decode_splitk.py, sweep_rows.py)."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 6144, 6144

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
1,6144,6144,any,aiter,32,64,128,0,0,0,0,12,,
4,6144,6144,any,aiter,32,32,256,0,0,0,0,4,,
16,6144,6144,any,aiter,32,64,256,0,0,0,0,8,,
64,6144,6144,any,aiter,64,64,256,0,0,0,0,4,,
128,6144,6144,any,aiter,128,64,256,0,0,2,4,4,,
192,6144,6144,any,aiter,64,128,256,0,0,2,4,3,,
256,6144,6144,any,aiter,128,128,256,0,0,0,0,2,,
320,6144,6144,any,aiter,128,64,256,0,0,2,4,1,,
384,6144,6144,any,aiter,128,64,256,0,0,2,4,1,,
512,6144,6144,any,aiter,64,128,256,0,0,2,4,1,,
640,6144,6144,any,aiter,128,128,256,0,0,2,4,1,,
768,6144,6144,any,aiter,64,128,256,0,0,2,4,1,,
1024,6144,6144,any,aiter,128,128,256,0,0,2,4,1,,
4096,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
8192,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
8224,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
10240,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
12288,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
14336,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
16384,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
18432,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
20480,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
22528,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
24576,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
26624,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
28672,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
30720,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
32768,6144,6144,any,aiter,128,256,128,0,0,2,4,1,,
8,6144,6144,bf16,decode_fq,8,32,128,2,0,0,0,1,6,3
16,6144,6144,bf16,decode,16,32,128,1,0,0,0,1,7,7
32,6144,6144,bf16,decode,32,32,128,1,0,0,0,1,,
64,6144,6144,bf16,decode_splitk,32,96,128,3,3,0,0,2,3,0,,
96,6144,6144,bf16,decode_splitk,48,96,128,3,3,0,0,2,3,0,,
128,6144,6144,bf16,decode_splitk,64,96,128,3,3,0,0,2,3,0,,
160,6144,6144,bf16,decode_splitk,80,96,128,3,3,0,0,2,3,0,,
192,6144,6144,bf16,decode_splitk,96,96,128,3,3,0,0,2,3,0,,
224,6144,6144,bf16,decode_splitk,112,96,128,3,3,0,0,2,3,0,,
256,6144,6144,bf16,decode_splitk,128,96,128,3,3,0,0,2,2,0,,
320,6144,6144,bf16,decode,48,96,128,2,0,0,0,1,,,,0
384,6144,6144,bf16,decode,96,96,128,3,0,0,0,1,,
448,6144,6144,bf16,decode,64,96,128,2,0,0,0,1,,,,0
512,6144,6144,bf16,decode,128,96,128,4,0,0,0,1,,
640,6144,6144,bf16,decode,160,96,128,3,0,0,0,1,,
1024,6144,6144,bf16,decode,128,96,128,2,0,0,0,1,,,,0
1280,6144,6144,bf16,4wave_nt,256,128,128,2,2,1,1,1,,
65536,6144,6144,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
