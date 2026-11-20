# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 1536 x 6144 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``). The bf16 rows at 1-256
are measured; the ``any`` rows are AITER's defaults."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 1536, 6144

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
1,1536,6144,any,aiter,32,64,256,0,0,2,0,8,,
4,1536,6144,any,aiter,32,64,256,0,0,2,0,8,,
16,1536,6144,any,aiter,32,64,256,0,0,2,0,8,,
64,1536,6144,any,aiter,64,64,256,0,0,0,0,4,,
256,1536,6144,any,aiter,64,64,256,0,0,0,0,4,,
1024,1536,6144,any,aiter,128,128,256,0,0,0,0,2,,
4096,1536,6144,any,aiter,128,128,256,0,0,0,0,1,,
8192,1536,6144,any,aiter,128,128,128,0,0,0,0,1,,
8224,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
10240,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
12288,1536,6144,any,aiter,128,128,128,0,0,2,4,1,,
14336,1536,6144,any,aiter,128,128,128,0,0,2,4,1,,
16384,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
18432,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
20480,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
22528,1536,6144,any,aiter,128,128,128,0,0,2,4,1,,
24576,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
26624,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
28672,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
30720,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
32768,1536,6144,any,aiter,128,256,128,0,0,2,4,1,,
4,1536,6144,bf16,decode_fq,4,16,128,3,0,0,0,1,,
8,1536,6144,bf16,decode_fq,8,16,128,3,0,0,0,1,4,2
16,1536,6144,bf16,decode_fq,16,32,128,2,0,0,0,3,3,4
32,1536,6144,bf16,decode,16,16,128,3,0,0,0,1,16,4
64,1536,6144,bf16,decode,16,16,128,1,0,0,0,1,,,,0
80,1536,6144,bf16,decode,16,32,128,3,0,0,0,1,,
96,1536,6144,bf16,decode,32,16,128,1,0,0,0,1,,,,0
128,1536,6144,bf16,decode,16,32,128,1,0,0,0,1,,,,0
192,1536,6144,bf16,decode,32,32,128,1,0,0,0,1,,,,0
256,1536,6144,bf16,decode,64,32,128,4,0,0,0,1,12,6,,0
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
