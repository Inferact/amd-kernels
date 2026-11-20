# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 2304 x 6144 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``)."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 2304, 6144

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
1,2304,6144,any,aiter,32,64,128,0,0,0,0,12,,
4,2304,6144,any,aiter,32,64,128,0,0,0,0,12,,
16,2304,6144,any,aiter,32,64,128,0,0,0,0,12,,
64,2304,6144,any,aiter,64,64,256,0,0,0,0,4,,
256,2304,6144,any,aiter,64,64,256,0,0,0,0,4,,
1024,2304,6144,any,aiter,128,128,256,0,0,2,4,1,,
4096,2304,6144,any,aiter,128,128,256,0,0,0,0,1,,
8192,2304,6144,any,aiter,128,128,256,0,0,0,0,1,,
8224,2304,6144,any,aiter,128,256,128,0,0,2,4,1,,
10240,2304,6144,any,aiter,128,128,256,0,0,0,0,1,,
12288,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
14336,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
16384,2304,6144,any,aiter,128,256,128,0,0,2,4,1,,
18432,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
20480,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
22528,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
24576,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
26624,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
28672,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
30720,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
32768,2304,6144,any,aiter,128,256,128,0,0,0,0,1,,
8,2304,6144,bf16,decode_fq,8,16,128,3,0,0,0,1,,
16,2304,6144,bf16,decode_fq,8,32,128,2,0,0,0,1,,
32,2304,6144,bf16,decode_fq,8,48,128,3,0,0,0,1,,
96,2304,6144,bf16,decode,32,32,128,1,0,0,0,1,,
112,2304,6144,bf16,decode,32,32,128,2,0,0,0,1,,,,0
160,2304,6144,bf16,decode,32,48,128,2,0,0,0,1,,
192,2304,6144,bf16,decode,64,32,128,4,0,0,0,1,,
224,2304,6144,bf16,decode,80,32,128,3,0,0,0,1,,
320,2304,6144,bf16,decode,64,48,128,3,0,0,0,1,,
384,2304,6144,bf16,decode,128,32,128,3,0,0,0,1,,
448,2304,6144,bf16,decode,64,64,128,3,0,0,0,1,,,,0
512,2304,6144,bf16,decode,128,48,128,3,0,0,0,1,,
896,2304,6144,bf16,aiter,32,128,256,0,0,2,4,1,,,769
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
