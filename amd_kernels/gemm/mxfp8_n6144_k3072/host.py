# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 6144 x 3072 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``)."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 6144, 3072

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
1,6144,3072,any,aiter,32,64,128,0,0,0,0,4,,
4,6144,3072,any,aiter,32,64,128,0,0,0,0,4,,
16,6144,3072,any,aiter,32,64,128,0,0,0,0,4,,
64,6144,3072,any,aiter,64,64,256,0,0,0,0,2,,
128,6144,3072,any,aiter,64,64,256,0,0,2,4,1,,
192,6144,3072,any,aiter,64,64,256,0,0,2,4,1,,
256,6144,3072,any,aiter,128,128,256,0,0,2,4,1,,
320,6144,3072,any,aiter,128,64,256,0,0,2,4,1,,
384,6144,3072,any,aiter,128,64,256,0,0,2,4,1,,
512,6144,3072,any,aiter,128,64,256,0,0,2,4,1,,
640,6144,3072,any,aiter,128,64,256,0,0,2,4,1,,
768,6144,3072,any,aiter,64,128,256,0,0,2,4,1,,
1024,6144,3072,any,aiter,128,128,256,0,0,2,4,1,,
4096,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
8192,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
8224,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
10240,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
12288,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
14336,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
16384,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
18432,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
20480,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
22528,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
24576,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
26624,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
28672,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
30720,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
32768,6144,3072,any,aiter,128,256,128,0,0,2,4,1,,
8,6144,3072,bf16,decode_fq,8,32,128,2,0,0,0,1,,
16,6144,3072,bf16,decode_fq,16,32,128,2,0,0,0,1,,
32,6144,3072,bf16,decode,32,32,128,1,0,0,0,1,,
64,6144,3072,bf16,decode,64,32,128,1,0,0,0,1,,
80,6144,3072,bf16,decode,48,48,128,3,0,0,0,1,,
128,6144,3072,bf16,decode,64,48,128,3,0,0,0,1,,
192,6144,3072,bf16,decode,96,48,128,3,0,0,0,1,,
224,6144,3072,bf16,decode,112,48,128,3,0,0,0,1,,
256,6144,3072,bf16,decode,128,48,128,3,0,0,0,1,,
320,6144,3072,bf16,decode,96,96,128,4,0,0,0,1,,
384,6144,3072,bf16,decode,96,96,128,3,0,0,0,1,,
512,6144,3072,bf16,decode,128,96,128,3,0,0,0,1,,
640,6144,3072,bf16,staged_ab,128,128,128,2,4,2,4,1,,
768,6144,3072,bf16,decode,96,96,128,1,0,0,0,1,,
1280,6144,3072,bf16,4wave,256,128,128,2,2,1,1,1,,
65536,6144,3072,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
