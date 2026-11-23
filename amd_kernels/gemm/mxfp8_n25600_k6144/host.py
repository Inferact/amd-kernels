# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 25600 x 6144 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``)."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 25600, 6144

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
16,25600,6144,any,aiter,32,128,256,0,0,0,4,2,,
32,25600,6144,any,aiter,32,128,256,0,0,0,4,2,,
64,25600,6144,any,aiter,64,64,256,0,0,2,0,1,,
128,25600,6144,any,aiter,128,128,256,0,0,0,0,1,,
192,25600,6144,any,aiter,128,128,256,0,0,0,4,1,,
256,25600,6144,any,aiter,128,128,256,0,0,2,4,1,,
384,25600,6144,any,aiter,128,256,128,0,0,0,4,1,,
512,25600,6144,any,aiter,128,128,256,0,0,2,4,1,,
768,25600,6144,any,aiter,128,128,128,0,0,0,4,1,,
16,25600,6144,bf16,decode,16,64,128,2,0,0,0,1,,
32,25600,6144,bf16,decode,32,64,128,2,0,0,0,1,,
64,25600,6144,bf16,decode,64,64,128,1,0,0,0,1,,
128,25600,6144,bf16,decode,128,64,128,2,0,0,0,1,,
192,25600,6144,bf16,staged_ab,128,256,128,2,4,2,4,1,,
256,25600,6144,bf16,staged_ab,128,256,128,2,4,2,4,1,,
384,25600,6144,bf16,aiter,128,256,128,0,0,0,4,1,,
512,25600,6144,bf16,staged_ab,128,256,128,2,4,2,4,1,,
768,25600,6144,bf16,aiter,128,128,128,0,0,0,4,1,,
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
