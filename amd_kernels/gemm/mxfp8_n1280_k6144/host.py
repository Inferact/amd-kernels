# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 1280 x 6144 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``)."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 1280, 6144

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
80,1280,6144,bf16,decode,16,64,128,1,0,0,0,8,,,65,0
128,1280,6144,bf16,decode,32,32,128,1,0,0,0,8,,,,0
192,1280,6144,bf16,decode,48,64,128,1,0,0,0,6,,,,0
256,1280,6144,bf16,decode,48,64,128,1,0,0,0,4,,,,0
512,1280,6144,bf16,decode,80,32,128,1,0,0,0,1,,,,0
1024,1280,6144,bf16,decode,176,32,128,1,0,0,0,1,,
2048,1280,6144,bf16,aiter,32,128,128,0,0,2,4,1,,
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
