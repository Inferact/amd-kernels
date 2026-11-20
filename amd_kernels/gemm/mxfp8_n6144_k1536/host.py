# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The rows of the 6144 x 1536 MXFP8 GEMM (``gemm.mxfp8_common`` reads them through ``select``). Every bf16 row
measured (benchmarks/gemm/mxfp8_common/sweep_rows.py)."""

import torch

from amd_kernels.gemm.mxfp8_common.gemm import _Config, parse_rows, select_from_rows

N, K = 6144, 1536

ROWS = parse_rows("""\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
8,6144,1536,bf16,decode_fq,8,32,128,3,0,0,0,1,,
16,6144,1536,bf16,decode_fq,16,32,128,3,0,0,0,1,,
32,6144,1536,bf16,decode_fq,32,32,128,3,0,0,0,1,,
48,6144,1536,bf16,decode,48,32,128,1,0,0,0,1,,
64,6144,1536,bf16,decode,32,48,128,1,0,0,0,1,,
96,6144,1536,bf16,decode,48,48,128,3,0,0,0,1,,
128,6144,1536,bf16,decode,32,64,128,1,0,0,0,1,,,,0
160,6144,1536,bf16,decode,80,48,128,3,0,0,0,1,,
192,6144,1536,bf16,decode,48,64,128,1,0,0,0,1,,,,0
256,6144,1536,bf16,decode,64,64,128,1,0,0,0,1,,,,0
288,6144,1536,bf16,decode,144,48,128,3,0,0,0,1,,
320,6144,1536,bf16,decode,80,96,128,2,0,0,0,1,,
384,6144,1536,bf16,decode,96,96,128,3,0,0,0,1,,
448,6144,1536,bf16,decode,64,96,128,2,0,0,0,1,,,,0
512,6144,1536,bf16,decode,128,96,128,2,0,0,0,1,,
640,6144,1536,bf16,decode,160,96,128,2,0,0,0,1,,
768,6144,1536,bf16,decode,112,96,128,2,0,0,0,1,,,,0
896,6144,1536,bf16,decode,112,96,128,2,0,0,0,1,,,,0
1024,6144,1536,bf16,4wave_nt,256,128,128,2,2,1,1,1,,
1280,6144,1536,bf16,4wave_nt,128,256,128,2,2,1,1,1,,
65536,6144,1536,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
""")

LAUNCHERS = {}  # the common kernels only


def select(m: int, out_dtype: torch.dtype, allow_decode: bool) -> _Config | None:
    return select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)
