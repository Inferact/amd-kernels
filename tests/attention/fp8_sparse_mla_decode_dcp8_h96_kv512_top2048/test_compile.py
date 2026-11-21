# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

Rows, strides, the key counts and the scale are runtime arguments; the row count picks one of two kernels. Warm up both q
modes (bf16 q, pre-quantized q8), then walk every served row count (the 7 critical sizes and the CUDA-graph sizes up
to 192) and assert nothing compiles.
"""

import flydsl.compiler as flyc
import torch
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 import fp8_sparse_mla_decode
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.reference import quantize_q

from .cases import GRAPH_ROWS, SERVED_ROWS, SM_SCALE, make_batch, require_gfx950, served_counts


@torch.inference_mode()
def test_no_compile_after_warmup(monkeypatch):
    require_gfx950()
    gen = torch.Generator().manual_seed(3)
    q, cache, indptr, slots = make_batch(served_counts(192, gen), gen)
    q8 = quantize_q(q)
    for rows in (1, 49):  # one call per kernel (the row count picks it)
        fp8_sparse_mla_decode(q[:rows], cache, indptr[: rows + 1], slots[: int(indptr[rows])], SM_SCALE)
        fp8_sparse_mla_decode(None, cache, indptr[: rows + 1], slots[: int(indptr[rows])], SM_SCALE, q8=q8[:rows])
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile

    def counting(fn, *args, **kwargs):
        compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    for rows in sorted(set(SERVED_ROWS) | set(GRAPH_ROWS)):
        fp8_sparse_mla_decode(q[:rows], cache, indptr[: rows + 1], slots[: int(indptr[rows])], SM_SCALE)
        fp8_sparse_mla_decode(None, cache, indptr[: rows + 1], slots[: int(indptr[rows])], SM_SCALE, q8=q8[:rows])
        assert not compiled, f"{rows} rows compiled {compiled}"
    torch.cuda.synchronize()
