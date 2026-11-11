# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode block's warm-up (``IndexDecodeTop16`` built + ``index_decode.warm_up``) compiles every variant the
block launches on one GPU: afterwards no step of 1-64 requests (and 65-256 for the attention) at 32K / 192K / 1M, bf16
and MXFP8 output, bf16 and fp16 compute, compiles."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario(ql, compute):
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import index_decode as blk
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.decode_top16 import IndexDecodeTop16

    torch.set_num_threads(8)
    warm_up.set_auto(True)
    warm_up.set_workers(1)
    count = []
    original = MlirCompiler.compile

    def spy(*a, **kw):
        count.append(1)
        return original(*a, **kw)

    MlirCompiler.compile = spy
    dev, hmax, max_reqs = torch.device("cuda"), 1 << 20, 256
    index = IndexDecodeTop16(dev, query_len=ql, max_reqs=max_reqs, max_seq_len=hmax)
    blk.warm_up(dev, num_heads=16, num_kv_heads=1, query_len=ql, max_reqs=max_reqs, quant_min_rows=8,
                compute_type=compute)
    warm = len(count)
    icache = torch.zeros(2, 128, 72, device=dev, dtype=torch.uint8)
    kv = torch.zeros(2, 2, 128, 72, device=dev, dtype=torch.uint8)
    late = []
    for n in list(range(1, 65)) + [65, 100, 128, 129, 200, 256]:
        rows = n * ql
        for ctx in ((32768, 196608, hmax) if n in (1, 2, 3, 4, 5, 8, 16, 32, 33, 64, 128, 256) else (hmax,)):
            for quant in (False, True):
                if quant and rows < 8:
                    continue
                iq = torch.zeros(rows, 1, 128, device=dev, dtype=torch.float8_e4m3fn)
                q = torch.zeros(rows, 16, 128, device=dev, dtype=torch.bfloat16)
                table = torch.zeros(n, ctx // 128, device=dev, dtype=torch.int32)
                seq = torch.full((n,), ctx, device=dev, dtype=torch.int32)
                ids = torch.empty(1, rows, 16, device=dev, dtype=torch.int32)
                before = len(count)
                blk.nvfp4_index_sparse_decode(index, iq, icache, table, seq, 0, 1, q, kv, ids, table, seq, 128**-0.5,
                                              torch.empty_like(q), k_scale=None, v_scale=None, decode_query_len=ql,
                                              max_seq_len=ctx, compute_type=compute,
                                              quant_out=host._quant_buffers(rows, 2048, dev) if quant else None)
                torch.cuda.synchronize()
                if len(count) != before:
                    late.append((n, ctx, quant))
    return dict(warm=warm, late=late, reported=warm_up.late_compiles())


@pytest.mark.parametrize("compute", ["bfloat16", "float16"])
@pytest.mark.parametrize("ql", [8, 4])
def test_block_warm_up_then_no_compile(ql, compute, tmp_path):
    from .test_sparse_decode import require_gfx950

    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__, str(ql), compute], capture_output=True, text=True,
                          timeout=2400,
                          env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                               "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["warm"] > 0 and not result["late"] and result["reported"] == 0, result


if __name__ == "__main__":
    print(json.dumps(scenario(int(sys.argv[1]), getattr(torch, sys.argv[2]))))
