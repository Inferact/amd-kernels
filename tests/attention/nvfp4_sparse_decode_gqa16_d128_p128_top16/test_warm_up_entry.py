# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``host.warm_up_decode`` (and the entry's first-call warm-up, ``amd_kernels.warm_up``) compiles every variant a TP4
server launches, bf16 and fp16 compute: afterwards no request count of 1-64 (bf16 output, and the MXFP8 output from 8
rows) compiles. Also the first-call warm-up alone: one call, then every count to ``WARM_MAX_REQS`` without a
compile."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario(ql, explicit, compute):
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host

    torch.set_num_threads(8)
    warm_up.set_auto(True)
    warm_up.set_workers(1)
    count = []
    original = MlirCompiler.compile

    def spy(*a, **kw):
        count.append(1)
        return original(*a, **kw)

    MlirCompiler.compile = spy
    dev = torch.device("cuda")
    heads, hkv, max_reqs = 16, 1, 64
    walk = range(1, max_reqs + 1) if explicit else (1, 2, 7, 33, 64, 100, 255, 256)
    kv = torch.zeros((2, 2 * hkv, 128, 72), dtype=torch.uint8, device=dev)

    def call(reqs, quant):
        rows = reqs * ql
        q = torch.zeros((rows, heads, 128), dtype=torch.bfloat16, device=dev)
        idx = torch.full((hkv, rows, 16), -1, dtype=torch.int32, device=dev)
        idx[..., 0] = 0
        seq = torch.full((reqs,), 4096, dtype=torch.int32, device=dev)
        bt = torch.zeros((reqs, 32), dtype=torch.int32, device=dev)
        host.nvfp4_sparse_decode(q, kv, idx, bt, seq, 128**-0.5, torch.empty_like(q), decode_query_len=ql,
                                 quant_out=host._quant_buffers(rows, heads * 128, dev) if quant else None,
                                 compute_type=compute)
        torch.cuda.synchronize()

    if explicit:
        host.warm_up_decode(dev, num_heads=heads, num_kv_heads=hkv, query_len=ql, max_reqs=max_reqs, quant_min_rows=8,
                            compute_type=compute)
    else:
        call(1, False)  # the first call: its warm-up
    warm = len(count)
    late = []
    for quant in (False, True):
        if not explicit and quant:
            continue
        for reqs in walk:
            if quant and reqs * ql < 8:
                continue
            before = len(count)
            call(reqs, quant)
            if len(count) != before:
                late.append((reqs, quant))
    return dict(warm=warm, late=late, reported=warm_up.late_compiles())


@pytest.mark.parametrize("compute", ["bfloat16", "float16"])
@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize("explicit", [True, False], ids=["warm_up_decode", "first_call"])
def test_warm_up_then_no_compile(ql, explicit, compute, tmp_path):
    from .test_sparse_decode import require_gfx950

    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__, str(ql), str(int(explicit)), compute], capture_output=True,
                          text=True, timeout=1500,
                          env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                               "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["warm"] > 0 and not result["late"] and result["reported"] == 0, result


if __name__ == "__main__":
    print(json.dumps(scenario(int(sys.argv[1]), bool(int(sys.argv[2])), getattr(torch, sys.argv[3]))))
