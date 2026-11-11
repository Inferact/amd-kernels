# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``host.warm_up_prefill`` compiles every prefill variant of a compute precision at
one KV head (TP4): afterwards no token count across the tiers compiles."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario(compute):
    lazy = compute == "fp16-lazy"
    compute = "fp16" if lazy else compute
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import host

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
    heads, hkv = 16, 1
    host.warm_up_prefill(dev, num_heads=heads, num_kv_heads=hkv, compute_dtype=compute, lazy_rescale=lazy)
    warm = len(count)
    kv = torch.zeros((2, 2 * hkv, 128, 72), dtype=torch.uint8, device=dev)
    late = []
    for n in (1, 255, 256, 257, 511, 512, 513, 1024, 1025, 1026, 1027, 1087, 1088, 1536, 1537, 4096, 16384):
        q = torch.zeros((n, heads, 128), dtype=torch.bfloat16, device=dev)
        idx = torch.full((hkv, n, 16), -1, dtype=torch.int32, device=dev)
        idx[..., 0] = 0
        zeros = torch.zeros((n,), dtype=torch.int32, device=dev)
        before = len(count)
        host.nvfp4_sparse_prefill(q, kv, idx, torch.zeros((1, 1), dtype=torch.int32, device=dev), zeros, zeros, hkv,
                                  128**-0.5, torch.empty_like(q), compute_dtype=compute, lazy_rescale=lazy)
        torch.cuda.synchronize()
        if len(count) != before:
            late.append(n)
    return dict(warm=warm, late=late, reported=warm_up.late_compiles())


@pytest.mark.parametrize("compute", ["fp16", "bf16", "fp8", "fp16-lazy"])
def test_warm_up_then_no_compile(compute, tmp_path):
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")
    proc = subprocess.run([sys.executable, "-m", __name__, compute], capture_output=True, text=True, timeout=1500,
                          env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                               "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["warm"] > 0 and not result["late"] and result["reported"] == 0, result


if __name__ == "__main__":
    print(json.dumps(scenario(sys.argv[1])))
