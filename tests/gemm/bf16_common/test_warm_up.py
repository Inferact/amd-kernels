# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 through the latent down entry (a fresh process and FlyDSL cache): the first call warms the routed package
up, every served row count then compiles nothing, and a CUDA graph of the entry call imports and allocates nothing
and replays the eager bits."""
import importlib
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.gemm.bf16_common import (bf16_latent_down_mxfp4_quant, bf16_latent_down_mxfp4_quant_shapes,
                                              latent_down)

    warm_up.set_auto(True)
    result = {}
    for key in sorted(bf16_latent_down_mxfp4_quant_shapes()):
        n, k = key
        host = latent_down._ROUTES.host(key)
        w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
        bf16_latent_down_mxfp4_quant(torch.zeros(16, k, dtype=torch.bfloat16, device="cuda"), w)
        torch.cuda.synchronize()
        warns = []
        compile_watch.arm(warns.append)
        for m in range(1, host.MAX_M + 1):
            bf16_latent_down_mxfp4_quant(torch.randn(m, k, device="cuda").bfloat16(), w)
        torch.cuda.synchronize()
        graph = []
        s = torch.cuda.Stream()
        for m in (1, host.MAX_M):
            x = torch.randn(m, k, device="cuda").bfloat16()
            codes = torch.empty(m, n // 2, dtype=torch.uint8, device="cuda")
            scales = torch.empty(m, n // 32, dtype=torch.uint8, device="cuda")
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                eager = [t.clone() for t in bf16_latent_down_mxfp4_quant(x, w, (codes, scales))]
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            codes.zero_()
            scales.zero_()
            real = importlib.import_module
            importlib.import_module = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("import in capture"))
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    allocs = torch.cuda.memory_stats()["allocation.all.allocated"]
                    bf16_latent_down_mxfp4_quant(x, w, (codes, scales))
                    new_allocs = torch.cuda.memory_stats()["allocation.all.allocated"] - allocs
            finally:
                importlib.import_module = real
            g.replay()
            torch.cuda.synchronize()
            graph.append({"m": m, "allocs": new_allocs,
                          "equal": torch.equal(codes, eager[0]) and torch.equal(scales, eager[1])})
        result[str(key)] = {"late": compile_watch.late_compiles(), "warns": warns[:2],
                            "entry_late": warm_up.late_compiles(), "graph": graph}
    return result


def test_warm_up_then_no_compile_and_graph(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    for key, r in result.items():
        assert r["late"] == 0 and r["entry_late"] == 0, (key, r)
        assert all(g["allocs"] == 0 and g["equal"] for g in r["graph"]), (key, r)


if __name__ == "__main__":
    print(json.dumps(scenario()))
