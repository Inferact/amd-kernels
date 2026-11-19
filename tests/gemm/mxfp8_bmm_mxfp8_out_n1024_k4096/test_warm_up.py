# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``mxfp8_bmm_mxfp8_out`` is an entry (``amd_kernels.warm_up``): its first call compiles every tile config;
calls at the CUDA graph batch sizes and a prefill's row counts then compile nothing."""

import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels import warm_up
    from amd_kernels.gemm.mxfp8_bmm_mxfp8_out_n1024_k4096 import mxfp8_bmm_mxfp8_out

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    g, n, k = 8, 1024, 4096
    w = torch.zeros((g, n, k), dtype=torch.float8_e4m3fn, device="cuda")
    ws = torch.zeros((g, n // 32, k // 32), dtype=torch.uint8, device="cuda")
    bad = []
    for m in (*range(1, 513, 7), 1000, 1536, 2047, 4096, 9000, 16384):
        before = len(late)
        x = torch.zeros((m, g * k), dtype=torch.float8_e4m3fn, device="cuda")
        xs = torch.zeros((m, g * k // 32), dtype=torch.uint8, device="cuda")
        mxfp8_bmm_mxfp8_out(x, xs, w, ws)
        if len(late) > before:
            bad.append(m)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 1, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
