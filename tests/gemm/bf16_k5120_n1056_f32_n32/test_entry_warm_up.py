# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry (``amd_kernels.warm_up``) compiles every M tile of its kernel on its first call (one mode per column split);
calls at other row counts compile nothing."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.gemm.bf16_k5120_n1056_f32_n32 import gemm, shuffle_weight

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    bad = []
    x = torch.randn((1024, 5120), dtype=torch.bfloat16, device="cuda")
    for na in (1024, 512, 0):
        w = shuffle_weight(torch.randn((na, 5120), dtype=torch.bfloat16, device="cuda") if na else None,
                           torch.randn((32, 5120), dtype=torch.bfloat16, device="cuda"))
        out_a = torch.empty((1024, na), dtype=torch.float32, device="cuda") if na else None
        out_b = torch.empty((1024, 32), dtype=torch.bfloat16, device="cuda")
        for n in (1, 3, 8, 21, 64, 150, 300, 768, 769, 1024):
            before = len(late)
            gemm(x[:n], w, None if out_a is None else out_a[:n], out_b[:n])
            if len(late) > before:
                bad.append(n)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=900,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 3, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
