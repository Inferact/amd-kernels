# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry warms every row-tile variant of a mode on its first call; any other M compiles nothing."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.gemm.bf16_latent_down_mxfp4_quant_k7168_n3584 import (bf16_latent_down_mxfp4_quant_k7168_n3584,
                                                                           shuffle_weight)
    from amd_kernels.gemm.bf16_latent_down_mxfp4_quant_k7168_n3584.reference import K, N
    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    w = shuffle_weight(torch.zeros(N, K, dtype=torch.bfloat16, device="cuda"))
    bad = []
    for with_bf16 in (False, True):
        for m in (192, 1, 4, 12, 24, 40, 65, 80, 128, 161, 190, 300):
            before = len(late)
            x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
            o = torch.empty(m, N, dtype=torch.bfloat16, device="cuda") if with_bf16 else None
            bf16_latent_down_mxfp4_quant_k7168_n3584(x, w, shuffled=True, out_bf16=o)
            if len(late) > before:
                bad.append((with_bf16, m))
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 2, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
