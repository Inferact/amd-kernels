# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry (``amd_kernels.warm_up``) compiles its kernel on its first call (one mode per cos | sin cache dtype); calls
at other row counts compile nothing."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.indexer.q_rope_mxfp4_quant_h32_d128_rot64 import q_rope_mxfp4_quant

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    from tests.indexer.q_rope_mxfp4_quant_h32_d128_rot64.test_q_rope_mxfp4_quant import make

    bad = []
    for dtype in (torch.float32, torch.bfloat16):
        q, pos, cs, w = make(768, dtype)
        for n in (1, 3, 8, 21, 64, 192, 768):
            before = len(late)
            q_rope_mxfp4_quant(q[:n], pos[:n], cs, w[:n], 0.0625)
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
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 2, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
