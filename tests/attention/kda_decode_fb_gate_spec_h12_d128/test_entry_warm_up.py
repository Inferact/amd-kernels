# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry (``amd_kernels.warm_up``) compiles its kernel on the first call of a mode (tokens per sequence, norm weight
dtype); calls at every served row count after that compile nothing."""
import json
import os
import subprocess
import sys

import pytest
import torch

SERVED = [1, 6, 12, 24, 40, 65, 80, 128, 192]


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.attention.kda_decode_fb_gate_spec_h12_d128 import kda_decode_fb_gate_spec

    from tests.attention.kda_decode_fb_gate_spec_h12_d128.inputs import make_inputs

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    bad = []
    for rows in SERVED:
        before = len(late)
        kda_decode_fb_gate_spec(**make_inputs(rows, seed=rows))
        if len(late) > before:
            bad.append(rows)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=900,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 1, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
