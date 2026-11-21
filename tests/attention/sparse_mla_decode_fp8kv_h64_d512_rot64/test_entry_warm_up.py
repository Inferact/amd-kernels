# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``sparse_mla_decode`` is an entry (``amd_kernels.warm_up``): the first call of a mode compiles its kernel; calls at
other row counts and key-list lengths compile nothing."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64 import sparse_mla_decode

    from tests.attention.sparse_mla_decode_fp8kv_h64_d512_rot64.cases import make_case, outputs, row_lengths
    from tests.attention.sparse_mla_decode_fp8kv_h64_d512_rot64.test_decode import ARGS

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    bad = []
    for qlens in ([6, 6, 4], [6] * 8, [6, 1, 3, 2, 6, 5], [6] * 32):
        before = len(late)
        case = make_case(row_lengths(qlens), seed=len(qlens))
        rows = case["q"].shape[0]
        codes, scales = outputs(rows)
        sparse_mla_decode(*[case[k] for k in ARGS], codes, scales)
        if len(late) > before:
            bad.append(rows)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_the_mode(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=900,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] >= 1, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
