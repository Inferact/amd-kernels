# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The first call of each mode warms the entries up in a fresh process (MXFP8 or bf16 rows, token or sorted-rows
entry); afterwards no served count compiles."""
import json
import os
import subprocess
import sys

import pytest

ENTRY = "moe/mxfp4_h5120_i2304_e48_top6"
ROWS = (1, 5, 17, 40, 70, 100, 150, 200, 300, 450, 600, 1024)


def scenario():
    import torch
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels import warm_up
    from amd_kernels.moe.mxfp4_h5120_i2304_e48_top6 import ExpertWeights, mxfp4_experts, mxfp4_experts_rows

    from .workload import make_activation, make_experts, make_routes, make_sorted_rows

    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    w = ExpertWeights.from_row_major(*make_experts())

    def call(entry, n, bf16, seed):
        if entry == "tokens":
            xc, xs = make_activation(n, seed=seed)
            x, xs = (torch.randn((n, 5120), device="cuda").bfloat16(), None) if bf16 else (xc, xs)
            ids, wt = make_routes(n, seed=seed)
            mxfp4_experts(x, xs, w, wt, ids)
        else:
            xc, xs = make_activation(6 * n, seed=seed)
            x, xs = (torch.randn((6 * n, 5120), device="cuda").bfloat16(), None) if bf16 else (xc, xs)
            offsets, weights = make_sorted_rows(6 * n, seed=seed)
            mxfp4_experts_rows(x, xs, w, weights, offsets)

    modes = [(entry, bf16) for entry in ("tokens", "rows") for bf16 in (False, True)]
    for entry, bf16 in modes:
        call(entry, 16, bf16, 0)
    count = []
    original = MlirCompiler.compile

    def spy(self, *args, **kw):
        if warm_up._in_warm_up.get() or warm_up._current.get() is not None:
            count.append(1)
        return original(self, *args, **kw)

    MlirCompiler.compile = spy
    late = []
    for entry, bf16 in modes:
        for n in ROWS:
            call(entry, n, bf16, n)
            torch.cuda.synchronize()
            if count:
                late.append((entry, bf16, n))
                count.clear()
    return {"modes": sorted(f"{name} {m}" for name, m in warm_up.warmed() if name.startswith(ENTRY)), "late": late}


def test_warm_up_covers_served_counts(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=3000,
                          env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                               "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert result["modes"] == [f"{ENTRY} (False, False, 48)", f"{ENTRY} (True, False, 48)",
                               f"{ENTRY}_rows (False, 48)", f"{ENTRY}_rows (True, 48)"]
    assert not result["late"]


if __name__ == "__main__":
    print(json.dumps(scenario()))
