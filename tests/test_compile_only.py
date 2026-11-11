# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``tests/compile_only.py``: in compile-only mode a FlyDSL kernel compiles once and runs never (a subprocess with
``AKR_COMPILE_ONLY=1``; its output stays as the test filled it)."""

import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch


def scenario():
    from amd_kernels import compile_watch
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an

    reports = []
    assert compile_watch.arm(reports.append)
    x = torch.randn((4, 6144), device="cuda").bfloat16()
    w = torch.randn(6144, device="cuda").bfloat16()
    res = torch.full((4, 6144), float("nan"), dtype=torch.bfloat16, device="cuda")
    an.gemma_add_rmsnorm_mxfp8_row_major(x, x, w, 1e-6, res_out=res)
    first = len(reports)
    an.gemma_add_rmsnorm_mxfp8_row_major(x, x, w, 1e-6, res_out=res)
    torch.cuda.synchronize()
    return {"first": first, "again": len(reports) - first, "ran": not bool(res.isnan().all())}


@pytest.mark.parametrize("compile_only", [False, True])
def test_compiles_once_and_runs_only_outside_the_mode(compile_only):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    root = pathlib.Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in ("AKR_COMPILE_ONLY", "COMPILE_ONLY")}
    if compile_only:
        env["AKR_COMPILE_ONLY"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "tests.test_compile_only"], cwd=root, capture_output=True, text=True, timeout=600,
        env={**env, "FLYDSL_RUNTIME_ENABLE_CACHE": "0"},
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {"first": 1, "again": 0, "ran": not compile_only}, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
