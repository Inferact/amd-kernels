# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``amd_kernels.compile_watch``: an armed watch reports every amd_kernels FlyDSL compile once, with its frame, and
nothing for a launch whose variant is already compiled. Runs in a fresh interpreter with FlyDSL's disk cache off, so
the first launch really compiles."""
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch


def scenario() -> dict:
    from amd_kernels import compile_watch
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm

    reports = []
    assert compile_watch.arm(reports.append)
    x = torch.zeros((32, 6144), dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(6144, dtype=torch.bfloat16, device="cuda")
    add_rmsnorm.gemma_add_rmsnorm_mxfp8_row_major(x, x, w, 1e-6)  # compiles: reported
    first = list(reports)
    add_rmsnorm.gemma_add_rmsnorm_mxfp8_row_major(x, x, w, 1e-6)  # compiled: silent
    torch.cuda.synchronize()
    count = compile_watch.late_compiles()
    compile_watch.disarm()
    add_rmsnorm.warmup_add_rmsnorm_mxfp8()  # disarmed: silent
    return {"first": first, "after": reports[len(first):], "count": count, "final": len(reports)}


def test_reports_amd_kernels_compiles_once():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    root = pathlib.Path(__file__).resolve().parents[1]
    env = dict(os.environ, FLYDSL_RUNTIME_ENABLE_CACHE="0")
    proc = subprocess.run([sys.executable, "-m", "tests.test_compile_watch"], cwd=root, env=env,
                          capture_output=True, text=True, timeout=900)  # fmt: skip
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    assert r["first"], "the first launch compiled but was not reported"
    assert all("amd_kernels" in m and "/norm/gemma_add_rmsnorm_h6144/" in m for m in r["first"]), r["first"]
    assert r["after"] == [] and r["count"] == len(r["first"]) and r["final"] == len(r["first"]), r


if __name__ == "__main__":
    print(json.dumps(scenario()))
