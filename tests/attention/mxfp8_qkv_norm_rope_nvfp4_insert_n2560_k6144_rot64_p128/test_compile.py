# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after the entry's first-call warm-up (root README, rule 4): in a fresh process the first call
warms every variant up (``host._sizes``: every token count 1-1536, compile-only), then calls at 1-288 rows and at
the delegated sizes 289-1536 compile nothing."""

import json
import os
import subprocess
import sys

import pytest

SIZES = list(range(1, 289)) + [289, 320, 384, 449, 512, 513, 768, 1024, 1025, 1280, 1536]


def scenario():
    import torch
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels import warm_up
    from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128 import (
        mxfp8_qkv_norm_rope_nvfp4_insert)

    from .test_qkv_insert import KW, make, outputs, weights

    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    warm_up.set_workers(1)
    count = []
    original = MlirCompiler.compile

    def spy(self, *args, **kw):
        if warm_up._in_warm_up.get() or warm_up._current.get() is not None:
            count.append(1)
        return original(self, *args, **kw)

    MlirCompiler.compile = spy
    _, (w, ws), _, _ = weights()

    def call(m):
        sa, _, ins = make(m, m)
        q, iq_rows = outputs(m)
        mxfp8_qkv_norm_rope_nvfp4_insert(*sa, w, ws, m, *ins, q, iq_rows.view(torch.float8_e4m3fn)[:, :128], **KW)

    call(8)  # the first call: the warm-up
    torch.cuda.synchronize()
    warm = len(count)
    late = []
    for m in SIZES:
        before = len(count)
        call(m)
        if len(count) > before:
            late.append(m)
    torch.cuda.synchronize()
    return {"warm_compiles": warm, "late": late, "late_reported": warm_up.late_compiles()}


def test_no_compile_after_warm_up(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    # an empty FlyDSL artifact cache: the warm-up itself compiles every variant
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=3000,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert result["warm_compiles"] > 0
    assert not result["late"]
    assert result["late_reported"] == 0


if __name__ == "__main__":
    print(json.dumps(scenario()))
