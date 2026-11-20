# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``warm_up_mxfp4_moe(w, partials=True, sp=True)`` on the full shared expert copy in a fresh process and an empty
FlyDSL cache: the BF16 entry's view variants (both modes), the preparation, the routed-only partials and
``shared_expert_full``; afterwards the served row counts compile nothing and no warm-up launched a kernel."""
import json
import os
import subprocess
import sys

import pytest

BF16_ROWS = (1, 8, 16, 17, 33, 95, 96, 300, 640, 641, 1024, 2049, 3583, 3584, 4097, 8191, 16385)
SP_ROWS = (2048, 2049, 4096, 4097, 8192, 8885, 16384, 16385, 31490, 32768)
FULL_ROWS = (1, 1024, 1280, 1281, 2048, 2560, 2561, 4096, 4097, 7873, 16384)


def scenario():
    import torch
    from flydsl.compiler.jit_function import CompiledFunction, MlirCompiler

    from amd_kernels import warm_up
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import (MoeWeights, SharedWeights, mxfp4_moe, prepare_mxfp4_moe,
                                                            shared_expert_full, warm_up_mxfp4_moe)
    from .workload import H, make_activation, make_experts, make_full_shared

    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    torch.set_num_threads(8)
    raw, _ = make_experts()
    full, _ = make_full_shared()
    w = MoeWeights.from_row_major(torch.randn(128, H, device="cuda") / H**0.5, torch.zeros(128, device="cuda"), *raw,
                                  shared=SharedWeights.from_row_major(*full, tp_rank=1))
    in_warm_up = []
    launch = CompiledFunction.__call__

    def spy_launch(self, *args):
        if warm_up._in_warm_up.get():
            in_warm_up.append(1)
        return launch(self, *args)

    CompiledFunction.__call__ = spy_launch
    warm_up_mxfp4_moe(w, partials=True, sp=True)
    x = torch.zeros((4096, H), device="cuda", dtype=torch.bfloat16)
    prepare_mxfp4_moe(x, w)  # the preparation's own entry (the hand-offs warm it in their shard compute)
    CompiledFunction.__call__ = launch
    count = []
    original = MlirCompiler.compile

    def spy(self, *args, **kw):
        if warm_up._in_warm_up.get() or warm_up._current.get() is not None:
            count.append(1)
        return original(self, *args, **kw)

    MlirCompiler.compile = spy
    late = []
    for m in BF16_ROWS:
        x = make_activation(m, seed=m)
        mxfp4_moe(x, w)
        mxfp4_moe(x, w, return_partials=True)
        torch.cuda.synchronize()
        if count:
            late.append(("bf16", m))
            count.clear()
    for m in SP_ROWS:
        p = prepare_mxfp4_moe(make_activation(m, seed=m), w)
        mxfp4_moe(p.routed(), w, return_partials=True)
        torch.cuda.synchronize()
        if count:
            late.append(("routed", m))
            count.clear()
    for m in FULL_ROWS:
        p = prepare_mxfp4_moe(make_activation(m, seed=m), w)
        shared_expert_full(p.shared_codes, p.shared_scales, w.shared_weights)
        torch.cuda.synchronize()
        if count:
            late.append(("full", m))
            count.clear()
    modes = sorted(str(m) for name, m in warm_up.warmed())
    return {"late": late, "launches_in_warm_up": len(in_warm_up), "modes": modes,
            "late_reported": warm_up.late_compiles()}


def test_sp_full_shared_warm_up(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=2400,
                          env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                               "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert not result["late"] and result["late_reported"] == 0
    assert result["launches_in_warm_up"] == 0
    assert any("routed" in m and "full" in m for m in result["modes"]), result["modes"]
    assert any("'bf16', 'full'" in m for m in result["modes"]), result["modes"]


if __name__ == "__main__":
    print(json.dumps(scenario()))
