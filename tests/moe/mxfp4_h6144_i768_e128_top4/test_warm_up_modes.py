# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``warm_up_mxfp4_moe`` in a fresh process: the reduced mode, with ``partials`` the partials mode too, once per mode
(a repeat calls nothing), the node's default worker count; afterwards the served row counts compile nothing."""
import json
import os
import subprocess
import sys

import pytest

ENTRY = "moe/mxfp4_h6144_i768_e128_top4"
ROWS = (1, 8, 300, 2049, 8191)


def scenario():
    import torch
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels import warm_up
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, chain, warm_up_mxfp4_moe

    from .test_shared_expert import weights
    from .workload import H, make_activation, make_experts

    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    torch.set_num_threads(8)
    raw, _ = make_experts()
    w = MoeWeights.from_row_major(torch.randn(128, H, device="cuda") / H**0.5, torch.zeros(128, device="cuda"), *raw)
    _, w.shared_weights = weights.__wrapped__()

    def modes():
        return sorted(bool(m[3]) for name, m in warm_up.warmed() if name == ENTRY)

    warm_up_mxfp4_moe(w)
    reduced = modes()
    warm_up_mxfp4_moe(w, partials=True)
    both = modes()
    calls = []
    real = chain.mxfp4_moe
    chain.mxfp4_moe = lambda *a, **kw: calls.append(1) or real(*a, **kw)
    warm_up_mxfp4_moe(w, partials=True)
    chain.mxfp4_moe = real
    count = []
    original = MlirCompiler.compile

    def spy(self, *args, **kw):
        if warm_up._in_warm_up.get() or warm_up._current.get() is not None:
            count.append(1)
        return original(self, *args, **kw)

    MlirCompiler.compile = spy
    late = []
    for m in ROWS:
        x = make_activation(m, seed=m)
        real(x, w)
        real(x, w, return_partials=True)
        torch.cuda.synchronize()
        if count:
            late.append(m)
            count.clear()
    return {"reduced": reduced, "both": both, "repeat_calls": len(calls), "late": late,
            "workers": warm_up._workers, "node_workers": chain._node_workers()}


def test_warm_up_modes_once_each(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                               "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert result["reduced"] == [False] and result["both"] == [False, True]
    assert result["repeat_calls"] == 0
    assert result["workers"] == result["node_workers"]
    assert not result["late"]


if __name__ == "__main__":
    print(json.dumps(scenario()))
