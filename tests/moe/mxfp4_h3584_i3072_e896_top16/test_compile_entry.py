# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Count every compiler invocation after the first-call warm-up (amd_kernels.warm_up) in a fresh process."""
import json
import os
import subprocess
import sys

import pytest

SIZES = [1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 65, 68, 80, 96, 128, 160, 192]


def scenario(workers=None):
    import torch
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.moe.mxfp4_h3584_i3072_e896_top16 import MoeWeights, moe_sort, mxfp4_moe_experts, quantize_activations
    from .workload import make_activation, make_checkpoint, make_routing

    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    warm_up.set_workers(workers)
    w = MoeWeights.from_checkpoint(*make_checkpoint())
    count = []
    original = MlirCompiler.compile

    def spy(self, *args, **kw):
        if warm_up._in_warm_up.get() or warm_up._current.get() is not None:
            count.append(1)
        return original(self, *args, **kw)

    MlirCompiler.compile = spy

    def call(m, fused):
        x = make_activation(m, seed=m)
        ids, tw = make_routing(m, "zipf")
        codes, scales = quantize_activations(x)
        return mxfp4_moe_experts(codes, scales, w, moe_sort(ids, tw), fused=fused)

    for fused in (True, False):  # each mode's first call
        call(1, fused)
    torch.cuda.synchronize()
    warm = len(count)
    late = []
    for m in SIZES:
        before = len(count)
        for fused in (True, False):
            call(m, fused)
        if len(count) > before:
            late.append(m)
    torch.cuda.synchronize()
    return {"warm_compiles": warm, "late": late, "warmed": len(warm_up.warmed()), "late_reported": warm_up.late_compiles()}


@pytest.mark.parametrize("workers", [1, None], ids=["in_process", "workers"])
def test_first_call_warm_up_compiles_every_variant(workers, tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__, str(workers or 0)], capture_output=True, text=True,
                          timeout=900, env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                                            "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert not result["late"]
    assert result["late_reported"] == 0


if __name__ == "__main__":
    print(json.dumps(scenario(int(sys.argv[1]) or None)))
