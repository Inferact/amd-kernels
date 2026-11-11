# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""vLLM's start-up order with the library's SP prefill, in a fresh process: the BF16 entry called at every warm-up
size, its partials mode, then sp_prefill_akr's first call (the compact partials input of a 32-row shard). The compact
mode's warm-up must launch nothing: a launch there runs the GEMM2 the BF16 calls loaded at 97-128 rows in its 17-row
step, on a sort only compiled there, and the server faults on every rank."""
import json
import os
import subprocess
import sys

import pytest

# the compact shapes the SP path reaches: the 17-row step that faulted, the warm-up shard, the smallest SP prefill
COMPACT_ROWS = (17, 32, 96, 4096)


def scenario():
    import torch
    from flydsl.compiler.jit_function import CompiledFunction
    from amd_kernels import warm_up
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, mxfp4_moe, prepare_mxfp4_moe
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import WARM_UP_TOKENS
    from .test_shared_expert import weights
    from .workload import H, make_activation, make_experts
    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    raw, _ = make_experts()
    g = torch.Generator(device='cuda').manual_seed(5)
    w = MoeWeights.from_row_major(torch.randn(128, H, device='cuda', generator=g) / H**.5,
                                  torch.randn(128, device='cuda', generator=g) * 1e-2, *raw)
    _, w.shared_weights = weights.__wrapped__()
    in_warm_up = []
    launch = CompiledFunction.__call__

    def spy(self, *args):  # a loaded handle's launch; the warm-up may launch none
        if warm_up._in_warm_up.get():
            in_warm_up.append(1)
        return launch(self, *args)

    CompiledFunction.__call__ = spy

    def zeros(m):
        return torch.zeros((m, H), device='cuda', dtype=torch.bfloat16)

    with torch.inference_mode():
        for m in WARM_UP_TOKENS:  # akr_moe.warm_up
            mxfp4_moe(zeros(m), w)
        mxfp4_moe(zeros(WARM_UP_TOKENS[0]), w, return_partials=True)
        # sp_prefill_akr._warm_up: the compact mode's first call
        mxfp4_moe(prepare_mxfp4_moe(zeros(32), w).compact(), w, return_partials=True)
        torch.cuda.synchronize()
        late = warm_up.late_compiles()
        same = []
        for m in COMPACT_ROWS:  # the gathered input's contract: the plain entry's partials, bit for bit
            x = make_activation(m, seed=m)
            expected = mxfp4_moe(x, w, return_partials=True)
            got = mxfp4_moe(prepare_mxfp4_moe(x, w).compact(), w, return_partials=True)
            same.append(all(torch.equal(a, b) and bool(torch.isfinite(a).all()) for a, b in zip(expected, got)))
        torch.cuda.synchronize()
    return {'launches_in_warm_up': len(in_warm_up), 'late_before': late, 'late': warm_up.late_compiles() - late,
            'same': same}


def test_sp_prefill_compact_warm_up_launches_nothing():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('needs a GPU')
    proc = subprocess.run([sys.executable, '-m', __name__], capture_output=True, text=True, timeout=900,
                          env={**os.environ, 'FLYDSL_RUNTIME_ENABLE_CACHE': '1'})
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert result['launches_in_warm_up'] == 0
    assert result['late_before'] == 0 and result['late'] == 0
    assert all(result['same']), dict(zip(COMPACT_ROWS, result['same']))


if __name__ == '__main__':
    print(json.dumps(scenario()))
