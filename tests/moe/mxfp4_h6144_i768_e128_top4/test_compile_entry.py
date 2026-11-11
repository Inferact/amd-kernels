# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Count every compiler invocation after the first-call warm-up (amd_kernels.warm_up) in a fresh process."""
import json
import os
import subprocess
import sys

import pytest


def scenario(workers=None):
    """The first call of each mode warms that mode up (amd_kernels.warm_up; ``workers`` 1: compiles in this process,
    the BF16 reduced mode only; None: in worker processes, every mode and the preparation); every later call compiles
    nothing."""
    import torch
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, mxfp4_moe, prepare_mxfp4_moe
    from .workload import H, make_experts, make_activation, make_shared
    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    warm_up.set_workers(workers)
    torch.set_num_threads(8)
    raw, _ = make_experts()
    gate = torch.randn(128, H, device='cuda') / H**.5
    bias = torch.zeros(128, device='cuda')
    w = MoeWeights.from_row_major(gate, bias, *raw, shared=make_shared()[1])
    count = []
    original = MlirCompiler.compile

    def spy(self, *args, **kw):
        # an entry's compiles only (its warm-up or its call); the caller's own ops (p.compact()) are not entries
        if warm_up._in_warm_up.get() or warm_up._current.get() is not None:
            count.append(1)
        return original(self, *args, **kw)

    MlirCompiler.compile = spy
    calls = [lambda x, p: mxfp4_moe(x, w), lambda x, p: mxfp4_moe(x, w, return_partials=True),
             lambda x, p: mxfp4_moe(p, w), lambda x, p: mxfp4_moe(p, w, return_partials=True),
             lambda x, p: mxfp4_moe(p.compact(), w), lambda x, p: mxfp4_moe(p.compact(), w, return_partials=True)]
    prepare = prepare_mxfp4_moe
    if workers == 1:  # one mode compiles in this process in minutes; the worker path takes them all
        calls, prepare = calls[:1], lambda x, w: None
    x = make_activation(1, seed=1)
    p = prepare(x, w)  # the preparation's first call
    for c in calls:  # each mode's first call
        c(x, p)
    torch.cuda.synchronize()
    warm = len(count)
    late = []
    for m in [1, 9, 15, 16, 18, 26, 27, 32, 40, 42, 63, 66, 80, 90, 127, 130, 159, 160, 191, 192, 194, 230, 255, 258, 384, 385,
              511, 1024, 1536, 2047, 2049, 2560, 3072, 3583, 3584, 3585, 3840, 3841, 4095, 4096, 4097, 7425, 7681, 8191, 8192, 8193, 16384, 24577, 32769,
              2689, 5377, 10881, 16385, 21761]:
        before = len(count)
        x = make_activation(m, seed=m)
        p = prepare(x, w)
        for c in calls:
            c(x, p)
        if len(count) > before:
            late.append(m)
    torch.cuda.synchronize()
    return {'warm_compiles': warm, 'late': late, 'warmed': len(warm_up.warmed()),
            'late_reported': warm_up.late_compiles()}


# both compile paths: in this process, in worker processes
@pytest.mark.parametrize("workers", [1, None], ids=["in_process", "workers"])
def test_first_call_warm_up_compiles_every_variant(workers, tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('needs a GPU')
    # the serving default: FlyDSL's artifact cache on (the warm-up compiles into it), here in an empty directory
    proc = subprocess.run([sys.executable, '-m', __name__, str(workers or 0)], capture_output=True,
                          text=True, timeout=900, env={**os.environ, 'FLYDSL_RUNTIME_ENABLE_CACHE': '1',
                                                      'FLYDSL_RUNTIME_CACHE_DIR': str(tmp_path / 'flydsl')})
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    # with workers every compile replays in a worker: one left here is a launcher without @launcher
    assert result['warm_compiles'] == 0 if workers is None else result['warm_compiles'] > 0
    assert any((tmp_path / 'flydsl').iterdir())  # the warm-up compiled into the cache
    assert not result['late']
    assert result['late_reported'] == 0
    assert result['warmed'] == (1 if workers == 1 else 7)  # the preparation + one key per mxfp4_moe mode


if __name__ == '__main__':
    print(json.dumps(scenario(int(sys.argv[1]) or None)))
