# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""First-call warm-up of the rank-owned ``score_topk`` entry covers every variant; ``COMPILE_ONLY`` runs nothing, so
no peers are needed (``test_split4_rank_owned_ranks4`` runs the exchanges)."""
import json
import os
import subprocess
import sys

import pytest


def scenario(ql, workers):
    import torch
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from .test_compile import RankOwned
    torch.set_num_threads(1)
    warm_up.set_auto(True)
    warm_up.set_workers(workers)
    count = []
    original = MlirCompiler.compile
    def spy(*a, **kw):
        count.append(1)
        return original(*a, **kw)
    MlirCompiler.compile = spy
    chain = RankOwned(ql, fused=True)
    chain.decode([ql], 1024)
    warm = len(count)
    late = []
    for n in (1, 16, 17, 21, 22, 31, 32, 33, 63, 64, 65, 128, 129, 256):
        for ctx in (8192, 196608, 1048576):
            before = len(count)
            chain.decode([ql] * n, ctx)
            if len(count) != before:
                late.append((n, ctx))
    return dict(warm=warm, late=late, reported=warm_up.late_compiles(), warmed=len(warm_up.warmed()))


@pytest.mark.parametrize('ql', [4, 8])
@pytest.mark.parametrize('workers', [1, 2], ids=['in_process', 'workers'])
def test_first_call_split4_warmup(ql, workers, tmp_path):
    from .inputs import require_gfx950
    require_gfx950()
    proc = subprocess.run([sys.executable, '-m', __name__, str(ql), str(workers)],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, 'COMPILE_ONLY': '1', 'FLYDSL_RUNTIME_ENABLE_CACHE': '1',
             'FLYDSL_RUNTIME_CACHE_DIR': str(tmp_path / 'flydsl')})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result['warm'] == 0 if workers > 1 else result['warm'] > 0
    assert not result['late'], result
    assert result['reported'] == 0 and result['warmed'] == 1, result


if __name__ == '__main__':
    print(json.dumps(scenario(int(sys.argv[1]), int(sys.argv[2]))))
