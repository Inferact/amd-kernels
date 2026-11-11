# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""First-call warm-up of the score-only entry (``nvfp4_index_score_rows``) covers every one-head scorer variant."""
import json
import os
import subprocess
import sys
import pytest


def scenario(ql):
    import torch
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import nvfp4_index_score_rows
    torch.set_num_threads(8)
    warm_up.set_auto(True)
    warm_up.set_workers(1)
    count = []
    original = MlirCompiler.compile
    def spy(*a, **kw):
        count.append(1)
        return original(*a, **kw)
    MlirCompiler.compile = spy
    cache = torch.zeros(2, 128, 72, device='cuda', dtype=torch.uint8)
    def call(n, ctx):
        q = torch.zeros(n * ql, 1, 128, device='cuda', dtype=torch.float8_e4m3fn)
        table = torch.zeros(n, max(1, (ctx + 127) // 128), device='cuda', dtype=torch.int32)
        seq = torch.full((n,), ql, device='cuda', dtype=torch.int32)
        score = torch.empty(1, n * ql, score_width(ctx), device='cuda')
        nvfp4_index_score_rows(q, cache, score, table, seq, 0, 1, query_len=ql, max_seq_len=ctx)
    call(1, 1024)
    warm = len(count)
    late = []
    for ctx in (8192, 196608, 1048576):
        for n in (1, 2, 3, 4, 5, 16, 17, 32, 33, 64, 128, 256):
            before = len(count)
            call(n, ctx)
            if len(count) != before:
                late.append((n, ctx))
    torch.cuda.synchronize()
    return dict(warm=warm, late=late, reported=warm_up.late_compiles(), warmed=len(warm_up.warmed()))


@pytest.mark.parametrize('ql', [4, 8])
def test_first_call_warmup(ql, tmp_path):
    from .inputs import require_gfx950
    require_gfx950()
    proc = subprocess.run([sys.executable, '-m', __name__, str(ql)],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, 'FLYDSL_RUNTIME_ENABLE_CACHE': '1',
             'FLYDSL_RUNTIME_CACHE_DIR': str(tmp_path / 'flydsl')})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert result['warm'] > 0
    assert not result['late'], result
    assert result['reported'] == 0 and result['warmed'] == 1


if __name__ == '__main__':
    print(json.dumps(scenario(int(sys.argv[1]))))
