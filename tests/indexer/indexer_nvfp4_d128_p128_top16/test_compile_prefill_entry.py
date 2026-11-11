# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The prefill top-16 entry's warm-up (``prefill_top16.warm_up``, as the layer's ``init`` calls it) covers every
variant the served steps reach: no FlyDSL compile afterwards at any row count up to 32768 and context up to 1M."""
import json
import os
import subprocess
import sys
import pytest


def scenario(workers):
    import torch
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import prefill_top16
    torch.set_num_threads(8)
    warm_up.set_auto(True)
    warm_up.set_workers(workers)
    count = []
    original = MlirCompiler.compile
    def spy(*a, **kw):
        count.append(1)
        return original(*a, **kw)
    MlirCompiler.compile = spy
    cache = torch.zeros(2, 128, 72, device='cuda', dtype=torch.uint8)
    def call(q_lens, ctx):
        rows = sum(q_lens)
        q = torch.zeros(rows, 1, 128, device='cuda', dtype=torch.float8_e4m3fn)
        table = torch.zeros(len(q_lens), (ctx + 127) // 128, device='cuda', dtype=torch.int32)
        cu = torch.tensor([0] + q_lens, device='cuda', dtype=torch.int32).cumsum(0).to(torch.int32)
        seq = torch.full((len(q_lens),), ctx, device='cuda', dtype=torch.int32)
        nvp = torch.full((rows,), (ctx + 127) // 128, device='cuda', dtype=torch.int32)
        ids = torch.empty(1, rows, 16, device='cuda', dtype=torch.int32)
        prefill_top16.index_prefill_top16(q, cache, table, cu, seq, 0, 1, ids, max_query_len=max(q_lens),
                                          max_seq_len=ctx, num_valid_pages=nvp)
    with torch.inference_mode():
        prefill_top16.warm_up(torch.device('cuda', 0), 1)
        warm = len(count)
        late = []
        for ctx in (4096, 8192, 32769, 65537, 131073, 163841, 212993, 262144, 1048576):
            for q_lens in ([1], [100], [512], [1000, 3096], [4096], [8192], [16384], [32768]):
                if sum(q_lens) > ctx or (ctx > 262144 and sum(q_lens) > 16384):
                    continue
                before = len(count)
                call(q_lens, ctx)
                if len(count) != before:
                    late.append((q_lens, ctx))
        torch.cuda.synchronize()
    return dict(warm=warm, late=late, reported=warm_up.late_compiles(), warmed=len(warm_up.warmed()))


@pytest.mark.parametrize('workers', [1, 2], ids=['in_process', 'workers'])
def test_warm_up_covers_served_sizes(workers, tmp_path):
    from .inputs import require_gfx950
    require_gfx950()
    proc = subprocess.run([sys.executable, '-m', __name__, str(workers)],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, 'FLYDSL_RUNTIME_ENABLE_CACHE': '1',
             'FLYDSL_RUNTIME_CACHE_DIR': str(tmp_path / 'flydsl')})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert result['warm'] == 0 if workers > 1 else result['warm'] > 0
    assert not result['late'], result
    assert result['reported'] == 0 and result['warmed'] == 1


if __name__ == '__main__':
    print(json.dumps(scenario(int(sys.argv[1]))))
