# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""First-call warm-up covers every variant (one chunk kernel a rows-per-sequence count, one merge kernel): after it,
no served batch size, width or row count compiles."""
import json
import os
import subprocess
import sys

import pytest


def scenario(workers):
    import torch
    from amd_kernels import warm_up
    from amd_kernels.indexer.mxfp4_paged_score_topk_d128_h32_top2048 import mxfp4_paged_score_topk
    from flydsl.compiler.jit_function import MlirCompiler

    from .inputs import make_inputs

    warm_up.set_auto(True)
    warm_up.set_workers(workers)
    count = []
    original = MlirCompiler.compile

    def spy(*a, **kw):
        count.append(1)
        return original(*a, **kw)

    MlirCompiler.compile = spy

    def call(seqs, qlen, width):
        inputs = make_inputs(seqs, qlen, width, seed=seqs + qlen + width)
        rows = seqs * qlen
        mxfp4_paged_score_topk(
            **inputs,
            out_indices=torch.empty(rows, 2048, dtype=torch.int32, device="cuda"),
            out_scores=torch.empty(rows, 2048, dtype=torch.float32, device="cuda"),
        )

    call(2, 4, 4096)
    warm = len(count)
    late = []
    for width in (4096, 32768, 131072):
        for seqs, qlen in ((1, 1), (1, 2), (2, 3), (3, 4), (8, 5), (2, 6), (13, 5), (32, 6)):
            before = len(count)
            call(seqs, qlen, width)
            if len(count) != before:
                late.append((seqs, qlen, width))
    torch.cuda.synchronize()
    return dict(warm=warm, late=late, reported=warm_up.late_compiles(), warmed=len(warm_up.warmed()))


@pytest.mark.parametrize("workers", [1, 2], ids=["in_process", "workers"])
def test_first_call_warmup(workers, tmp_path):
    from .inputs import require_gfx950

    require_gfx950()
    proc = subprocess.run(
        [sys.executable, "-m", __name__, str(workers)], capture_output=True, text=True, timeout=1800,
        env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1", "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")},
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert result["warm"] == 0 if workers > 1 else result["warm"] > 0
    assert not result["late"], result
    assert result["reported"] == 0 and result["warmed"] == 1


if __name__ == "__main__":
    print(json.dumps(scenario(int(sys.argv[1]))))
