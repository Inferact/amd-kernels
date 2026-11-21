# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Warm-up through the entry, in a fresh process with an empty FlyDSL cache: the first call of each vocabulary warms
its package up, then every served row count runs through the entry with no compile (``compile_watch``), and a CUDA
graph of the entry captured with ``importlib.import_module`` failing replays bit for bit as the eager call."""

import importlib
import json
import os
import subprocess
import sys

import pytest
import torch

from amd_kernels.sampling.top_p_mask_common import available

ROWS = {163840: tuple(range(1, 65)), 200064: tuple(range(1, 65)) + (65, 100, 192, 256)}


def scenario():
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.sampling.top_p_mask_common import top_p_mask

    warm_up.set_auto(True)
    gen = torch.Generator().manual_seed(5)
    buffer = torch.empty(256, 200064, device="cuda")

    def call(vocab, n, out=None):
        x = (torch.randn(n, vocab, generator=gen) * 3).cuda() if out is None else out
        p = torch.full((n,), 0.9, device="cuda")
        return top_p_mask(x, p, buffer if n > 64 else None)

    for vocab in ROWS:
        call(vocab, 3)
    torch.cuda.synchronize()
    late = []
    compile_watch.arm(late.append)
    for vocab, ns in ROWS.items():
        for n in ns:
            call(vocab, n)
    torch.cuda.synchronize()
    replay_equal = []
    real_import = importlib.import_module

    def no_import(name, *a, **k):
        raise AssertionError(f"import of {name} inside capture")

    for vocab, n in ((163840, 24), (200064, 8), (200064, 100)):
        src = (torch.randn(n, vocab, generator=gen) * 3).cuda()
        eager = call(vocab, n, src.clone())
        buf = src.clone()
        g = torch.cuda.CUDAGraph()
        importlib.import_module = no_import
        try:
            with torch.cuda.graph(g):
                call(vocab, n, buf)
        finally:
            importlib.import_module = real_import
        buf.copy_(src)
        g.replay()
        torch.cuda.synchronize()
        replay_equal.append(torch.equal(buf.view(torch.int32), eager.view(torch.int32)))
    return {"late": len(late), "watch": compile_watch.late_compiles(), "warm_up_late": warm_up.late_compiles(),
            "warmed": sorted(name for name, _ in warm_up.warmed()), "replay_equal": replay_equal}


def test_warm_up_through_the_entry(tmp_path):
    if not available():
        pytest.skip("requires gfx950 with FlyDSL")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=900, check=False,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {"late": 0, "watch": 0, "warm_up_late": 0, "replay_equal": [True, True, True],
                      "warmed": ["sampling/top_p_mask_f32_v163840", "sampling/top_p_mask_f32_v200064"]}, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
