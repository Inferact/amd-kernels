# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Warm-up through the entry, in a fresh process with an empty FlyDSL cache: the first call warms the package up,
then every served row count runs through the entry with no compile (``compile_watch``), and a CUDA graph of the
entry captured with ``importlib.import_module`` failing replays bit for bit as the eager call."""

import importlib
import json
import os
import subprocess
import sys

import pytest
import torch

from amd_kernels.sampling.markov_gumbel_common import available

K = 3


def scenario():
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.sampling.markov_gumbel_common import markov_gumbel_fused

    from .inputs import clone, make

    warm_up.set_auto(True)
    d = make(16, K, 1)
    weights = (d["w1"], d["w2"])
    markov_gumbel_fused(**d)
    torch.cuda.synchronize()
    late = []
    compile_watch.arm(late.append)
    for n in range(1, 49):
        markov_gumbel_fused(**make(n, K, n, weights))
    torch.cuda.synchronize()
    src = make(20, K, 99, weights)
    eager = clone(src)
    markov_gumbel_fused(**eager)
    run = clone(src)
    g = torch.cuda.CUDAGraph()
    real_import = importlib.import_module

    def no_import(name, *a, **k):
        raise AssertionError(f"import of {name} inside capture")

    importlib.import_module = no_import
    try:
        with torch.cuda.graph(g):
            markov_gumbel_fused(**run)
    finally:
        importlib.import_module = real_import
    for key in ("draft_tokens", "cache"):
        run[key].copy_(src[key])
    g.replay()
    torch.cuda.synchronize()
    equal = all(torch.equal(run[key], eager[key]) for key in ("draft_tokens", "cache"))
    return {"late": len(late), "watch": compile_watch.late_compiles(), "warm_up_late": warm_up.late_compiles(),
            "warmed": sorted(name for name, _ in warm_up.warmed()), "replay_equal": equal}


def test_warm_up_through_the_entry(tmp_path):
    if not available():
        pytest.skip("requires gfx950 with FlyDSL")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=900, check=False,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {"late": 0, "watch": 0, "warm_up_late": 0, "replay_equal": True,
                      "warmed": ["sampling/markov_gumbel_fused_v163840_r512"]}, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
