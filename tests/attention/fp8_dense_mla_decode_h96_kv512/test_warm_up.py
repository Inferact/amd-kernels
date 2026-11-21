# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4): the first call of a (query rows, cp world) mode compiles the
partials and the merge, and every request count 1..63 of the served modes, with its split layout, then reuses them
(the kernels take the request count and the key counts at run time)."""

import json
import os
import subprocess
import sys

import torch

from .test_decode import launch, make_batch, require_gfx950

SERVED = (3, 4)


def scenario():
    from amd_kernels import warm_up
    from flydsl.compiler.jit_function import MlirCompiler

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    bad = []
    for qlen in SERVED:
        for reqs in (1, 2, 5, 8, 13, 20, 33, 48, 63):
            lens = [200 + 31 * i for i in range(reqs)]
            batch = make_batch(qlen, lens, reqs % 8, seed=reqs)
            before = len(late)
            launch(batch, qlen, reqs % 8)
            if len(late) > before and (qlen, reqs) != (SERVED[0], 1):
                bad.append((qlen, reqs))
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    require_gfx950()
    proc = subprocess.run(
        [sys.executable, "-m", __name__],
        capture_output=True,
        text=True,
        timeout=1500,
        env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")},
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == len(SERVED), result


if __name__ == "__main__":
    print(json.dumps(scenario()))
