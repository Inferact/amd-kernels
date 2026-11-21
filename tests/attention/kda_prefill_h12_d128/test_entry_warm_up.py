# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``prefill`` is an entry (``amd_kernels.warm_up``): the first call of each mode (norm weight dtype, checkpoint export) compiles
every kernel of the layer (prologue, chain, fold, walk, norm); calls at other lengths and batches compile nothing."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from tests.attention.kda_prefill_h12_d128.test_compile import call
    from tests.attention.kda_prefill_h12_d128.inputs import make_case

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    bad = []
    for w_f32, ck in ((True, False), (False, False), (True, True)):
        for seq in ([300], [64], [4096], [300, 64, 77, 1], [1500, 700, 64, 3000], [10240]):
            before = len(late)
            call(make_case(seq), w_f32, ck)
            if len(late) > before:
                bad.append((w_f32, ck, seq))
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1200,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 3, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
