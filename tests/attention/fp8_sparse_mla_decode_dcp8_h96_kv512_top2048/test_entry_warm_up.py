# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode is an entry (``amd_kernels.warm_up``): the first call of each q mode (bf16 q, pre-quantized q8) compiles
every kernel of that mode in a warm-up; every served row count after it compiles nothing."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 import fp8_sparse_mla_decode
    from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.reference import quantize_q

    from .cases import GRAPH_ROWS, SERVED_ROWS, SM_SCALE, make_batch, served_counts

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    gen = torch.Generator().manual_seed(3)
    q, cache, indptr, slots = make_batch(served_counts(192, gen), gen)
    bad = []
    for rows in sorted(set(SERVED_ROWS) | set(GRAPH_ROWS)):
        before = len(late)
        fp8_sparse_mla_decode(q[:rows], cache, indptr[: rows + 1], slots[: int(indptr[rows])], SM_SCALE)
        fp8_sparse_mla_decode(None, cache, indptr[: rows + 1], slots[: int(indptr[rows])], SM_SCALE, q8=quantize_q(q[:rows]))
        if len(late) > before:
            bad.append(rows)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_the_kernel(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=600,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 2, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
