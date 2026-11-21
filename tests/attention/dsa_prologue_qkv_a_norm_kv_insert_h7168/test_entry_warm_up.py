# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Both kernels are entries (``amd_kernels.warm_up``): the first call of each mode compiles its kernel in the warm-up;
calls at other row counts compile nothing."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.indexer_prep import indexer_prep
    from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.qkv_a_norm_kv_insert import qkv_a_norm_kv_insert

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    dev, bf16 = "cuda", torch.bfloat16
    kvc = torch.zeros((8, 64, 656), dtype=torch.uint8, device=dev)
    ic = torch.zeros((8, 64, 68), dtype=torch.uint8, device=dev)
    qw, kw = torch.ones(1536, dtype=bf16, device=dev), torch.ones(512, dtype=bf16, device=dev)
    lw, lb = torch.ones(128, dtype=bf16, device=dev), torch.zeros(128, dtype=bf16, device=dev)
    bad = []
    for n in (1, 3, 12, 65, 192):
        before = len(late)
        x = torch.randn((n, 2112), dtype=bf16, device=dev)
        q, kv = torch.empty((n, 1536), dtype=bf16, device=dev), torch.empty((n, 512), dtype=bf16, device=dev)
        slots = torch.arange(n, dtype=torch.int64, device=dev) % 512
        qkv_a_norm_kv_insert(x, qw, kw, 1e-6, q, kv)
        qkv_a_norm_kv_insert(x, qw, kw, 1e-6, q, kv, kvc, slots)
        out = torch.zeros((n + 2, 2048), dtype=torch.int32, device=dev)
        indexer_prep(
            torch.randn((n, 32, 128), dtype=bf16, device=dev), torch.randn((n, 128), dtype=bf16, device=dev),
            torch.randn((n, 32), dtype=bf16, device=dev), lw, lb, 1e-6, slots, ic, out, 0.0156,
            torch.empty((n, 32, 64), dtype=torch.uint8, device=dev),
            torch.empty((n, 1, 4, 16, 4), dtype=torch.uint8, device=dev),
            torch.empty((n, 32), dtype=torch.float32, device=dev),
        )  # fmt: skip
        if len(late) > before:
            bad.append(n)
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=600,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 3, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
