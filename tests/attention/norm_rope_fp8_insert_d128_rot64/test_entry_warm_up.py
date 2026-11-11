# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The two inserts are entries (``amd_kernels.warm_up``): the first call of each mode (with / without the cache
insert, the multi-layer insert) compiles its kernel; calls at other token counts compile nothing."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import (
        kv_norm_rope_insert_multilayer,
    )
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.qkv_norm_rope_insert import qkv_norm_rope_insert
    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    dev, bf16 = torch.device("cuda"), torch.bfloat16
    nq, nkv, nl = 16, 1, 5
    caches = [torch.zeros((8, nkv, 128, 256), dtype=torch.uint8, device=dev) for _ in range(nl)]
    one = torch.ones((), dtype=torch.float32, device=dev)
    w = torch.ones(128, dtype=bf16, device=dev)
    cos_sin = torch.zeros((4096, 64), dtype=bf16, device=dev)
    bad = []
    for n in (1, 3, 7, 8, 21, 64, 448, 1792):
        before = len(late)
        qkv = torch.randn((n, (nq + 2 * nkv) * 128), dtype=bf16, device=dev)
        pos = torch.arange(n, dtype=torch.int64, device=dev)
        q = torch.empty((n, nq * 128), dtype=bf16, device=dev)
        slots = torch.arange(n, dtype=torch.int64, device=dev) % 1024
        qkv_norm_rope_insert(qkv, pos, w, w, cos_sin, 1e-6, nq, nkv, q)
        qkv_norm_rope_insert(qkv, pos, w, w, cos_sin, 1e-6, nq, nkv, q, caches[0], slots, one, one)
        kv = torch.randn((n, nl * 2 * nkv * 128), dtype=bf16, device=dev)
        kv_norm_rope_insert_multilayer(kv, pos, torch.ones((nl, 128), dtype=bf16, device=dev), cos_sin, caches,
                                       [slots] * nl, [one] * nl, [one] * nl, 1e-6)
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
