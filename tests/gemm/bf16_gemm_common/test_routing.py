# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the shuffled-weight bf16 GEMM entry on the host (meta tensors, no GPU)."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.gemm import bf16_gemm_common as E
from amd_kernels.gemm.bf16_gemm_common import host as gemm

BF = torch.bfloat16
KEYS = sorted(gemm._ROUTES.hosts)


def t(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def test_table_and_signature():
    for (n, k), path in gemm._ROUTES.hosts.items():
        pkg = importlib.import_module(path)
        for name in (f"bf16_gemm_n{n}_k{k}", f"warmup_bf16_gemm_n{n}_k{k}", "supports", "shuffle_weight_"):
            assert hasattr(pkg, name), (path, name)
    assert inspect.signature(E.bf16_gemm_blocker).parameters == \
        inspect.signature(E.bf16_gemm).parameters
    assert E.bf16_gemm_shapes() == set(KEYS)


@pytest.mark.parametrize("key", KEYS, ids=[str(k[0]) for k in KEYS])
def test_covered_routes_to_package(gfx950, monkeypatch, key):
    n, k = key
    pkg = importlib.import_module(gemm._ROUTES.hosts[key])
    x, w = t(128, k), t(n, k)
    assert E.bf16_gemm_blocker(x, w) is None and E.bf16_gemm_weight_blocker(w) is None
    seen = []
    monkeypatch.setattr(pkg, f"bf16_gemm_n{n}_k{k}", lambda *a, **kw: seen.append((a, kw)) or "out")
    monkeypatch.setattr(pkg, "shuffle_weight_", lambda a: seen.append(a) or a)
    assert E.bf16_gemm(x, w, shuffled=True) == "out"
    h = E.prepare_bf16_gemm(w, warm_up=False)
    assert h.weight is w and h.shuffled
    assert seen == [((x, w, None), {"shuffled": True}), w]


@pytest.mark.parametrize("shape", [(50016, 4096), (1000, 6144)])
def test_uncovered_reason_and_raise(gfx950, shape):
    x, w = t(7, shape[1]), t(*shape)
    why = E.bf16_gemm_blocker(x, w)
    assert "no kernel" in why and "covered:" in why, why
    with pytest.raises(ValueError) as e:
        E.bf16_gemm(x, w)
    assert str(e.value) == why
    assert "no kernel" in E.prepare_bf16_gemm(w)


def test_reasons(gfx950):
    w = t(50016, 6144)
    assert "weight" in E.bf16_gemm_blocker(t(7, 6144), t(50016, 6144, dtype=torch.float32))
    assert "x " in E.bf16_gemm_blocker(t(7, 5120), w)
    assert "out " in E.bf16_gemm_blocker(t(7, 6144), w, t(7, 100))
    assert "bias" in E.bf16_gemm_weight_blocker(w, t(50016))
    assert "[N, K]" in E.bf16_gemm_blocker(t(7, 6144), t(6144))


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert E.bf16_gemm_blocker(t(7, 6144), t(50016, 6144)).endswith("needs gfx950 with FlyDSL")


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.gemm.bf16_gemm_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
