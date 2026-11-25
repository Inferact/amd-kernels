# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the Gemma norm + RoPE + insert entry on the host (meta tensors, no GPU): the covered key
calls its shape package with the op's arguments, an uncovered call gets a reason and the op raises it, the reason
branches are reachable, and importing the entry imports no shape package."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.attention import gemma_norm_rope_insert_common as E
from amd_kernels.attention.gemma_norm_rope_insert_common import qkv

BF, F32, I64, U8 = torch.bfloat16, torch.float32, torch.int64, torch.uint8


def t(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


def args(d=128, rot=64, n=24, nq=16, nkv=1):
    return {
        "qkv": t(n, (nq + 2 * nkv) * d), "positions": t(n, dtype=I64), "q_norm_weight": t(d), "k_norm_weight": t(d),
        "cos_sin_cache": t(4096, rot), "eps": 1e-6, "num_heads": nq, "num_kv_heads": nkv, "q_out": t(n, nq * d),
        "kv_cache": t(8, nkv, 16, 2 * d, dtype=U8), "slot_mapping": t(n, dtype=I64), "k_scale": t((), dtype=F32),
        "v_scale": t((), dtype=F32),
    }  # fmt: skip


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def test_table_imports_and_signatures():
    for key, path in qkv._ROUTES.hosts.items():
        pkg = importlib.import_module(path)
        assert hasattr(pkg, "rope_blocker") and hasattr(qkv._op_module(key), "qkv_norm_rope_insert")
    sig = inspect.signature(E.gemma_qkv_norm_rope_insert).parameters
    assert list(inspect.signature(E.gemma_qkv_norm_rope_insert_blocker).parameters.values()) == list(sig.values())
    assert list(inspect.signature(E.gemma_qkv_norm_rope_insert_kernel).parameters.values()) == list(sig.values())
    assert E.gemma_qkv_norm_rope_insert_shapes() == {(128, 64)}


def test_covered_routes_to_package(gfx950, monkeypatch):
    a = args()
    assert E.gemma_qkv_norm_rope_insert_blocker(**a) is None
    mod = qkv._op_module((128, 64))
    seen = []
    monkeypatch.setattr(mod, "qkv_norm_rope_insert", lambda *x: seen.append(x))
    E.gemma_qkv_norm_rope_insert(**a)
    assert len(seen) == 1 and seen[0][0] is a["qkv"] and seen[0][-1] is a["v_scale"]
    assert E.gemma_qkv_norm_rope_insert_kernel(**a) is mod.qkv_norm_rope_insert


@pytest.mark.parametrize("d,rot", [(64, 64), (128, 32)])
def test_uncovered_reason_and_raise(gfx950, d, rot):
    a = args(d=d, rot=rot)
    why = E.gemma_qkv_norm_rope_insert_blocker(**a)
    assert "no kernel" in why and "(covered: [(128, 64)])" in why, why
    with pytest.raises(ValueError) as e:
        E.gemma_qkv_norm_rope_insert(**a)
    assert str(e.value) == why


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert E.gemma_qkv_norm_rope_insert_blocker(**args()).endswith("needs gfx950 with FlyDSL")


def _set(k, v):
    return lambda a: a.update({k: v})


BRANCHES = [
    ("qkv", _set("qkv", t(24, 2048)), "qkv"),
    ("positions", _set("positions", t(24, dtype=torch.int32)), "positions"),
    ("q_weight", _set("q_norm_weight", t(128, dtype=F32)), "q norm weight"),
    ("cos_sin", _set("cos_sin_cache", t(64, 4096).T), "cos / sin cache"),
    ("q_out", _set("q_out", t(24, 1024)), "q_out"),
    ("cache_dtype", _set("kv_cache", t(8, 1, 16, 256)), "KV cache"),
    ("cache_heads", _set("kv_cache", t(8, 2, 16, 256, dtype=U8)), "KV cache"),
    ("slots", _set("slot_mapping", t(24, dtype=torch.int32)), "slot mapping"),
    ("scales", _set("k_scale", None), "fp32 k / v scales"),
]


@pytest.mark.parametrize("label,edit,want", BRANCHES, ids=[b[0] for b in BRANCHES])
def test_reasons(gfx950, label, edit, want):
    a = args()
    edit(a)
    why = E.gemma_qkv_norm_rope_insert_blocker(**a)
    assert why is not None and why.startswith("gemma_qkv_norm_rope_insert: ") and want in why, why


def test_rope_reason():
    cs = t(4096, 64)
    assert E.gemma_norm_rope_insert_rope_blocker(128, 64, True, cs) is None
    assert "not NeoX" in E.gemma_norm_rope_insert_rope_blocker(128, 64, False, cs)
    assert "no kernel" in E.gemma_norm_rope_insert_rope_blocker(64, 64, True, cs)


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.attention.gemma_norm_rope_insert_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
