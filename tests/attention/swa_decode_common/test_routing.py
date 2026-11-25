# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the sliding-window decode entry on the host (meta tensors, no GPU)."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.attention import swa_decode_common as E
from amd_kernels.attention.swa_decode_common import decode

BF, F32, I32, U8 = torch.bfloat16, torch.float32, torch.int32, torch.uint8


def t(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


def args(d=128, nq=16, nkv=1, reqs=4, ql=3, flat=True):
    rows = reqs * ql
    return {
        "q": t(rows, nq * d) if flat else t(rows, nq, d), "kv_cache": t(10, nkv, 64, 2 * d, dtype=U8),
        "block_table": t(reqs, 8, dtype=I32), "seq_lens": t(reqs, dtype=I32), "query_start_loc": t(reqs + 1, dtype=I32),
        "max_query_len": ql, "sm_scale": 0.088, "window": 1024, "causal": True, "k_scale": t((), dtype=F32),
        "v_scale": t((), dtype=F32),
    }  # fmt: skip


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def test_table_and_signature():
    for key, path in decode._ROUTES.hosts.items():
        pkg = importlib.import_module(path)
        assert hasattr(pkg, "layer_blocker") and hasattr(pkg, "MAX_QL") and hasattr(pkg, "max_window")
        assert hasattr(decode._ROUTES.module(key, "oproj_input"), "swa_decode_fp8kv_oproj_input")
    assert inspect.signature(E.swa_decode_oproj_input_blocker).parameters == \
        inspect.signature(E.swa_decode_oproj_input).parameters
    assert E.swa_decode_oproj_input_shapes() == {(128, 16)}


@pytest.mark.parametrize("flat", [True, False])
def test_covered_routes_to_package(gfx950, monkeypatch, flat):
    a = args(flat=flat)
    assert E.swa_decode_oproj_input_blocker(**a) is None
    mod = decode._ROUTES.module((128, 16), "oproj_input")
    seen = []
    monkeypatch.setattr(mod, "swa_decode_fp8kv_oproj_input", lambda *x, **k: seen.append((x, k)) or "out")
    assert E.swa_decode_oproj_input(**a, oproj_n=6144) == "out"
    assert seen[0][0][0] is a["q"] and seen[0][1] == {"oproj_n": 6144, "output": None}


@pytest.mark.parametrize("nq,d", [(8, 128), (16, 64)])
def test_uncovered_reason_and_raise(gfx950, nq, d):
    a = args(nq=nq, d=d)
    why = E.swa_decode_oproj_input_blocker(**a)
    assert "no kernel" in why and "(covered: [(128, 16)])" in why, why
    with pytest.raises(ValueError) as e:
        E.swa_decode_oproj_input(**a)
    assert str(e.value) == why


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert E.swa_decode_oproj_input_blocker(**args()).endswith("needs gfx950 with FlyDSL")


BRANCHES = [
    ("q_dtype", {"q": t(12, 2048, dtype=F32)}, "q "),
    ("cache_dtype", {"kv_cache": t(10, 1, 64, 256)}, "KV cache"),
    ("page", {"kv_cache": t(10, 1, 48, 256, dtype=U8)}, "KV cache"),
    ("rows", {"max_query_len": 9}, "max_query_len"),
    ("causal", {"causal": torch.tensor(True)}, "max_query_len"),
    ("window", {"window": 10**6}, "window"),
    ("tables", {"seq_lens": t(4, dtype=torch.int64)}, "seq lens"),
    ("scales", {"k_scale": None}, "k / v scales"),
    ("output", {"output": t(12, 1024)}, "output"),
]


@pytest.mark.parametrize("label,edit,want", BRANCHES, ids=[b[0] for b in BRANCHES])
def test_reasons(gfx950, label, edit, want):
    a = args()
    a.update(edit)
    why = E.swa_decode_oproj_input_blocker(**a)
    assert why is not None and why.startswith("swa_decode_oproj_input: ") and want in why, why


def test_layer_reason_and_max_query_len():
    assert E.swa_decode_layer_blocker(128, 16, 1, 1024) is None
    assert "window" in E.swa_decode_layer_blocker(128, 16, 1, None)
    assert "no kernel" in E.swa_decode_layer_blocker(128, 8, 1, 1024)
    assert E.swa_decode_max_query_len(128, 32, 2) == importlib.import_module(decode._ROUTES.hosts[(128, 16)]).MAX_QL


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.attention.swa_decode_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
