# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the MXFP4 MoE entries on the host (meta tensors, no GPU)."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.moe import mxfp4_moe_common as E
from amd_kernels.moe.mxfp4_moe_common import moe

F32, U8, E4, BF = torch.float32, torch.uint8, torch.float8_e4m3fn, torch.bfloat16
KEYS = sorted(moe._ROUTES.hosts)


def t(*shape, dtype=U8):
    return torch.empty(*shape, dtype=dtype, device="meta")


def wargs(h=6144, i=768, e=128):
    return [t(e, h, dtype=F32), t(e, dtype=F32), t(e, 2 * i, h // 2), t(e, 2 * i, h // 32), t(e, h, i // 2),
            t(e, h, i // 32), t(2 * i, h, dtype=E4), t(2 * i, h // 32), t(h, i, dtype=E4), t(h, i // 32)]


KW = {"top_k": 4, "routed_scaling_factor": 2.0}


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def test_table_and_signatures():
    for path in moe._ROUTES.hosts.values():
        pkg = importlib.import_module(path)
        for name in ("MoeWeights", "SharedWeights", "mxfp4_moe", "warm_up_mxfp4_moe", "set_dbo_query",
                     "unsupported_reason", "SWIGLU_OAI"):
            assert hasattr(pkg, name), (path, name)
    for op, blocker in ((E.mxfp4_moe, E.mxfp4_moe_blocker),
                        (E.mxfp4_moe_weights, E.mxfp4_moe_weights_blocker),
                        (E.mxfp4_moe_weights, E.prepare_mxfp4_moe_weights)):
        assert inspect.signature(blocker).parameters == inspect.signature(op).parameters
    assert E.mxfp4_moe_shapes() == {(6144, 768, 128, 4), (6144, 1536, 128, 4)}


@pytest.mark.parametrize("key", KEYS, ids=[str(k[1]) for k in KEYS])
def test_covered_weights_and_op_route(gfx950, monkeypatch, key):
    pkg = importlib.import_module(moe._ROUTES.hosts[key])
    a = wargs(i=key[1])
    assert E.mxfp4_moe_weights_blocker(*a, **KW) is None
    monkeypatch.setattr(pkg.SharedWeights, "from_row_major", classmethod(lambda cls, *x: ("shared", x)))
    built = []

    def build(cls, *x, **k):
        w = object.__new__(cls)
        w.shared_weights = k["shared"]
        built.append((x, k))
        return w

    monkeypatch.setattr(pkg.MoeWeights, "from_row_major", classmethod(build))
    w = E.prepare_mxfp4_moe_weights(*a, **KW)
    assert isinstance(w, pkg.MoeWeights) and built[0][1]["routed_scaling_factor"] == 2.0
    assert built[0][0][0] is a[0] and w.shared_weights[0] == "shared"
    x = torch.empty(5, 6144, dtype=BF, device="meta")
    assert E.mxfp4_moe_blocker(x, w) is None
    seen = []
    monkeypatch.setattr(pkg, "mxfp4_moe", lambda *y, **k: seen.append((y, k)) or "out")
    monkeypatch.setattr(pkg, "warm_up_mxfp4_moe", lambda *y, **k: seen.append((y, k)))
    assert E.mxfp4_moe(x, w, return_partials=True) == "out"
    E.warm_up_mxfp4_moe(w, partials=True)
    assert seen == [((x, w), {"return_partials": True}), ((w,), {"partials": True})]


@pytest.mark.parametrize("shape", [dict(i=512), dict(e=64), dict(h=4096)], ids=["i512", "e64", "h4096"])
def test_uncovered_reason_and_raise(gfx950, shape):
    a = wargs(**shape)
    why = E.mxfp4_moe_weights_blocker(*a, **KW)
    assert "no kernel" in why and "covered:" in why, why
    assert E.prepare_mxfp4_moe_weights(*a, **KW) == why
    with pytest.raises(ValueError) as e:
        E.mxfp4_moe_weights(*a, **KW)
    assert str(e.value) == why
    assert "no kernel" in E.mxfp4_moe_weights_blocker(*wargs(), top_k=6, routed_scaling_factor=2.0)


def test_reasons(gfx950):
    a = wargs()
    assert "SwiGLU" in E.mxfp4_moe_weights_blocker(*a, **KW, swiglu=(1.0, 7.0, 1.0))
    a[0] = t(128, 6144, dtype=BF)
    assert "router weight" in E.mxfp4_moe_weights_blocker(*a, **KW)
    a = wargs()
    a[1] = None
    assert "correction bias" in E.mxfp4_moe_weights_blocker(*a, **KW)
    a = wargs()
    a[6] = t(1536, 6144)
    assert "shared w13" in E.mxfp4_moe_weights_blocker(*a, **KW)
    x = torch.empty(5, 6144, dtype=BF, device="meta")
    assert "not built" in E.mxfp4_moe_blocker(x, object())
    with pytest.raises(ValueError):
        E.mxfp4_moe(x, object())
    pkg = importlib.import_module(moe._ROUTES.hosts[KEYS[1]])
    w = object.__new__(pkg.MoeWeights)
    w.shared_weights = None
    assert "shared expert" in E.mxfp4_moe_blocker(x, w)
    w.shared_weights = object()
    assert "x " in E.mxfp4_moe_blocker(x.float(), w)


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert E.mxfp4_moe_weights_blocker(*wargs(), **KW).endswith("needs gfx950 with FlyDSL")


def test_dbo_query_reaches_every_package(monkeypatch):
    seen = []
    for path in moe._ROUTES.hosts.values():
        pkg = importlib.import_module(path)
        monkeypatch.setattr(pkg, "set_dbo_query", lambda q, p=path: seen.append((p, q)))
    E.mxfp4_moe_set_dbo_query(len)
    assert sorted(seen) == sorted((p, len) for p in moe._ROUTES.hosts.values())


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.moe.mxfp4_moe_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
