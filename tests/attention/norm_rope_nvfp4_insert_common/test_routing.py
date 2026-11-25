# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the NVFP4 insert entries on the host (meta tensors, no GPU)."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.attention import norm_rope_nvfp4_insert_common as E
from amd_kernels.attention.norm_rope_nvfp4_insert_common import qkv

BF, F32, I64, U8, E4 = torch.bfloat16, torch.float32, torch.int64, torch.uint8, torch.float8_e4m3fn
KEY = (128, 64, 128)


def t(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


def ins(d=128, rot=64, page=128, n=24, nkv=1):
    return [t(d), t(d), t(d), t(d), t(4096, rot), t(n, dtype=I64), t(n, dtype=I64), t(n, dtype=I64),
            t(8, 2 * nkv, page, 72, dtype=U8), t(8, page, 72, dtype=U8), t((), dtype=F32), t((), dtype=F32)]


KW = {"num_heads": 16, "num_kv_heads": 1, "num_index_heads": 1, "eps": 1e-6}


def qkv_args(n=24, **k):
    return [t(n, 20 * 128)] + ins(n=n, **k)


def gemm_args(m=24, **k):
    return [t(m, 6144, dtype=E4), t(m, 192, dtype=U8), t(2560, 6144, dtype=E4), t(2560, 192, dtype=U8), m, BF] + \
        ins(n=m, **k)


OPS = [("qkv", E.qkv_norm_rope_nvfp4_insert, E.qkv_norm_rope_nvfp4_insert_blocker, qkv_args),
       ("gemm", E.mxfp8_gemm_qkv_norm_rope_nvfp4_insert, E.mxfp8_gemm_qkv_norm_rope_nvfp4_insert_blocker, gemm_args)]
IDS = [o[0] for o in OPS]


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


@pytest.mark.parametrize("name,op,blocker,args", OPS, ids=IDS)
def test_table_and_signature(name, op, blocker, args):
    pkg = importlib.import_module(qkv._ROUTES.hosts[KEY])
    assert hasattr(pkg, op.__name__) and hasattr(pkg, "insert_blocker")
    assert inspect.signature(blocker).parameters == inspect.signature(op).parameters
    assert E.qkv_norm_rope_nvfp4_insert_shapes() == {KEY}


@pytest.mark.parametrize("name,op,blocker,args", OPS, ids=IDS)
def test_covered_routes_to_package(gfx950, monkeypatch, name, op, blocker, args):
    a = args()
    assert blocker(*a, **KW) is None
    pkg = importlib.import_module(qkv._ROUTES.hosts[KEY])
    seen = []
    monkeypatch.setattr(pkg, op.__name__, lambda *x, **k: seen.append((x, k)) or "out")
    assert op(*a, **KW) == "out"
    assert seen[0][0][0] is a[0] and seen[0][1]["num_heads"] == 16


@pytest.mark.parametrize("d,rot,page", [(64, 64, 128), (128, 32, 128), (128, 64, 64)])
@pytest.mark.parametrize("name,op,blocker,args", OPS, ids=IDS)
def test_uncovered_reason_and_raise(gfx950, name, op, blocker, args, d, rot, page):
    a = args(d=d, rot=rot, page=page)
    why = blocker(*a, **KW)
    assert "no kernel" in why and f"(covered: [{KEY}])" in why, why
    with pytest.raises(ValueError) as e:
        op(*a, **KW)
    assert str(e.value) == why


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    for _, _, blocker, args in OPS:
        assert blocker(*args(), **KW).endswith("needs gfx950 with FlyDSL")


def _at(i, v):
    def edit(a):
        a[i] = v
    return edit


BRANCHES = [
    ("heads", None, {"num_heads": 8}, "query"),
    ("index_q_dtype", None, {"index_q_dtype": BF}, "index-q"),
    ("eps", None, {"eps": 0.0}, "eps"),
    ("weights", _at(2, t(128, dtype=F32)), {}, "k norm weight"),
    ("positions", _at(6, t(24, dtype=torch.int32)), {}, "positions"),
    ("kv_cache", _at(9, t(8, 4, 128, 72, dtype=U8)), {}, "KV cache"),
    ("index_cache", _at(10, t(8, 128, 64, dtype=U8)), {}, "index cache"),
    ("scales", _at(11, None), {}, "k / v scales"),
    ("qkv", _at(0, t(24, 2048)), {}, "qkv"),
]


@pytest.mark.parametrize("label,edit,kw,want", BRANCHES, ids=[b[0] for b in BRANCHES])
def test_qkv_reasons(gfx950, label, edit, kw, want):
    a = qkv_args()
    if edit is not None:
        edit(a)
    why = E.qkv_norm_rope_nvfp4_insert_blocker(*a, **{**KW, **kw})
    assert why is not None and why.startswith("qkv_norm_rope_nvfp4_insert: ") and want in why, why


@pytest.mark.parametrize("i,v,want", [(2, t(2048, 6144, dtype=E4), "weight"), (5, torch.float16, "out dtype"),
                                     (0, t(24, 4096, dtype=E4), "activation")])
def test_gemm_reasons(gfx950, i, v, want):
    a = gemm_args()
    a[i] = v
    why = E.mxfp8_gemm_qkv_norm_rope_nvfp4_insert_blocker(*a, **KW)
    assert why is not None and why.startswith("mxfp8_gemm_qkv_norm_rope_nvfp4_insert: ") and want in why, why


def test_layer_reason():
    kw = dict(head_dim=128, index_head_dim=128, block_size=128)
    assert E.norm_rope_nvfp4_insert_layer_blocker(16, 1, 1, 64, **kw) is None
    assert "not NeoX" in E.norm_rope_nvfp4_insert_layer_blocker(16, 1, 1, 64, neox=False, **kw)
    assert "no kernel" in E.norm_rope_nvfp4_insert_layer_blocker(16, 1, 1, 32, **kw)


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.attention.norm_rope_nvfp4_insert_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
