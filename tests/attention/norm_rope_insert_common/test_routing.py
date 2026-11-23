# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the norm + RoPE + insert entries on the host (meta tensors, no GPU): each covered key routes
to its shape package, an uncovered call gets a reason and the op raises it, every reason branch is reachable, and
importing the entry imports no shape package."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.attention import norm_rope_insert_common as E
from amd_kernels.attention.norm_rope_insert_common import device, multilayer, qkv

BF, F32, I64, U8 = torch.bfloat16, torch.float32, torch.int64, torch.uint8
META = "meta"


def t(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device=META)


def qkv_args(d=64, rot=64, n=40, nq=12, nkv=2, cache=U8):
    return {
        "qkv": t(n, (nq + 2 * nkv) * d), "positions": t(n, dtype=I64), "q_norm_weight": t(d), "k_norm_weight": t(d),
        "cos_sin_cache": t(4096, rot), "eps": 1e-5, "num_heads": nq, "num_kv_heads": nkv, "q_out": t(n, nq * d),
        "kv_cache": t(8, nkv, 16, 2 * d, dtype=cache), "slot_mapping": t(n, dtype=I64), "k_scale": t((), dtype=F32),
        "v_scale": t((), dtype=F32),
    }  # fmt: skip


def ml_args(d=64, rot=64, n=40, nl=5, nkv=2, cache=U8):
    caches = [t(8, nkv, 16, 2 * d, dtype=cache) for _ in range(nl)]
    scales = [t((), dtype=F32) for _ in range(nl)]
    return {
        "kv": t(n, nl * 2 * nkv * d), "positions": t(n, dtype=I64), "k_norm_weight": t(nl, d),
        "cos_sin_cache": t(4096, rot), "kv_caches": caches, "slot_mappings": [t(n, dtype=I64)] * nl,
        "k_scales": scales, "v_scales": scales, "eps": 1e-5,
    }  # fmt: skip


OPS = [
    ("qkv", qkv, E.qkv_norm_rope_insert, E.qkv_norm_rope_insert_blocker, qkv_args,
     ("rope_blocker",)),
    ("multilayer", multilayer, E.kv_norm_rope_insert_multilayer, E.kv_norm_rope_insert_multilayer_blocker,
     ml_args, ("rope_blocker",)),
]  # fmt: skip
IDS = [o[0] for o in OPS]


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


@pytest.mark.parametrize("name,mod,op,reason,args,exports", OPS, ids=IDS)
def test_table_imports_and_exports(name, mod, op, reason, args, exports):
    for key, path in mod._ROUTES.hosts.items():
        pkg = importlib.import_module(path)
        assert all(hasattr(pkg, e) for e in exports), (path, exports)
        assert mod._ROUTES.host(key) is pkg
        assert hasattr(mod._op_module(key), op.__name__)


@pytest.mark.parametrize("name,mod,op,reason,args,exports", OPS, ids=IDS)
def test_reason_and_kernel_take_the_op_arguments(name, mod, op, reason, args, exports):
    kernel = getattr(E, op.__name__ + "_kernel")
    sig = list(inspect.signature(op).parameters.values())
    assert list(inspect.signature(reason).parameters.values()) == sig
    assert list(inspect.signature(kernel).parameters.values()) == sig


def test_shapes_are_the_table_keys():
    assert E.qkv_norm_rope_insert_shapes() == frozenset(qkv._ROUTES.hosts) == {(64, 64)}
    assert E.kv_norm_rope_insert_multilayer_shapes() == frozenset(multilayer._ROUTES.hosts) == {(64, 64)}


@pytest.mark.parametrize("cache", [U8, torch.float8_e4m3fn, BF])
@pytest.mark.parametrize("name,mod,op,reason,args,exports", OPS, ids=IDS)
def test_covered_shape_routes_and_has_no_reason(gfx950, name, mod, op, reason, args, exports, cache):
    a = args(cache=cache)
    assert reason(**a) is None
    key = mod._key(a["q_norm_weight"] if name == "qkv" else a["k_norm_weight"], a["cos_sin_cache"])
    assert mod._ROUTES.host(key).__name__ == mod._ROUTES.hosts[key]
    assert getattr(E, op.__name__ + "_kernel")(**a) is getattr(mod._op_module(key), op.__name__)


@pytest.mark.parametrize("d,rot", [(128, 64), (64, 32)])
@pytest.mark.parametrize("name,mod,op,reason,args,exports", OPS, ids=IDS)
def test_uncovered_shape_reason_and_raise(gfx950, name, mod, op, reason, args, exports, d, rot):
    a = args(d=d, rot=rot)
    why = reason(**a)
    assert "no kernel" in why and str(sorted(mod._ROUTES.hosts)) in why, why
    for fn in (op, getattr(E, op.__name__ + "_kernel")):
        with pytest.raises(ValueError) as e:
            fn(**a)
        assert str(e.value) == why


def test_no_device_reason():
    if device.available():
        pytest.skip("gfx950 present")
    for _, _, _, reason, args, _ in OPS:
        assert reason(**args()).endswith("needs gfx950 with FlyDSL")


def test_rope_reason():
    cs = t(4096, 64)
    assert E.norm_rope_insert_rope_blocker(64, 64, True, cs) is None
    assert "not NeoX" in E.norm_rope_insert_rope_blocker(64, 64, False, cs)
    assert "no kernel" in E.norm_rope_insert_rope_blocker(128, 64, True, cs)
    assert "cos / sin cache" in E.norm_rope_insert_rope_blocker(64, 64, True, t(4096, 64, dtype=F32))


def _set(k, v):
    return lambda a: a.update({k: v})


QKV_BRANCHES = [
    ("qkv_width", _set("qkv", t(40, 1088)), "qkv"),
    ("qkv_dtype", _set("qkv", t(40, 1024, dtype=F32)), "qkv"),
    ("qkv_stride", _set("qkv", t(40, 1030)[:, :1024]), "row stride a multiple of 8"),
    ("positions", _set("positions", t(40, dtype=torch.int32)), "positions"),
    ("positions_rows", _set("positions", t(39, dtype=I64)), "positions"),
    ("q_weight", _set("q_norm_weight", t(64, dtype=F32)), "q norm weight"),
    ("k_weight", _set("k_norm_weight", t(32)), "k norm weight"),
    ("cos_sin", _set("cos_sin_cache", t(64, 4096).T), "cos / sin cache"),
    ("q_out", _set("q_out", t(40, 1024)), "q_out"),
    ("cache_heads", _set("kv_cache", t(8, 1, 16, 128, dtype=U8)), "KV cache"),
    ("cache_dtype", _set("kv_cache", t(8, 2, 16, 128, dtype=torch.float16)), "KV cache"),
    ("cache_stride", _set("kv_cache", t(8, 2, 16, 256, dtype=U8)[..., ::2]), "KV cache"),
    ("slots", _set("slot_mapping", t(40, dtype=torch.int32)), "slot mapping"),
    ("scales", _set("k_scale", None), "fp32 k / v scales"),
]
ML_BRANCHES = [
    ("caches", lambda a: a.update(kv_caches=a["kv_caches"][:4] + [t(8, 2, 32, 128, dtype=U8)]), "KV caches"),
    ("cache_dtype", lambda a: a.update(kv_caches=[t(8, 2, 16, 128, dtype=torch.float16)] * 5), "KV caches"),
    ("scales", _set("k_scales", None), "KV caches"),
    ("kv_width", _set("kv", t(40, 1152)), "kv"),
    ("kv_stride", _set("kv", t(40, 1284)[:, :1280]), "row stride a multiple of 8"),
    ("positions", _set("positions", t(40, dtype=torch.int32)), "positions"),
    ("weights", _set("k_norm_weight", t(4, 64)), "K norm weights"),
    ("cos_sin", _set("cos_sin_cache", t(4096, 64, dtype=F32)), "cos / sin cache"),
    ("slot_layers", _set("slot_mappings", [t(40, dtype=I64)] * 4), "slot mappings"),
    ("slot_none", _set("slot_mappings", [t(40, dtype=I64)] * 4 + [None]), "slot mappings"),
    ("slot_lengths", _set("slot_mappings", [t(40, dtype=I64)] * 4 + [t(39, dtype=I64)]), "every layer the same"),
]  # fmt: skip


@pytest.mark.parametrize("label,edit,want", QKV_BRANCHES, ids=[b[0] for b in QKV_BRANCHES])
def test_qkv_reasons(gfx950, label, edit, want):
    a = qkv_args()
    edit(a)
    why = E.qkv_norm_rope_insert_blocker(**a)
    assert why is not None and why.startswith("qkv_norm_rope_insert: ") and want in why, why


@pytest.mark.parametrize("label,edit,want", ML_BRANCHES, ids=[b[0] for b in ML_BRANCHES])
def test_multilayer_reasons(gfx950, label, edit, want):
    a = ml_args()
    edit(a)
    why = E.kv_norm_rope_insert_multilayer_blocker(**a)
    assert why is not None and why.startswith("kv_norm_rope_insert_multilayer: ") and want in why, why


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.attention.norm_rope_insert_common; print(sorted(m for m in sys.modules "
            "if m.startswith('amd_kernels.attention.norm_rope_') and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
