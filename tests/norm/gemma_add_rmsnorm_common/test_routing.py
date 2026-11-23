# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the Gemma add + RMSNorm entries on the host (meta tensors, a stub group, no GPU)."""

import importlib
import inspect
import subprocess
import sys
import types

import pytest
import torch

from amd_kernels import routes
from amd_kernels.norm import gemma_add_rmsnorm_common as E
from amd_kernels.norm.gemma_add_rmsnorm_common import add_rmsnorm, allreduce

BF, E4, U8 = torch.bfloat16, torch.float8_e4m3fn, torch.uint8


def t(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)
    for path in allreduce._ROUTES.hosts.values():
        monkeypatch.setattr(importlib.import_module(path), "unsupported_reason",
                            lambda world, hidden=6144: None)


def group(world, rank=0):
    return types.SimpleNamespace(get_world_size=lambda g: world, get_rank=lambda g: rank)


def test_tables_and_signatures():
    mod = add_rmsnorm._module(6144)
    assert all(hasattr(mod, n) for n in ("gemma_add_rmsnorm_mxfp8_for", "add_rmsnorm_ok", "warmup_for_gemm"))
    for path in list(allreduce._ROUTES.hosts.values()) + list(allreduce._MLP.hosts.values()):
        pkg = importlib.import_module(path)
        assert all(hasattr(pkg, n) for n in ("DecoderArNorm", "DenseMlp", "unsupported_reason", "dense_mlp_supported"))
    for op, others in ((E.gemma_add_rmsnorm_mxfp8_for, (E.gemma_add_rmsnorm_mxfp8_for_blocker,)),
                       (E.decoder_ar_norm, (E.decoder_ar_norm_blocker, E.prepare_decoder_ar_norm)),
                       (E.ar_norm_dense_mlp, (E.ar_norm_dense_mlp_blocker, E.prepare_ar_norm_dense_mlp))):
        for other in others:
            assert inspect.signature(other).parameters == inspect.signature(op).parameters
    assert E.gemma_add_rmsnorm_mxfp8_for_shapes() == {6144}
    assert E.decoder_ar_norm_shapes() == {(6144, 4), (6144, 2)}


def norm_args(h=6144):
    return (t(9, h), t(9, h), t(h), 1e-6, t(2560, h, dtype=E4))


def test_add_rmsnorm_routes(gfx950, monkeypatch):
    a = norm_args()
    assert E.gemma_add_rmsnorm_mxfp8_for_blocker(*a) is None
    seen = []
    monkeypatch.setattr(add_rmsnorm._module(6144), "gemma_add_rmsnorm_mxfp8_for", lambda *x: seen.append(x) or "out")
    assert E.gemma_add_rmsnorm_mxfp8_for(*a) == "out" and seen == [a]


def test_add_rmsnorm_reasons(gfx950):
    a = norm_args(4096)
    why = E.gemma_add_rmsnorm_mxfp8_for_blocker(*a)
    assert "no kernel" in why and "(covered: [6144])" in why
    with pytest.raises(ValueError) as e:
        E.gemma_add_rmsnorm_mxfp8_for(*a)
    assert str(e.value) == why
    x, r, w, eps, g = norm_args()
    assert "x " in E.gemma_add_rmsnorm_mxfp8_for_blocker(x.float(), r, w, eps, g)
    assert "GEMM weight" in E.gemma_add_rmsnorm_mxfp8_for_blocker(x, r, w, eps, t(2560, 4096, dtype=E4))


@pytest.mark.parametrize("world", [4, 2])
def test_decoder_ar_norm_routes(gfx950, monkeypatch, world):
    monkeypatch.setattr(allreduce, "dist", group(world))
    pkg = importlib.import_module(allreduce._ROUTES.hosts[(6144, world)])
    built = []
    monkeypatch.setattr(pkg, "DecoderArNorm", lambda *a, **k: built.append((a, k)) or "comm")
    w = t(6144)
    assert E.decoder_ar_norm_world_blocker(world, 6144) is None
    assert E.prepare_decoder_ar_norm(0, "g", "cuda", w, 1e-6, moe="moe") == "comm"
    assert built == [((0, "g", "cuda", w, 1e-6), {"moe": "moe"})]


def test_decoder_ar_norm_reasons(gfx950, monkeypatch):
    monkeypatch.setattr(allreduce, "dist", group(8))
    why = E.prepare_decoder_ar_norm(0, "g", "cuda", t(6144), 1e-6)
    assert "no kernel" in why and "covered:" in why
    monkeypatch.setattr(allreduce, "dist", group(4, rank=1))
    assert "rank 0" in E.decoder_ar_norm_blocker(0, "g", "cuda", t(6144), 1e-6)
    assert "norm weight" in E.decoder_ar_norm_blocker(1, "g", "cuda", t(6144, dtype=torch.float32), 1e-6)
    pkg = importlib.import_module(allreduce._ROUTES.hosts[(6144, 4)])
    monkeypatch.setattr(pkg, "unsupported_reason", lambda world, hidden=6144: "not with expandable segments")
    assert E.decoder_ar_norm_world_blocker(4, 6144) == "decoder_ar_norm: not with expandable segments"


@pytest.mark.parametrize("key", [(6144, 3072), (6144, 6144)])
def test_dense_mlp_routes(gfx950, monkeypatch, key):
    pkg = importlib.import_module(allreduce._MLP.hosts[key])
    monkeypatch.setattr(pkg, "dense_mlp_supported", lambda g, d: True)
    monkeypatch.setattr(pkg, "DenseMlp", lambda *a: ("mlp", a))
    a = (t(2 * key[1], 6144, dtype=E4), t(1, dtype=U8), t(*key, dtype=E4), t(1, dtype=U8), 1.702, 1.0, 7.0)
    assert E.prepare_ar_norm_dense_mlp(*a) == ("mlp", a)
    monkeypatch.setattr(pkg, "dense_mlp_supported", lambda g, d: False)
    assert "gate_up" in E.prepare_ar_norm_dense_mlp(*a)
    assert "no kernel" in E.ar_norm_dense_mlp_blocker(a[0], a[1], t(6144, 1024, dtype=E4), *a[3:])


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert E.gemma_add_rmsnorm_mxfp8_for_blocker(*norm_args()).endswith("needs gfx950 with FlyDSL")
    assert E.decoder_ar_norm_world_blocker(4, 6144).endswith("needs gfx950 with FlyDSL")


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.norm.gemma_add_rmsnorm_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
