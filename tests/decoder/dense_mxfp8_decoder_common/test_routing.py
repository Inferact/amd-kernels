# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the dense MXFP8 decoder-stack entries on the host (meta tensors, a stub group, no GPU)."""

import importlib
import inspect
import subprocess
import sys
import types

import pytest
import torch

from amd_kernels import routes
from amd_kernels.decoder import dense_mxfp8_decoder_common as E
from amd_kernels.decoder.dense_mxfp8_decoder_common import stack

BF, E4, U8 = torch.bfloat16, torch.float8_e4m3fn, torch.uint8
KEYS = sorted(stack._ROUTES.hosts)


def t(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


def group(world, rank=0):
    return types.SimpleNamespace(get_world_size=lambda g: world, get_rank=lambda g: rank)


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def test_tables_and_signatures():
    for (h, world), path in stack._ROUTES.hosts.items():
        pkg = importlib.import_module(path)
        for name in (f"DenseDecoderRanks{world}", "DenseLayer", "MAX_ROWS", "stack_blocker", "context_kv_blocker"):
            assert hasattr(pkg, name), (path, name)
        assert stack._ROUTES.module((h, world), "host").FC_K == 5 * h
    for path in stack._LAYERS.hosts.values():
        assert hasattr(importlib.import_module(path), "DenseLayer")
    for other in (E.dense_decoder_blocker, E.prepare_dense_decoder):
        assert inspect.signature(other).parameters == inspect.signature(E.dense_decoder).parameters
    assert E.dense_decoder_shapes() == {(6144, 4), (6144, 2)}


def layer(i):
    return E.dense_decoder_layer(t(6144), (2560, 6144), t(6144), t(2 * i, 6144, dtype=E4), t(1, dtype=U8),
                                 t(6144, i, dtype=E4), t(1, dtype=U8))


@pytest.mark.parametrize("key", KEYS, ids=[f"ranks{k[1]}" for k in KEYS])
def test_covered_routes(gfx950, monkeypatch, key):
    h, world = key
    pkg = importlib.import_module(stack._ROUTES.hosts[key])
    i = 3072 if world == 4 else 6144
    lay = layer(i)
    assert type(lay) is pkg.DenseLayer
    fc, emb = t(h, 5 * h), t(1000, h)
    seen = []
    monkeypatch.setattr(pkg, "stack_blocker", lambda *a: seen.append(a))
    assert E.dense_decoder_stack_blocker(world, 64, [True], fc, emb, [lay]) is None
    assert seen == [(world, 64, [True], fc, emb, [lay])]
    monkeypatch.setattr(stack, "dist", group(world))
    monkeypatch.setattr(pkg, f"DenseDecoderRanks{world}", lambda *a, **k: ("block", a, k))
    block = E.prepare_dense_decoder(0, "g", "cuda", fc_weight=fc, embedding=emb)
    assert block == ("block", (0, "g", "cuda"), {"fc_weight": fc, "embedding": emb, "max_rows": pkg.MAX_ROWS})


def test_uncovered_and_reasons(gfx950, monkeypatch):
    fc, emb = t(6144, 30720), t(1000, 6144)
    why = E.dense_decoder_stack_blocker(8, 64, [True], fc, emb, [])
    assert "no kernel" in why and "covered:" in why
    monkeypatch.setattr(stack, "dist", group(8))
    assert E.prepare_dense_decoder(0, "g", "cuda", fc_weight=fc, embedding=emb) == why
    with pytest.raises(ValueError):
        E.dense_decoder(0, "g", "cuda", fc_weight=fc, embedding=emb)
    monkeypatch.setattr(stack, "dist", group(4, rank=2))
    assert "rank 0" in E.dense_decoder_blocker(0, "g", "cuda", fc_weight=fc, embedding=emb)
    assert "fc weight" in E.dense_decoder_blocker(2, "g", "cuda", fc_weight=t(6144, 6144), embedding=emb)
    assert "embedding" in E.dense_decoder_blocker(2, "g", "cuda", fc_weight=fc, embedding=t(1000, 4096))
    assert "rows per call" in E.dense_decoder_blocker(2, "g", "cuda", fc_weight=fc, embedding=emb, max_rows=10**6)
    assert E.dense_decoder_blocker(2, "g", "cuda", fc_weight=fc, embedding=emb) is None
    bad = E.dense_decoder_layer(t(4096), (2560, 6144), t(6144), t(6144, 6144, dtype=E4), t(1, dtype=U8),
                                t(6144, 3072, dtype=E4), t(1, dtype=U8))
    assert "layer 0" in E.dense_decoder_stack_blocker(4, 64, [True], fc, emb, [bad])
    with pytest.raises(ValueError) as e:
        layer(1024)
    assert "dense_decoder_layer: no kernel" in str(e.value)


def test_context_kv_reason(monkeypatch):
    assert "not built" in E.dense_decoder_context_kv_blocker(object(), True, t(1, 1), 64, True, t(1, 64), [], [], [])
    pkg = importlib.import_module(stack._ROUTES.hosts[(6144, 4)])
    block = object.__new__(pkg.DenseDecoderRanks4)
    assert "no KV caches" in E.dense_decoder_context_kv_blocker(block, True, t(1, 1), 64, True, t(1, 64), [], [], [])
    why = E.dense_decoder_context_kv_blocker(block, False, t(1, 1), 64, True, t(1, 64), [t(1)], [t(1)], [t(1)])
    assert why == "dense_decoder_context_kv: the context projection does not run gemm.mxfp8_common"


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert E.dense_decoder_stack_blocker(4, 64, [True], t(6144, 30720), t(10, 6144), []).endswith(
        "needs gfx950 with FlyDSL")


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.decoder.dense_mxfp8_decoder_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
