# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the sequence-parallel hand-off entries on the host (a stub group, no GPU)."""

import importlib
import inspect
import subprocess
import sys
import types

import pytest
import torch

from amd_kernels import routes
from amd_kernels.collectives import seq_parallel_handoffs_common as E
from amd_kernels.collectives.seq_parallel_handoffs_common import host

KEYS = sorted(host._ROUTES.hosts)


def group(world, rank=0):
    return types.SimpleNamespace(get_world_size=lambda g: world, get_rank=lambda g: rank)


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def test_tables_and_signatures():
    for (h, q, world), path in host._ROUTES.hosts.items():
        pkg = importlib.import_module(path)
        assert (pkg.H, pkg.Q, pkg.WORLD) == (h, q, world)
        for name in ("SeqParallelHandoffs", "SdmaSeqParallelHandoffs", "MIN_TOKENS", "MAX_TOKENS", "shard"):
            assert hasattr(pkg, name), (path, name)
    for other in (E.seq_parallel_handoffs_blocker, E.prepare_seq_parallel_handoffs):
        assert inspect.signature(other).parameters == inspect.signature(E.seq_parallel_handoffs).parameters
    assert E.seq_parallel_handoffs_shapes() == set(KEYS)


@pytest.mark.parametrize("sdma", [False, True])
@pytest.mark.parametrize("key", KEYS, ids=[f"ranks{k[2]}" for k in KEYS])
def test_covered_routes(gfx950, monkeypatch, key, sdma):
    h, q, world = key
    pkg = importlib.import_module(host._ROUTES.hosts[key])
    assert E.seq_parallel_handoffs_layer_blocker(world, h, q) is None
    assert E.seq_parallel_min_tokens(world, h, q) == pkg.MIN_TOKENS
    monkeypatch.setattr(host, "dist", group(world))
    name = "SdmaSeqParallelHandoffs" if sdma else "SeqParallelHandoffs"

    class Stub(getattr(pkg, name)):
        def __init__(self, *a):
            self.args = a

    Stub.__module__ = pkg.__name__
    monkeypatch.setattr(pkg, name, Stub)
    ph = E.prepare_seq_parallel_handoffs(0, "g", "cuda", hidden=h, q_per_rank=q, sdma=sdma)
    assert isinstance(ph, Stub) and ph.args == (0, "g", "cuda", pkg.MAX_TOKENS)
    x = torch.arange(10 * 3).view(10, 3)
    assert torch.equal(E.seq_parallel_shard(ph, x, 1), pkg.shard(x, 1))


def test_uncovered_and_reasons(gfx950, monkeypatch):
    why = E.seq_parallel_handoffs_layer_blocker(8, 6144, 1024)
    assert "no kernel" in why and "covered:" in why
    monkeypatch.setattr(host, "dist", group(4, rank=1))
    assert E.prepare_seq_parallel_handoffs(1, "g", "cuda", hidden=6144, q_per_rank=4096).startswith(
        "seq_parallel_handoffs: no kernel")
    assert "rank 0" in E.seq_parallel_handoffs_blocker(0, "g", "cuda", hidden=6144, q_per_rank=2048)
    assert "max_tokens" in E.seq_parallel_handoffs_blocker(1, "g", "cuda", hidden=6144, q_per_rank=2048,
                                                           max_tokens=10**7)
    with pytest.raises(ValueError):
        E.seq_parallel_shard(object(), torch.zeros(4, 3), 0)


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert E.seq_parallel_handoffs_layer_blocker(4, 6144, 2048).endswith("needs gfx950 with FlyDSL")


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.collectives.seq_parallel_handoffs_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
