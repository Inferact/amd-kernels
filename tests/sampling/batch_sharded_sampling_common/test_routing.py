# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the batch-sharded sampling entries on the host (a stub group, no GPU)."""

import importlib
import inspect
import subprocess
import sys
import types

import pytest
import torch

from amd_kernels import routes
from amd_kernels.sampling import batch_sharded_sampling_common as E
from amd_kernels.sampling.batch_sharded_sampling_common import host

KEYS = sorted(host._ROUTES.hosts)


def group(world, rank=0):
    return types.SimpleNamespace(get_world_size=lambda g: world, get_rank=lambda g: rank)


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def test_tables_and_signatures():
    for path in host._ROUTES.hosts.values():
        pkg = importlib.import_module(path)
        for name in ("BatchShardedSampling", "sampling_blocker", "markov_blocker", "MAX_REQS", "MAX_ROWS"):
            assert hasattr(pkg, name), (path, name)
    for other in (E.batch_sharded_sampling_blocker, E.prepare_batch_sharded_sampling):
        assert inspect.signature(other).parameters == inspect.signature(E.batch_sharded_sampling).parameters
    assert E.batch_sharded_sampling_shapes() == {(200064, 4), (200064, 2)}


@pytest.mark.parametrize("key", KEYS, ids=[f"ranks{k[1]}" for k in KEYS])
def test_covered_routes(gfx950, monkeypatch, key):
    vocab, world = key
    pkg = importlib.import_module(host._ROUTES.hosts[key])
    assert E.batch_sharded_sampling_config_blocker(vocab, world, 64, 4) is None
    monkeypatch.setattr(host, "dist", group(world))

    class Stub(pkg.BatchShardedSampling):
        def __init__(self, *a, **k):
            self.args = (a, k)

    Stub.__module__ = pkg.__name__
    monkeypatch.setattr(pkg, "BatchShardedSampling", Stub)
    s = E.prepare_batch_sharded_sampling(0, "g", "cuda", vocab=vocab, max_reqs=64, target_rows=256)
    assert s.args == ((0, "g", "cuda"), {"max_reqs": 64, "target_rows": 256})
    seen = []
    monkeypatch.setattr(pkg, "markov_blocker", lambda *a, **k: seen.append((a, k)) or "too many steps")
    assert E.batch_sharded_markov_blocker(s, 3, 64, "w1", "w2") == "batch_sharded_markov: too many steps"
    assert seen[0][0] == (3, 64, "w1", "w2") and seen[0][1]["lm_weight_shuffled"] is True


def test_uncovered_and_reasons(gfx950, monkeypatch):
    why = E.batch_sharded_sampling_config_blocker(129280, 4, 64, 4)
    assert "no kernel" in why and "covered:" in why
    assert "logits rows per request" in E.batch_sharded_sampling_config_blocker(200064, 4, 64, 100)
    monkeypatch.setattr(host, "dist", group(8))
    why = E.prepare_batch_sharded_sampling(0, "g", "cuda", vocab=200064, max_reqs=64, target_rows=256)
    assert "no kernel" in why
    monkeypatch.setattr(host, "dist", group(4, rank=1))
    assert "rank 0" in E.batch_sharded_sampling_blocker(0, "g", "cuda", vocab=200064, max_reqs=64, target_rows=256)
    assert "requests" in E.batch_sharded_sampling_blocker(1, "g", "cuda", vocab=200064, max_reqs=1, target_rows=256)
    lm = torch.empty(1, device="meta")
    assert "lm_weight" in E.batch_sharded_sampling_blocker(1, "g", "cuda", vocab=200064, max_reqs=64, target_rows=256,
                                                           lm_weight=lm)
    monkeypatch.setattr(host, "dist", group(2))
    assert E.batch_sharded_sampling_blocker(0, "g", "cuda", vocab=200064, max_reqs=64, target_rows=256,
                                            lm_weight=lm) is None
    assert "not built" in E.batch_sharded_markov_blocker(object(), 3, 64, None, None)


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert E.batch_sharded_sampling_config_blocker(200064, 4, 64, 4).endswith("needs gfx950 with FlyDSL")


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.sampling.batch_sharded_sampling_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
