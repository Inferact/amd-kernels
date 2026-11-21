# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of ``vocab_gather_comm`` and the gather's calls, host only (meta tensors; the group's size is
stubbed): every table module exports ``VocabGather``, each (hidden, world size) builds its package's object, an
uncovered shape gets a reason (and the factory a ValueError with it), every host-checkable reason branch, the
reasons' signatures, and importing the entry imports no shape package."""

import importlib
import inspect
import subprocess
import sys
import types

import pytest
import torch

from amd_kernels.embedding.vocab_gather_common import host as E

KEYS = sorted(E._SHAPE_HOSTS)


@pytest.fixture
def world(monkeypatch):
    """``group`` = its world size; this process is rank 0."""
    monkeypatch.setattr(E, "available", lambda: True)
    fake = types.SimpleNamespace(get_world_size=lambda group: group, get_rank=lambda group: 0,
                                 is_initialized=lambda: True)
    monkeypatch.setattr(E, "dist", fake)
    return fake


def shard(hidden, vocab=128, dtype=torch.bfloat16):
    return torch.empty(vocab, hidden, dtype=dtype, device="meta")


def test_table():
    assert E.vocab_gather_shapes() == frozenset(KEYS)
    for name in E._SHAPE_HOSTS.values():
        assert callable(importlib.import_module(name).VocabGather)


@pytest.mark.parametrize("key", KEYS, ids=str)
def test_factory_routes(key, world, monkeypatch):
    pkg = importlib.import_module(E._SHAPE_HOSTS[key])
    assert E._shape_host(*key) is pkg
    built = []
    monkeypatch.setattr(pkg, "VocabGather", lambda *a: built.append(a) or "gather")
    w = shard(key[0])
    assert E.vocab_gather_comm(w, 0, key[1]) == "gather" and built == [(w, 0, key[1])]
    assert E.vocab_gather_comm_unsupported_reason(w, 0, key[1]) == "vocab_gather_comm: weight on meta (a CUDA device)"


@pytest.mark.parametrize("hidden,ranks", [(7168, 4), (6144, 8), (4096, 8)])
def test_uncovered_shape(hidden, ranks, world):
    why = E.vocab_gather_comm_unsupported_reason(shard(hidden), 0, ranks)
    assert f"no kernel for hidden={hidden}, world={ranks}" in why and str(KEYS) in why
    with pytest.raises(ValueError) as e:
        E.vocab_gather_comm(shard(hidden), 0, ranks)
    assert str(e.value) == why


def test_factory_reasons(world, monkeypatch):
    why = E.vocab_gather_comm_unsupported_reason
    assert "weight torch.float32" in why(shard(7168, dtype=torch.float32), 0, 8)
    assert "weight torch.bfloat16 (2, 64, 7168)" in why(shard(7168).view(2, 64, 7168), 0, 8)
    assert "contiguous" in why(shard(7168 * 2)[:, ::2], 0, 8)
    assert "weight torch.bfloat16 (7168,)" in why(torch.empty(7168, dtype=torch.bfloat16, device="meta"), 0, 8)
    monkeypatch.setattr(world, "is_initialized", lambda: False)
    assert why(shard(7168), 0, 8) == "vocab_gather_comm: torch.distributed is not initialized"
    monkeypatch.setattr(E, "available", lambda: False)
    assert why(shard(7168), 0, 8) == "vocab_gather_comm: needs gfx950 with FlyDSL"


def test_call_reasons():
    pkg = importlib.import_module(E._SHAPE_HOSTS[(7168, 8)])
    gather = object.__new__(pkg.VocabGather)
    gather.weight = shard(7168)
    why = E.vocab_gather_unsupported_reason
    ids = torch.empty(40, dtype=torch.int32, device="meta")
    assert why(gather, ids) is None and why(gather, ids.long()) is None
    assert why(object(), ids) == "vocab_gather: object was not built by vocab_gather_comm"
    assert "ids torch.float32" in why(gather, ids.float())
    assert "ids torch.int32 (4, 10)" in why(gather, ids.view(4, 10))
    assert "contiguous" in why(gather, ids[::2])
    assert why(gather, torch.empty(40, dtype=torch.int32)) == "vocab_gather: ids on cpu, weight on meta"


def test_reasons_take_the_calls_arguments():
    def sig(f):
        return list(inspect.signature(f).parameters.values())

    assert sig(E.vocab_gather_comm) == sig(E.vocab_gather_comm_unsupported_reason)
    assert [p.name for p in sig(E.vocab_gather_unsupported_reason)] == ["gather", "ids"]


def test_entry_import_imports_no_shape_package():
    code = ("import sys, amd_kernels.embedding.vocab_gather_common as e; "
            "print(sorted(m for m in sys.modules if m.startswith('amd_kernels.embedding.vocab_gather_h')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
