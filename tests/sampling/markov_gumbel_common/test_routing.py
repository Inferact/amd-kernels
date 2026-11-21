# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of ``markov_gumbel_fused`` on meta tensors (host only): the table module exports the op, the
(vocab, rank) of ``w1`` routes to its package (op, scratch size, row limit), an uncovered shape gets a reason (and
the op a ValueError with it), every reason branch, the reason's signature is the op's, and importing the entry
imports no shape package."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels.sampling.markov_gumbel_common import host as E

KEYS = sorted(E._SHAPE_HOSTS)
NAMES = list(inspect.signature(E.markov_gumbel_fused).parameters)


@pytest.fixture
def on_gfx950(monkeypatch):
    """gfx950, and meta tensors stand for tensors on the current device."""
    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(E, "_off_device", lambda **tensors: None)


def meta(*s, dtype=torch.int64):
    return torch.empty(*s, dtype=dtype, device="meta")


def call(n=8, k=3, vocab=163840, rank=512, **change):
    m = meta
    args = {
        "base": m(n, k, vocab, dtype=torch.bfloat16), "w1": m(vocab, rank, dtype=torch.bfloat16),
        "w2": m(vocab, rank, dtype=torch.bfloat16), "input_ids": m(n + 4, dtype=torch.int32), "anchor_idx": m(n),
        "draft_tokens": m(64, k), "idx_mapping": m(n, k), "sample_pos": m(n, k),
        "temperature": m(64, dtype=torch.float32), "seeds": m(64), "cache": m(64, k + 1, vocab, dtype=torch.bfloat16),
        "scratch": m(1 << 16, dtype=torch.int32), "row_mapping": m(64, dtype=torch.int32),
    }
    args.update(change)
    return [args[name] for name in NAMES]


def reason(*args):
    return E.markov_gumbel_fused_unsupported_reason(*args)


def test_table():
    assert E.markov_gumbel_fused_shapes() == frozenset(KEYS)
    for key, name in E._SHAPE_HOSTS.items():
        pkg = importlib.import_module(name)
        assert (pkg.V, pkg.RANK) == key and callable(pkg.markov_gumbel_fused) and callable(pkg.scratch_dwords)


@pytest.mark.parametrize("key", KEYS, ids=str)
def test_routes_by_w1(key, on_gfx950, monkeypatch):
    pkg = importlib.import_module(E._SHAPE_HOSTS[key])
    assert E._shape_host(*key) is pkg
    args = call(vocab=key[0], rank=key[1])
    w1 = args[1]
    assert E.markov_max_rows(w1) == pkg.MAX_ROWS
    assert E.markov_scratch_dwords(w1, 20) == pkg.scratch_dwords(20)
    calls = []
    monkeypatch.setattr(pkg, "markov_gumbel_fused", lambda *a: calls.append(a))
    assert reason(*args) is None
    E.markov_gumbel_fused(*args)
    assert len(calls) == 1 and all(a is b for a, b in zip(calls[0], args))
    assert reason(*call(n=pkg.MAX_ROWS, k=pkg.host.MAX_STEPS, vocab=key[0], rank=key[1])) is None


def test_uncovered_shape(on_gfx950):
    args = call(vocab=163840, rank=256)
    why = reason(*args)
    assert "no kernel for (vocab, rank)=(163840, 256)" in why and str(KEYS) in why
    for fn in (lambda: E.markov_gumbel_fused(*args), lambda: E.markov_max_rows(args[1]),
               lambda: E.markov_scratch_dwords(args[1], 8)):
        with pytest.raises(ValueError) as e:
            fn()
        assert str(e.value) == why


def test_needs_gfx950(monkeypatch):
    monkeypatch.setattr(E, "available", lambda: False)
    assert reason(*call()) == "markov_gumbel_fused: needs gfx950 with FlyDSL"


BF = torch.bfloat16
CASES = [
    ({"w1": meta(163840, 512)}, "w1 torch.int64"),
    ({"w2": meta(163840, 512)}, "w2 torch.int64"),
    ({"w2": meta(512, 163840, dtype=BF).t()}, "w2 torch.bfloat16"),
    ({"w1": meta(512, 163840, dtype=BF).t()}, "w1 torch.bfloat16"),
    ({"base": meta(8, 3, 163840)}, "base torch.int64"),
    ({"base": meta(8, 3, 163840 * 2, dtype=BF)[:, :, :163840]}, "step stride 163840"),
    ({"base": meta(8, 163840, 3, dtype=BF).transpose(1, 2)}, "unit column stride"),
    ({"base": meta(49, 3, 163840, dtype=BF)}, "49 rows (the kernel takes 1..48)"),
    ({"base": meta(8, 9, 163840, dtype=BF)}, "9 steps (the kernel takes 1..8)"),
    ({"input_ids": meta(12)}, "input_ids torch.int64"),
    ({"anchor_idx": meta(7)}, "anchor_idx torch.int64 (7,)"),
    ({"anchor_idx": meta(8, dtype=torch.int32)}, "anchor_idx torch.int32"),
    ({"draft_tokens": meta(64, 2)}, "draft_tokens"),
    ({"draft_tokens": meta(7, 3)}, "draft_tokens"),
    ({"idx_mapping": meta(3, 8).t()}, "idx_mapping"),
    ({"sample_pos": meta(8, 3, dtype=torch.int32)}, "sample_pos"),
    ({"temperature": meta(64, dtype=BF)}, "temperature torch.bfloat16"),
    ({"seeds": meta(64, dtype=torch.int32)}, "seeds torch.int32"),
    ({"cache": meta(64, 2, 163840, dtype=BF)}, "cache torch.bfloat16 (64, 2, 163840)"),
    ({"cache": meta(64, 4, 163840)}, "cache torch.int64"),
    ({"cache": meta(64, 4, 163840 * 2, dtype=BF)[:, :, ::2]}, "unit column stride"),
    ({"cache": meta(6554, 4, 163840, dtype=BF)}, "4 GiB"),
    ({"row_mapping": meta(64)}, "row_mapping torch.int64"),
    ({"scratch": meta(1 << 16)}, "scratch torch.int64"),
    ({"scratch": meta(100, dtype=torch.int32)}, "scratch torch.int32 of 100"),
]


@pytest.mark.parametrize("change,text", CASES, ids=[t for _, t in CASES])
def test_reason_branches(change, text, on_gfx950):
    why = reason(*call(**change))
    assert why is not None and why.startswith("markov_gumbel_fused: ") and text in why, why


def test_row_mapping_optional(on_gfx950):
    assert reason(*call(row_mapping=None)) is None


def test_device_reasons(monkeypatch):
    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert reason(*call()) == "markov_gumbel_fused: base on meta (the current CUDA device)"
    assert E._off_device(row_mapping=None, seeds=torch.empty(4)) == \
        "markov_gumbel_fused: seeds on cpu (the current CUDA device)"


def test_reason_takes_the_ops_arguments():
    op, why = inspect.signature(E.markov_gumbel_fused), inspect.signature(E.markov_gumbel_fused_unsupported_reason)
    assert list(op.parameters.values()) == list(why.parameters.values())


def test_entry_import_imports_no_shape_package():
    code = ("import sys, amd_kernels.sampling.markov_gumbel_common as e; "
            "print(sorted(m for m in sys.modules if m.startswith('amd_kernels.sampling.markov_gumbel_fused')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
