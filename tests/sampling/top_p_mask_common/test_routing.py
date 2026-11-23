# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of ``top_p_mask`` on meta tensors (host only): every table module exports the op, each
vocabulary routes to its package, an uncovered call gets a reason (and the op a ValueError with it), every reason
branch, the reason's signature is the op's, and importing the entry imports no shape package."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.sampling.top_p_mask_common import host as E

KEYS = sorted(E._ROUTES.hosts)
OFF_DEVICE = E._off_device


@pytest.fixture
def on_gfx950(monkeypatch):
    """gfx950, and meta tensors stand for tensors on the current device."""
    monkeypatch.setattr(routes, "available", lambda: True)
    monkeypatch.setattr(E, "_off_device", lambda **tensors: None)
    for name in set(E._ROUTES.hosts.values()):
        pkg = importlib.import_module(name)
        monkeypatch.setattr(pkg, "available", lambda: True)
        if hasattr(pkg, "rows_grid"):
            monkeypatch.setattr(pkg, "rows_grid", lambda n, index: min(n, 256))


def rows(n, vocab):
    return torch.empty(n, vocab, device="meta"), torch.empty(n, device="meta")


def reason(*args, **kwargs):
    return E.top_p_mask_blocker(*args, **kwargs)


def test_table():
    assert E.top_p_mask_shapes() == frozenset(KEYS)
    for (vocab,), name in E._ROUTES.hosts.items():
        pkg = importlib.import_module(name)
        assert pkg.V == vocab and callable(pkg.top_p_mask) and callable(pkg.top_p_blocker)
        assert pkg.SPLIT_MAX_BATCH == 64


@pytest.mark.parametrize("key", KEYS, ids=str)
def test_routes_by_vocab(key, on_gfx950, monkeypatch):
    pkg = importlib.import_module(E._ROUTES.hosts[key])
    assert E._ROUTES.host(key) is pkg
    calls = []
    monkeypatch.setattr(pkg, "top_p_mask", lambda *a: calls.append(a) or a[0])
    logits, p = rows(5, key[0])
    assert reason(logits, p) is None
    assert E.top_p_mask(logits, p) is logits and calls == [(logits, p)]


def test_uncovered_vocab(on_gfx950):
    logits, p = rows(4, 151936)
    why = reason(logits, p)
    assert "no kernel for vocab=151936" in why and str(KEYS) in why
    with pytest.raises(ValueError) as e:
        E.top_p_mask(logits, p)
    assert str(e.value) == why


def test_needs_gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert reason(*rows(4, 163840)) == "top_p_mask: needs gfx950 with FlyDSL"


def test_package_blocker_text(on_gfx950, monkeypatch):
    pkg = importlib.import_module(E._ROUTES.hosts[(163840,)])
    monkeypatch.setattr(pkg, "top_p_blocker", lambda vocab, mask: f"blocked {vocab} {mask}")
    assert reason(*rows(4, 163840)) == "top_p_mask: blocked 163840 -inf"


@pytest.mark.parametrize("vocab", [163840, 200064])
def test_operand_reasons(vocab, on_gfx950):
    logits, p = rows(4, vocab)
    assert "logits torch.bfloat16" in reason(logits.bfloat16(), p)
    assert "logits" in reason(torch.empty(vocab, device="meta"), p)
    assert "unit column stride" in reason(torch.empty(vocab, 4, device="meta").t(), p)
    assert "p torch.float16" in reason(logits, p.half())
    assert "p torch.float32 (5,)" in reason(logits, torch.empty(5, device="meta"))
    assert "contiguous" in reason(logits, torch.empty(4, 2, device="meta")[:, 0])
    assert reason(*rows(0, vocab)).startswith("top_p_mask: 0 rows")


def test_row_limits(on_gfx950):
    assert reason(*rows(64, 163840)) is None
    assert reason(*rows(65, 163840)) == "top_p_mask: 65 rows (the kernel takes 1..64)"
    assert reason(*rows(65, 200064)) is None and reason(*rows(4096, 200064)) is None


def test_buffer(on_gfx950):
    buf = torch.empty(256, 200064, device="meta")
    assert reason(*rows(8, 163840), torch.empty(8, 163840, device="meta")) == \
        "top_p_mask: buffer (the vocab 163840 kernel takes none)"
    assert reason(*rows(8, 200064), buf[:1]) is None  # <= 64 rows: the split kernel, buffer unused
    assert reason(*rows(300, 200064), buf) is None
    assert "fp32 [>= 100, 200064]" in reason(*rows(100, 200064), buf[:99])
    assert "buffer torch.bfloat16" in reason(*rows(100, 200064), buf.bfloat16())
    assert "buffer" in reason(*rows(100, 200064), buf[:, :163840])
    assert "buffer" in reason(*rows(100, 200064), torch.empty(200064, 256, device="meta").t())


def test_device_reasons(on_gfx950, monkeypatch):
    monkeypatch.setattr(E, "_off_device", OFF_DEVICE)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert reason(*rows(4, 163840)) == "top_p_mask: logits on meta (the current CUDA device)"
    assert reason(torch.empty(4, 163840), torch.empty(4)) == "top_p_mask: logits on cpu (the current CUDA device)"


def test_reason_takes_the_ops_arguments():
    op, why = inspect.signature(E.top_p_mask), inspect.signature(E.top_p_mask_blocker)
    assert list(op.parameters.values()) == list(why.parameters.values())


def test_entry_import_imports_no_shape_package():
    code = ("import sys, amd_kernels.sampling.top_p_mask_common as e; "
            "print(sorted(m for m in sys.modules if m.startswith('amd_kernels.sampling.top_p_mask_f32')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_old_name_aliases_the_blocker():
    from amd_kernels.sampling.top_p_mask_common import top_p_mask_unsupported_reason
    assert top_p_mask_unsupported_reason.renamed_to == "top_p_mask_blocker"


def test_old_name_aliases_the_blocker():
    from amd_kernels.sampling.top_p_mask_common import top_p_mask_unsupported_reason

    assert top_p_mask_unsupported_reason.renamed_to == "top_p_mask_blocker"
