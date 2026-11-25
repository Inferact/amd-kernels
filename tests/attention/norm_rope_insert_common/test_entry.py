# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Each entry, at the smallest and largest served row counts and both cache dtypes, writes the same bits (q and every
cache) as a direct call of the shape package it routes to, and its reason is None on those calls."""

import importlib

import pytest
import torch

from amd_kernels.attention import norm_rope_insert_common as E

from . import cases


def require_gfx950():
    if not E.available():
        pytest.skip("needs gfx950 with FlyDSL")


def same_bits(build):
    a, _ = build()
    b, _ = build()
    a.run(True)
    b.run(False)
    torch.cuda.synchronize()
    for x, y in zip(a.outputs(), b.outputs(), strict=True):
        assert x.shape == y.shape and torch.equal(x.view(torch.uint8), y.view(torch.uint8))


CASES = [(n, kind) for n in (1, 240) for kind in ("bf16", "fp8")]


@pytest.mark.parametrize("n,kind", CASES, ids=[f"{n}_{k}" for n, k in CASES])
@torch.inference_mode()
def test_qkv(n, kind):
    require_gfx950()
    _, a = cases.qkv(n, kind=kind)
    assert E.qkv_norm_rope_insert_blocker(*a) is None
    same_bits(lambda: cases.qkv(n, kind=kind))


@pytest.mark.parametrize("n,kind", CASES, ids=[f"{n}_{k}" for n, k in CASES])
@torch.inference_mode()
def test_multilayer(n, kind):
    require_gfx950()
    _, a = cases.multilayer(n, kind=kind)
    assert E.kv_norm_rope_insert_multilayer_blocker(*a) is None
    same_bits(lambda: cases.multilayer(n, kind=kind))


@torch.inference_mode()
def test_multilayer_reason_in_capture():
    """Inside CUDA graph capture, a cache set without an eager call yet gets a reason (the caller runs stock in that
    graph); after one eager call it has none."""
    require_gfx950()
    importlib.import_module(cases.PKG + ".kv_norm_rope_insert_multilayer")._tables.clear()  # freed sets' addresses
    call, a = cases.multilayer(40, seed=7)
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        why = E.kv_norm_rope_insert_multilayer_blocker(*a)
    assert why is not None and "inside CUDA graph capture" in why, why
    call.run(True)
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        why = E.kv_norm_rope_insert_multilayer_blocker(*a)
        call.run(True)
    assert why is None, why
