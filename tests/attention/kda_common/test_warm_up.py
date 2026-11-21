# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Through the KDA entries: the first call warms the routed package up (``amd_kernels.warm_up``), every served row
count after it compiles nothing (``compile_watch``), and a CUDA graph of the entry call, captured with imports
disabled, replays the eager call bit for bit."""

import importlib

import pytest
import torch

from amd_kernels import compile_watch, warm_up
from amd_kernels.attention import kda_common as E

from . import cases

DECODE_ROWS = [1, 6, 12, 24, 40, 65, 80, 128, 192, 256]
PREFILL_SEQS = [[300], [64], [4096], [300, 64, 77, 1], [1500, 700, 64, 3000], [10240]]
MONO_ROWS = [4, 8, 12, 16]


@pytest.fixture
def auto_warm_up():
    if not E.available():
        pytest.skip("needs gfx950 with FlyDSL")
    warm_up.set_auto(True)
    yield
    warm_up.set_auto(False)


def walk(name, builds):
    """First call warms ``name`` up; the rest, with compile_watch armed, compile nothing."""
    builds[0]()[0].run(True)
    torch.cuda.synchronize()
    assert any(entry == name for entry, _ in warm_up.warmed()), warm_up.warmed()
    warned, late = [], warm_up.late_compiles()
    watch = compile_watch.late_compiles()
    assert compile_watch.arm(warned.append)
    try:
        for build in builds[1:]:
            build()[0].run(True)
        torch.cuda.synchronize()
    finally:
        compile_watch.disarm()
    assert not warned and compile_watch.late_compiles() == watch and warm_up.late_compiles() == late, warned


def graph_replays_eager(build, monkeypatch):
    a, _ = build()
    b, _ = build()
    a.run(True)

    def no_import(name, *args, **kwargs):
        raise AssertionError(f"import of {name} inside capture")

    g = torch.cuda.CUDAGraph()
    with monkeypatch.context() as m:
        m.setattr(importlib, "import_module", no_import)
        with torch.cuda.graph(g):
            b.run(True)
    g.replay()
    torch.cuda.synchronize()
    for x, y in zip(a.outputs(), b.outputs(), strict=True):
        assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


@torch.inference_mode()
def test_decode(auto_warm_up, monkeypatch):
    for uniform, rows in ((False, DECODE_ROWS), (True, [r for r in DECODE_ROWS if r % 6 == 0])):
        walk("attention/kda_decode_fb_gate_spec_h12_d128",
             [lambda r=r, u=uniform: cases.decode(r, 6, u, seed=r) for r in rows])
    graph_replays_eager(lambda: cases.decode(40, 6, False, seed=3), monkeypatch)


@torch.inference_mode()
def test_prefill(auto_warm_up, monkeypatch):
    walk("attention/kda_prefill_h12_d128", [lambda s=s: cases.prefill(s) for s in PREFILL_SEQS])
    graph_replays_eager(lambda: cases.prefill([300, 64, 77, 1]), monkeypatch)
    graph_replays_eager(lambda: cases.prefill([4096]), monkeypatch)


@torch.inference_mode()
def test_mono(auto_warm_up, monkeypatch):
    walk("attention/kda_mono_decode_k1_h12_d128_ranks8", [lambda s=s: cases.mono(s, 4) for s in MONO_ROWS])
    graph_replays_eager(lambda: cases.mono(12, 4), monkeypatch)
