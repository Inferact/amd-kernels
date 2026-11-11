# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4), one-GPU kernels; the 4-rank kernels' checks are in their tests.

No compile key holds the token count: ``gemma_add_rmsnorm``, ``gemma_add_rmsnorm_mxfp8`` (tiled codes) and
``gemma_add_rmsnorm_mxfp8_row_major`` compile once each. The model's warm-up (``warmup_mxfp8_gemm_weight`` ->
``warmup_for_gemm``) runs ``warmup_add_rmsnorm_mxfp8`` (both code layouts); the bf16 norm compiles at its first call
(the profile run). Warm those, then probe the served token range.
"""

import logging
import warnings

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import host

from .test_add_rmsnorm import EPS, H, make_rows, make_weight, require_gfx950

PROBE_M = [1, 3, 24, 63, 100, 172, 257, 1000, 4096, 8191]


# the host's cached kernel getters; their arguments are the compile keys
GETTERS = (
    "_add_rmsnorm_kernel",
    "_add_rmsnorm_mxfp8_kernel",
    "_add_rmsnorm_mxfp8_row_major_kernel",
)
# the variants the served range reaches: the test's mirror of the keys
EXPECTED_KEYS = {(name, ()) for name in GETTERS}


@pytest.fixture
def keys(monkeypatch):
    """Fresh getters (earlier tests' compiled kernels would hide a warm-up gap), spied: the list of (getter, key) the
    host asks for."""
    seen = []
    for name in GETTERS:
        real = getattr(host, name)
        real.cache_clear()

        def spy(*key, _real=real, _name=name):
            seen.append((_name, tuple(bool(k) if isinstance(k, bool) else k for k in key)))
            return _real(*key)

        monkeypatch.setattr(host, name, spy)
    return seen


class CountCompiles:
    def __init__(self, monkeypatch):
        self.compiled = []
        real = flyc.compile

        def counting(fn, *args, **kwargs):
            self.compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
            return real(fn, *args, **kwargs)

        monkeypatch.setattr(flyc, "compile", counting)


def serving_warmup():
    """The model's warm-up, plus the first call of the bf16 norm."""
    an.warmup_add_rmsnorm_mxfp8()
    x = torch.zeros((1, H), dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(H, dtype=torch.bfloat16, device="cuda")
    an.gemma_add_rmsnorm(x, x, w, 1e-6)
    torch.cuda.synchronize()


def every_variant(m, gen):
    x, r, w = make_rows(m, gen, "randn"), make_rows(m, gen, "randn"), make_weight(gen)
    an.gemma_add_rmsnorm(x, r, w, EPS)
    an.gemma_add_rmsnorm_mxfp8(x, r, w, EPS, 64)
    an.gemma_add_rmsnorm_mxfp8_row_major(x, r, w, EPS)


@torch.inference_mode()
def test_no_compile_after_warmup(monkeypatch, keys):
    require_gfx950()
    counter = CountCompiles(monkeypatch)
    serving_warmup()
    warm = set(keys)
    assert warm == EXPECTED_KEYS, sorted(warm ^ EXPECTED_KEYS)
    assert len(counter.compiled) == len(EXPECTED_KEYS), counter.compiled  # one each
    counter.compiled.clear()
    gen = torch.Generator().manual_seed(5)
    for m in PROBE_M:
        keys.clear()
        every_variant(m, gen)
        assert set(keys) <= warm, f"m={m} reached {set(keys) - warm}"
        assert not counter.compiled, f"m={m} compiled {counter.compiled} after warm-up"
    torch.cuda.synchronize()


@pytest.mark.xfail(
    strict=True,
    reason="_run_compiled compiles silently on first use; rule 4 asks "
    "for one warning per late compile (kernel name, variant key, shapes, call site, "
    "compile time)",
)
@torch.inference_mode()
def test_late_compile_logs_warning(caplog, keys):
    require_gfx950()
    gen = torch.Generator().manual_seed(6)
    x, r, w = make_rows(8, gen, "randn"), make_rows(8, gen, "randn"), make_weight(gen)
    # no warm-up (the fixture cleared the getters): the first call compiles late
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        an.gemma_add_rmsnorm(x, r, w, EPS)
    torch.cuda.synchronize()
    assert ("_add_rmsnorm_kernel", ()) in keys
    exe = host._add_rmsnorm_kernel()
    assert exe._cf is not None  # it did compile
    name = exe.kernel_name
    logged = [rec.getMessage() for rec in caplog.records] + [str(c.message) for c in caught]
    assert any(name in msg for msg in logged), f"no warning names {name}: {logged}"
