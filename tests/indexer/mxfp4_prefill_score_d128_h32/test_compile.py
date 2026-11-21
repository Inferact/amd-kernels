# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4): no compile key holds a row count or a key count."""

import flydsl.compiler as flyc
import torch
from amd_kernels import warm_up
from amd_kernels.indexer.mxfp4_prefill_score_d128_h32 import host

from .inputs import make_inputs, require_gfx950

PROBE = [(1, 64), (5, 700), (8, 513), (64, 9000), (200, 40000), (2048, 70000)]


def test_one_variant(monkeypatch):
    require_gfx950()
    warm_up.set_auto(True)
    try:
        inputs = make_inputs(8, 4096)
        out = torch.empty(8, 4096, dtype=torch.float32, device="cuda")
        host.score_rows(*inputs, out)
        compiled = []
        real = flyc.compile
        monkeypatch.setattr(flyc, "compile", lambda fn, *a, **k: (compiled.append(fn), real(fn, *a, **k))[1])
        for rows, keys in PROBE:
            inputs = make_inputs(rows, keys)
            out = torch.empty(rows, inputs[3].shape[0] * 64, dtype=torch.float32, device="cuda")
            host.score_rows(*inputs, out)
        torch.cuda.synchronize()
        assert not compiled
        assert warm_up.late_compiles() == 0
    finally:
        warm_up.set_auto(False)
