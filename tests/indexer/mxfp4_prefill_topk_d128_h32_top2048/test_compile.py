# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4): no compile key holds a row count or a key count."""

import flydsl.compiler as flyc
import torch

from amd_kernels import warm_up
from amd_kernels.indexer.mxfp4_prefill_topk_d128_h32_top2048 import host

from .inputs import make_logits, make_windows, require_gfx950

PROBE = [(1, 64), (5, 700), (8, 6200), (64, 9000), (200, 40000), (300, 150000)]


def test_one_variant_per_kernel(monkeypatch):
    require_gfx950()
    warm_up.set_auto(True)
    try:
        logits = make_logits("gauss", 4, 4096)
        ks, ke = make_windows(4, 4096, "causal")
        host.topk_rows(logits, ks, ke, torch.empty(4, 2048, dtype=torch.int32, device="cuda"))
        compiled = []
        real = flyc.compile
        monkeypatch.setattr(flyc, "compile", lambda fn, *a, **k: (compiled.append(fn), real(fn, *a, **k))[1])
        for rows, width in PROBE:
            logits = make_logits("gauss", rows, width)
            ks, ke = make_windows(rows, width, "causal")
            host.topk_rows(logits, ks, ke, torch.empty(rows, 2048, dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()
        assert not compiled
        assert warm_up.late_compiles() == 0
    finally:
        warm_up.set_auto(False)
