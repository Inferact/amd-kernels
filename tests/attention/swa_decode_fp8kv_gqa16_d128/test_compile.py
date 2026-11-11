# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

A served configuration fixes the rows per request, the KV heads, the page size, the
window, the causal mode and the XCD count; the request count R picks the split count
(``host.plan_splits``: one round of workgroups on the 256 CUs, the grid padded to the 8
XCDs), and the split count picks the row grouping. ``variant`` names that choice. Per
configuration: warm one R per variant over the served range, then probe other R; a probe
that compiles means R itself (or something derived from it beyond the split count)
reaches a compile key.
"""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import host

from .test_decode import launch, make_batch, require_gfx950

PROBES = 6
# (id, rows per request, KV heads, page, causal, served request counts)
SERVED = [
    ("tp4_spec7_p128", 7, 1, 128, True, range(1, 257)),
    # upstream vLLM's DSpark draft block (non-causal), TP4 up to 256 requests
    ("tp4_spec7_noncausal_p128", 7, 1, 128, False, range(1, 257)),
    ("tp4_spec3_p128", 4, 1, 128, True, range(1, 257)),
]


def count_compiles(monkeypatch):
    compiled = []
    real = flyc.compile

    def counting(fn, *args, **kwargs):
        compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    return compiled


def sub_batch(batch, reqs):
    """The first ``reqs`` requests of ``batch``."""
    pool, bt, seq_lens, qsl, q, qlens = batch
    rows = int(qsl[reqs])
    return [pool, bt[:reqs], seq_lens[:reqs], qsl[: reqs + 1], q[:rows], qlens[:reqs]]


@pytest.mark.parametrize("name,ql,hkv,page,causal,served", SERVED, ids=[s[0] for s in SERVED])
@torch.inference_mode()
def test_no_compile_after_warmup(name, ql, hkv, page, causal, served, monkeypatch):
    require_gfx950()
    gen = torch.Generator().manual_seed(len(served) + hkv)
    full = make_batch(served[-1], ql, 0, gen, page, hkv)
    q = full[4]
    out = torch.empty_like(q)

    def variant(reqs):
        tile = host._tp4_q7_rows(reqs) if ql == 7 and hkv == 1 and page == 128 else 0
        return host.launch_splits(q, full[0], reqs), tile

    def run(reqs):
        batch = sub_batch(full, reqs)
        launch(batch, ql, causal, out[: reqs * ql])

    warm = {}
    for reqs in served:
        warm.setdefault(variant(reqs), reqs)
    for reqs in warm.values():
        run(reqs)
    torch.cuda.synchronize()
    rest = [r for r in served if r not in warm.values()]
    probes = rest[:: max(1, len(rest) // PROBES)][:PROBES]
    compiled = count_compiles(monkeypatch)
    for reqs in probes:
        run(reqs)
        assert not compiled, (
            f"{name}: {reqs} requests (splits {variant(reqs)}) compiled {compiled} "
            f"after warm-up on {sorted(warm.items())}"
        )
    print(f"{name}: split counts {sorted(warm)} warmed at {sorted(warm.values())}; probed {probes}")
