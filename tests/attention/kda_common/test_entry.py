# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Each KDA entry, at the smallest and largest served row counts of each path, writes the same bits (outputs and
every in-place state) as a direct call of the shape package it routes to, and its reason is None on those calls."""

import pytest
import torch

from amd_kernels.attention import kda_common as E

from . import cases


def require_gfx950():
    if not E.available():
        pytest.skip("needs gfx950 with FlyDSL")


def bits(x):
    return x.contiguous().view(torch.uint8)


def same_bits(build):
    """Entry and direct call on two builds of the same inputs."""
    a, _ = build()
    b, _ = build()
    a.run(True)
    b.run(False)
    torch.cuda.synchronize()
    for x, y in zip(a.outputs(), b.outputs(), strict=True):
        assert x.shape == y.shape and torch.equal(bits(x), bits(y))


DECODE = [(1, 1, True), (256, 1, True), (8, 8, True), (256, 8, True), (13, 6, False), (251, 6, False)]


@pytest.mark.parametrize("rows,tokens,uniform", DECODE, ids=[f"{r}x{t}{'u' if u else 'r'}" for r, t, u in DECODE])
@torch.inference_mode()
def test_decode(rows, tokens, uniform):
    require_gfx950()
    _, a = cases.decode(rows, tokens, uniform)
    assert E.kda_decode_fb_gate_spec_blocker(**a, uniform=uniform) is None
    same_bits(lambda: cases.decode(rows, tokens, uniform))


PREFILL = [
    ([1], "mega", None), ([3000, 600], "mega", None), ([64], "staged", None), ([10240], "staged", None),
    ([300, 64, 77, 1], None, None), ([2048, 300], "mega", ([1024, 256], [3, 1])),
    ([2048, 300], "staged", ([1024, 256], [3, 1])),
]  # fmt: skip


PREFILL_IDS = ["1_mega", "3600_mega", "64_staged", "10240_staged", "ragged4", "ckpt_mega", "ckpt_staged"]


@pytest.mark.parametrize("seq,impl,ck", PREFILL, ids=PREFILL_IDS)
@torch.inference_mode()
def test_prefill(seq, impl, ck):
    require_gfx950()
    _, (args, kw) = cases.prefill(seq, impl, ck)
    assert E.kda_prefill_blocker(*args, **kw) is None
    same_bits(lambda: cases.prefill(seq, impl, ck))


MONO = [(4, 4, -1, True), (16, 4, -1, True), (16, 8, -1, True), (12, 6, 4, False)]


@pytest.mark.parametrize("S,L,write_idx,delta", MONO, ids=["4x4", "16x4", "16x8", "12x6_write_no_delta"])
@torch.inference_mode()
def test_mono(S, L, write_idx, delta):
    require_gfx950()
    _, (ws, args, kw) = cases.mono(S, L, write_idx=write_idx, delta=delta)
    assert E.kda_mono_workspace_blocker("cuda", hidden=7168, heads=12, head_dim=128) is None
    assert E.kda_mono_decode_blocker(ws[True], 0, *args, **kw) is None
    a, (wa, _, _) = cases.mono(S, L, write_idx=write_idx, delta=delta)
    b, (wb, _, _) = cases.mono(S, L, write_idx=write_idx, delta=delta)
    a.run(True)
    b.run(False)
    torch.cuda.synchronize()
    assert not wa[True].failed() and not wb[False].failed()
    for x, y in zip(a.outputs(), b.outputs(), strict=True):
        assert torch.equal(bits(x), bits(y))
