# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``top_p_mask`` through the entry is the vocabulary package's call, bit for bit (masked rows and the > 64-row
kernel's ``buffer``), at the smallest and largest served row counts of each package, with no reason on those calls;
a too-small ``buffer`` and an operand off the current device get a reason (the real ``rows_grid``)."""

import importlib

import pytest
import torch

from amd_kernels.sampling.top_p_mask_common import available, top_p_mask, top_p_mask_blocker
from amd_kernels.sampling.top_p_mask_common.host import _ROUTES


def require_gfx950():
    if not available():
        pytest.skip("requires gfx950 with FlyDSL")


def lm_rows(n, vocab, seed):
    """bf16-valued logits with a heavy head, per-row p in 0.6..1 (some rows 1: untouched)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, vocab, generator=gen) * 2.0
    x.scatter_(1, torch.randint(0, vocab, (n, 64), generator=gen), torch.rand(n, 64, generator=gen) * 12 + 6)
    p = torch.rand(n, generator=gen) * 0.4 + 0.6
    p[1::7] = 1.0
    return x.bfloat16().float().cuda(), p.cuda()


def same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize("vocab,n,with_buffer", [(163840, 1, False), (163840, 64, False), (200064, 1, False),
                                                 (200064, 64, False), (200064, 65, False), (200064, 65, True),
                                                 (200064, 256, True)])
def test_entry_is_the_package(vocab, n, with_buffer):
    require_gfx950()
    pkg = importlib.import_module(_ROUTES.hosts[(vocab,)])
    x, p = lm_rows(n, vocab, n + vocab)
    buf = [torch.zeros(pkg.rows_grid(n, 0), vocab, device="cuda") for _ in range(2)] if with_buffer else [None, None]
    assert top_p_mask_blocker(x, p, buf[0]) is None
    a, b = x.clone(), x.clone()
    assert top_p_mask(a, p, buf[0]) is a
    if with_buffer:
        pkg.top_p_mask(b, p, buf[1])
    else:
        pkg.top_p_mask(b, p)
    torch.cuda.synchronize()
    assert same(a, b) and not same(a, x)
    if with_buffer:
        assert same(buf[0], buf[1])


def test_small_buffer_reason():
    require_gfx950()
    x, p = lm_rows(100, 200064, 3)
    grid = importlib.import_module(_ROUTES.hosts[(200064,)]).rows_grid(100, 0)
    why = top_p_mask_blocker(x, p, torch.empty(grid - 1, 200064, device="cuda"))
    assert why == f"top_p_mask: buffer torch.float32 ({grid - 1}, 200064) (fp32 [>= {grid}, 200064], contiguous)"


def test_off_device_reasons():
    require_gfx950()
    x, p = lm_rows(100, 200064, 4)
    assert top_p_mask_blocker(x, p.cpu()) == "top_p_mask: p on cpu (the current CUDA device)"
    assert top_p_mask_blocker(x.cpu(), p) == "top_p_mask: logits on cpu (the current CUDA device)"
    buf = torch.empty(256, 200064)
    assert top_p_mask_blocker(x, p, buf) == "top_p_mask: buffer on cpu (the current CUDA device)"
    assert top_p_mask_blocker(x[:8], p[:8], buf) is None
