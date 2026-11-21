# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Helpers of the entry tests: cloned operands and a byte-for-byte comparison of every tensor a call may write."""

import pytest
import torch


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


def clone(x):
    """``x`` with every tensor (also inside tuples, lists, dicts) copied, strides kept."""
    if isinstance(x, torch.Tensor):
        out = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
        out.copy_(x)
        return out
    if isinstance(x, (tuple, list)):
        return type(x)(clone(v) for v in x)
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    return x


def tensors(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for v in x for t in tensors(v)]
    if isinstance(x, dict):
        return [t for v in x.values() for t in tensors(v)]
    return []


def same_bytes(a, b) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                              b.contiguous().reshape(-1).view(torch.uint8))


def assert_same(got, want, tag):
    """Every tensor of ``got`` equals its counterpart in ``want`` byte for byte."""
    g, w = tensors(got), tensors(want)
    assert len(g) == len(w), tag
    bad = [i for i, (a, b) in enumerate(zip(g, w)) if not same_bytes(a, b)]
    assert not bad, (tag, "differs at tensor", bad)
