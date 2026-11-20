# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Every shape folder (``gemm._SHAPE_HOSTS``) owns its rows: the dispatcher's config at each row's window edges is
the folder's ``select`` where it has a row (its own kernels' windows, else its ``ROWS``); ``tuned_shapes`` is the folders'
dtype keys; there is no common table."""

import importlib.util

import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G

SHAPES = sorted(G._SHAPE_HOSTS)


def _edges(rows):
    """Each row's window edges, over every (N, K, dtype) block of ``rows``."""
    ms = set()
    for block in rows.values():
        lo = 1
        for top, _ in block:
            ms.update({lo, top})
            lo = top + 1
    return sorted(m for m in ms if m <= 65536)


@pytest.mark.parametrize("n,k", SHAPES, ids=[f"n{n}_k{k}" for n, k in SHAPES])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_dispatch_is_the_folders_select(n, k, out_dtype):
    host = G._shape_host(n, k)
    assert host.ROWS and all((nn, kk) == (n, k) for nn, kk, _ in host.ROWS)
    for m in _edges(host.ROWS):
        for allow_decode in (True, False):
            want = host.select(m, out_dtype, allow_decode)  # None: no row of the folder (the generic tiles)
            if want is not None:
                assert G._select_config(m, n, k, out_dtype, allow_decode) == want, (m, allow_decode)


def test_tuned_shapes_are_the_folders_rows():
    for dtype, name in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        want = {s for s in SHAPES if (*s, name) in G._shape_host(*s).ROWS}
        assert G.tuned_shapes(dtype) == want


def test_no_common_table():
    assert importlib.util.find_spec("amd_kernels.gemm.mxfp8_common.mxfp8_gemm_tuned") is None
