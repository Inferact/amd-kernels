# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Paired-lane cosine/sine loads use the current token's cache row on every graph replay."""

import pytest
import torch

from . import test_qkv_insert as checks
from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128 import host


@pytest.mark.parametrize("m", [193, 256, 288])
def test_changed_positions_and_cache_on_replay(m):
    checks.require_gfx950()
    _, (weight, scale), _, _ = checks.weights()
    codes, row_major, inputs = checks.make(m, 18200 + m)
    cache = torch.empty((256, 64), device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
    positions = torch.randint(0, 256, (m,), device="cuda", dtype=torch.int64)
    inputs = (*inputs[:4], cache, positions, *inputs[6:])
    checks.run("entry", m, codes, row_major, inputs)
    q, iq = checks.outputs(m)
    kv, index = inputs[8].clone(), inputs[9].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        host.mxfp8_qkv_norm_rope_nvfp4_insert(
            *codes, weight, scale, m, *inputs[:8], kv, index, *inputs[10:], q,
            iq.view(torch.float8_e4m3fn)[:, :128], **checks.KW,
        )
    for _ in range(6):
        # Unrelated cache rows and token positions distinguish both the paired lanes and the token halves.
        cache.uniform_(-1, 1)
        positions.copy_(torch.randint(0, 256, (m,), device="cuda", dtype=torch.int64))
        expected = checks.run("chain", m, codes, row_major, inputs)
        kv.copy_(inputs[8])
        index.copy_(inputs[9])
        q.fill_(7)
        iq[:, :128].fill_(0x77)
        graph.replay()
        torch.cuda.synchronize()
        assert checks.nbytes_differ((q, iq, kv, index), expected) == [0] * 4
