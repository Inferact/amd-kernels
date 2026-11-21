# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MLP against reference.py at the served row counts and the tier edges: eager, graph, repeats."""
import pytest
import torch
from amd_kernels.decoder.bf16_dense_mlp_h7168_i1792 import MlpWorkspace, dense_mlp, pack_down, pack_gate_up, pack_rows
from amd_kernels.decoder.bf16_dense_mlp_h7168_i1792 import host
from amd_kernels.decoder.bf16_dense_mlp_h7168_i1792.pack import unpack_rows
from amd_kernels.decoder.bf16_dense_mlp_h7168_i1792.reference import HIDDEN, INTER, mlp_reference

ROWS = (1, 12, 16, 17, 32, 33, 40, 48, 49, 52, 60, 64, 65, 80, 81, 96, 97, 128, 129, 144, 145, 192)


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator().manual_seed(11)
    gu = (torch.randn(2 * INTER, HIDDEN, generator=g) * 0.02).bfloat16().cuda()
    dn = (torch.randn(HIDDEN, INTER, generator=g) * 0.03).bfloat16().cuda()
    return gu, dn, pack_gate_up(gu), pack_down(dn), MlpWorkspace(torch.device("cuda"))


def check(out, ref, tag):
    cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=1).min().item()
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert cos >= 0.99999 and rel <= 1e-3, (tag, cos, rel)


@pytest.mark.parametrize("rows", ROWS)
def test_mlp(weights, rows):
    gu, dn, gup, dnp, ws = weights
    g = torch.Generator().manual_seed(rows)
    x = (torch.randn(rows, HIDDEN, generator=g) * 0.5).bfloat16().cuda()
    xp = pack_rows(x, host.packed_tiles(rows))
    act_ref, out_ref = mlp_reference(x, gu, dn)
    out = dense_mlp(ws, xp, gup, dnp, rows)
    torch.cuda.synchronize()
    check(unpack_rows(ws.act, rows), act_ref, ("act", rows))
    check(out, out_ref, ("out", rows))
    first = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = dense_mlp(ws, xp, gup, dnp, rows)
    for _ in range(3):
        gout.view(torch.int16).fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(gout.view(torch.int16), first.view(torch.int16)), rows
