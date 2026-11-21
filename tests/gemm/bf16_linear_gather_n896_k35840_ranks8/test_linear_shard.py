# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The shard GEMM and the activation packing against reference.py at the served row counts and tier edges."""
import pytest
import torch
from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8 import host
from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8.reference import K, N_SHARD, linear_reference

ROWS = (1, 12, 16, 17, 32, 40, 48, 49, 65, 80, 81, 128, 129, 192)


@pytest.fixture(scope="module")
def weight():
    g = torch.Generator().manual_seed(7)
    w = (torch.randn(N_SHARD, K, generator=g) * 0.02).bfloat16().cuda()
    return w, host.pack_weight(w)


def aux_rows(rows, seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(rows + 3, K // 5, generator=g) * 0.4).bfloat16().cuda() for _ in range(5)]


@pytest.mark.parametrize("rows", ROWS)
def test_pack_aux(rows):
    aux = aux_rows(rows, rows)
    out = host.packed_rows_buffer(rows, "cuda")
    host.pack_aux([h[:rows] for h in aux], out, rows)
    ref = host.pack_rows(torch.cat([h[:rows] for h in aux], dim=-1), tiles=out.shape[0])
    full_tiles = rows // 16
    assert torch.equal(out[:full_tiles].view(torch.int16), ref[:full_tiles].view(torch.int16))
    if rows % 16:
        r = rows % 16
        a = out[full_tiles].view(K // 32, 4, 16, 8)[:, :, :r]
        b = ref[full_tiles].view(K // 32, 4, 16, 8)[:, :, :r]
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("rows", ROWS)
def test_linear_shard(weight, rows):
    w, wp = weight
    x = torch.cat(aux_rows(rows, 100 + rows), dim=-1)[:rows].contiguous()
    xp = host.pack_rows(x, tiles=host.packed_tiles(rows))
    cfg = host.config_for(rows)
    slab = torch.empty(cfg[2] * rows * N_SHARD, dtype=torch.float32, device="cuda")
    host.linear_shard(xp, wp, slab, rows)
    torch.cuda.synchronize()
    out = slab.view(cfg[2], rows, N_SHARD).sum(0).bfloat16()
    ref = linear_reference(x, w)
    cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=1).min().item()
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert cos >= 0.99999 and rel <= 1e-3, (rows, cos, rel)
    slab2 = torch.empty_like(slab)
    host.linear_shard(xp, wp, slab2, rows)
    assert torch.equal(slab.view(torch.int32), slab2.view(torch.int32))
