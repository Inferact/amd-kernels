# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Token-wave geometry: the base config's partials bit for bit, the partial contract, guards, changed graph inputs."""
import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm as rg
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import router_gemm_partials_ref
from .workload import make_activation, require_gfx950


@pytest.mark.parametrize('m,waves', [(512,4), (513,5), (640,5), (641,6), (768,6),
    (769,8), (896,8), (897,8), (1024,8), (1025,5), (1280,5), (1281,6),
    (1536,6), (1537,7), (1792,7), (1793,8), (2047,8)])
def test_round_geometry(m, waves):
    active, helpers = rg._waves(32, m)
    assert active == waves
    if m >= 513:
        assert active + helpers == 8
        groups = (m + 32 * active - 1) // (32 * active)
        assert (groups + 3) // 4 == (m + 1023) // 1024
        if active > 1 and not 769 <= m <= 896:
            smaller = (m + 32 * (active - 1) - 1) // (32 * (active - 1))
            assert (smaller + 3) // 4 > (m + 1023) // 1024


@pytest.mark.parametrize('m', [193, 224, 256, 257, 320, 384, 385, 512, 513, 640, 641, 768, 769, 896, 897, 1024,
    1025, 1280, 1281, 1536, 1537, 1792, 1793, 2047])
def test_partial_contract_and_graph(m):
    require_gfx950()
    torch.manual_seed(456)
    # An even, noncontiguous row stride exercises the public router contract.
    source = torch.empty((m, 6146), dtype=torch.bfloat16, device='cuda')
    x = source[:, :6144]
    x.copy_(make_activation(m, seed=m))
    w = torch.randn((128, 6144), device='cuda') / 6144**.5
    size = 16 * m * 128
    storage = torch.full((size + 64,), 17., device='cuda')
    p = storage[32:-32].view(16, m, 128)
    f = rg._get_kernel(*rg._kernel_config(m))
    base = rg._get_kernel(32, 16, 64, 8, 0)
    original = torch.empty_like(p)
    def launch(exe, out):
        rg._run_compiled(exe, x.data_ptr(), w.data_ptr(), out.data_ptr(), m,
                         x.stride(0), torch.cuda.current_stream())
    launch(base, original)
    p.fill_(float('nan'))
    launch(f, p)
    assert torch.equal(p, original)
    exact, bound = router_gemm_partials_ref(x, w)
    assert ((p.double()-exact).abs() <= 2**-18 * bound).all()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launch(f, p)
    for seed in [901, 902, 901]:
        x.copy_(make_activation(m, seed=seed))
        launch(base, original)
        p.fill_(float('nan'))
        g.replay()
        assert torch.equal(p, original)
        # Repeat without changing inputs: deterministic, including output poisoning.
        p.fill_(float('nan'))
        g.replay()
        assert torch.equal(p, original)
    torch.cuda.synchronize()
    assert (storage[:32] == 17).all() and (storage[-32:] == 17).all()


@pytest.fixture(scope='module')
def spinner():
    require_gfx950()
    from tests.mfma_spinner import mfma_spinner
    return mfma_spinner(256)


@pytest.mark.parametrize('m', [193, 256, 384, 513, 641, 769, 1025, 1537])
def test_partials_next_to_mfma(spinner, m):
    x = make_activation(m, seed=m)
    w = torch.randn((128, 6144), device='cuda') / 6144**.5
    expected = rg.router_gemm_partials(x, w).clone()
    torch.cuda.synchronize()
    foreign = torch.cuda.Stream(priority=-1)
    foreign.wait_stream(torch.cuda.current_stream())
    spinner(2_000_000, foreign)
    results = [rg.router_gemm_partials(x, w) for _ in range(3)]
    torch.cuda.synchronize()
    assert all(torch.equal(p, expected) for p in results)
