# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The 16-row MFMA router tiers through ``router_gemm_direct`` (1-16 rows; 1-32 from the line-layout gate copy) and
``router_gemm_nblocks`` (1-384 rows): partials equal ``router_gemm.py``'s kernel bit for bit (strided and contiguous
x), graph replay equals eager, repeated runs equal, guard bytes kept."""
import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm as rg
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm_nblocks as rn
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm_direct as rd
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.layout import router_line_layout
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import router_gemm_partials_ref
from .workload import make_activation, require_gfx950

M = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 48, 52, 57, 58, 63, 64, 65, 97, 128, 129, 160, 192, 193,
     255, 256, 257, 320, 384]


@pytest.fixture(autouse=True)
def _gpu():
    require_gfx950()


def test_tiers():
    for m in range(1, 2048):
        cfg = rg._kernel_config(m)
        nb = rg.nblocks_config(m)
        if m <= 384:
            assert nb is not None and cfg[0] == 16 and cfg[4] == 0
            assert nb[:2] == cfg[1:3] and nb[3] == 1
            assert nb[2] == (2 if m <= rg.NBLOCKS_TWO_WAVES_MAX else cfg[3])
        else:
            assert nb is None


@pytest.mark.parametrize("m", M)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("kernel", ["nblocks", "nblocks_x_direct", "direct", "direct_lines"])
def test_partials_equal_and_graph(m, strided, kernel):
    if kernel == "direct" and m > 16:
        pytest.skip("direct kernel: one m-tile")
    if kernel == "direct_lines" and m > rg.DIRECT_LINES_MAX:
        pytest.skip("line-layout direct kernel: two m-tiles")
    if kernel == "nblocks_x_direct" and rg.X_DIRECT_FROM <= m <= rg.X_DIRECT_MAX:
        pytest.skip("served there: the nblocks case")
    torch.manual_seed(m)
    if strided:
        source = torch.empty((m, 6146), dtype=torch.bfloat16, device="cuda")
        x = source[:, :6144]
    else:
        x = torch.empty((m, 6144), dtype=torch.bfloat16, device="cuda")
    x.copy_(make_activation(m, seed=m))
    w = torch.randn((128, 6144), device="cuda") / 6144**0.5
    lines = router_line_layout(w)
    cfg = rg._kernel_config(m)
    split = cfg[1]
    base = rg._get_kernel(*cfg)
    original = torch.empty((split, m, 128), device="cuda")

    def launch_base():
        rg._run_compiled(base, x.data_ptr(), w.data_ptr(), original.data_ptr(), m, x.stride(0),
                         torch.cuda.current_stream())

    storage = torch.full((split * m * 128 + 64,), 17.0, device="cuda")
    p = storage[32:-32].view(split, m, 128)
    if kernel == "direct":
        k = rd._get_kernel(48, 64, 1)

        def launch():
            rd._run_compiled(k, x.data_ptr(), w.data_ptr(), p.data_ptr(), m, x.stride(0),
                             torch.cuda.current_stream())
    elif kernel == "nblocks_x_direct":  # every tier, not only the served rows
        k = rn._get_kernel(*rg.nblocks_config(m), 0, True)

        def launch():
            rn._run_compiled(k, x.data_ptr(), w.data_ptr(), p.data_ptr(), m, x.stride(0), 0, 0, 0, 0,
                             torch.cuda.current_stream())
    elif kernel == "direct_lines":
        k = rd._get_kernel(48, 64, (m + 15) // 16, 2, True)

        def launch():
            rd._run_compiled(k, x.data_ptr(), lines.data_ptr(), p.data_ptr(), m, x.stride(0),
                             torch.cuda.current_stream())
    else:
        # the served form (x through LDS, or from global memory at X_DIRECT_FROM..X_DIRECT_MAX)
        k = rn._get_kernel(*rg.nblocks_config(m), 0, rg.X_DIRECT_FROM <= m <= rg.X_DIRECT_MAX)

        def launch():
            rn._run_compiled(k, x.data_ptr(), w.data_ptr(), p.data_ptr(), m, x.stride(0), 0, 0, 0, 0,
                             torch.cuda.current_stream())

    launch_base()
    p.fill_(float("nan"))
    launch()
    assert torch.equal(p.view(torch.int32), original.view(torch.int32))
    # the public call takes this kernel
    pub = rg.router_gemm_partials(x, w)
    assert pub.shape == (m, split, 128)
    assert torch.equal(pub.permute(1, 0, 2).view(torch.int32), original.view(torch.int32))
    # with the line-layout copy (the entry's MoeWeights): the same partials at every tier
    pub = rg.router_gemm_partials(x, w, lines)
    assert torch.equal(pub.permute(1, 0, 2).view(torch.int32), original.view(torch.int32))
    exact, bound = router_gemm_partials_ref(x, w, split_k=split)
    assert ((p.double() - exact).abs() <= 2**-18 * bound).all()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launch()
    for seed in (901, 902, 901):
        x.copy_(make_activation(m, seed=seed))
        launch_base()
        p.fill_(float("nan"))
        g.replay()
        assert torch.equal(p.view(torch.int32), original.view(torch.int32))
        p.fill_(float("nan"))
        g.replay()
        assert torch.equal(p.view(torch.int32), original.view(torch.int32))
    torch.cuda.synchronize()
    assert (storage[:32] == 17).all() and (storage[-32:] == 17).all()


def test_direct_rejects_untested_splits():
    # only S48 is served: S16 / S24 gave wrong partials
    for split in (8, 16, 24):
        with pytest.raises(AssertionError):
            rd.compile_router_gemm_direct(split, 64, 1)
