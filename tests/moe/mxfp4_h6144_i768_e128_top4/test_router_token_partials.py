# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Token-major S48 stores preserve logical partials, bounds, strides and replay bytes."""
import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm as rg
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm_direct as rd
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm_nblocks as rn
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_gemm_nblocks_lines as rl
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.layout import router_line_layout
from .workload import require_gfx950


@pytest.mark.parametrize("m", [1, 2, 3, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 47, 48, 57, 58, 63, 64, 65, 128, 256])
@pytest.mark.parametrize("lines", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_token_partials_match_split_major(m, lines, strided):
    require_gfx950()
    gen = torch.Generator(device="cuda").manual_seed(771 + m)
    backing = torch.randn((m, 6152 if strided else 6144), device="cuda", dtype=torch.bfloat16, generator=gen)
    x = backing[:, :6144]
    gate = torch.randn((128, 6144), device="cuda", generator=gen)
    wl = router_line_layout(gate) if lines else None
    split, bk, waves, nb = rg.nblocks_config(m)
    guard = 64
    buffers = [torch.full((split * m * 128 + 2 * guard,), -777., device="cuda") for _ in range(2)]
    parts = [v[guard:-guard].view(split, m, 128) for v in buffers]

    def run(index):
        # token-major stores: the standalone router's through 64, the quant-in-router entry's (S16) also at 65-128
        token_major = index == 1 and m <= 128
        args = (x.data_ptr(), (wl if lines and m <= 32 else gate).data_ptr(), parts[index].data_ptr(), m, x.stride(0))
        if m <= 16:
            kernel = rd._get_kernel(48, 64, 1, 2 if lines else 0, lines, token_major)
            rg._run_compiled(kernel, *args, torch.cuda.current_stream())
        elif lines and m <= 32:
            rg._run_compiled(rl._get_kernel(token_major), *args, torch.cuda.current_stream())
        else:
            kernel = rn._get_kernel(split, bk, waves, nb, 0, rg.X_DIRECT_FROM <= m <= rg.X_DIRECT_MAX, token_major)
            rg._run_compiled(kernel, *args, 0, 0, 0, 0, torch.cuda.current_stream())
        return parts[index].view(m, split, 128) if token_major else parts[index].permute(1, 0, 2)

    def equal(a, b):
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    expected = run(0).clone()
    actual = run(1)
    public = rg.router_gemm_partials(x, gate, wl)
    assert equal(expected, actual) and equal(expected, public)
    assert public.stride() == ((split * 128, 128, 1) if m <= 64 else (128, m * 128, 1))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(1)
    for change in (False, True, False):
        if change:
            x.copy_(torch.randn(x.shape, device="cuda", dtype=torch.bfloat16, generator=gen))
            expected = run(0).clone()
        parts[1].fill_(float("nan"))
        graph.replay()
        assert equal(expected, actual)
        for buf in buffers:
            assert torch.all(buf[:guard] == -777) and torch.all(buf[-guard:] == -777)
