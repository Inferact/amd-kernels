# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""This shape's kernel variants: the owner-read decode rows equal the common decode kernel bit for bit, the persistent
256x192 tile's DBO plain grid equals its persistent grid, and the LDS-ring tiles keep their bits next to resident
MFMA work."""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import compile_mxfp8_gemm_decode
from amd_kernels.gemm.mxfp8_n6144_k4096 import host as H

from ..mxfp8_common.test_gemm import make_inputs, require_gfx950

N, K = 6144, 4096


@pytest.mark.parametrize("m", [225, 256, 320, 384, 448, 513, 640, 641, 768])
@torch.inference_mode()
def test_owner_read_rows_bitwise(m, monkeypatch):
    require_gfx950()
    config = g._select_prequant_config(m, N, K, torch.bfloat16)
    assert H.decode_launcher(config, False) is not None
    x, _, _, (w, s) = make_inputs(m, N, K, seed=m)
    a, sa = g.mxfp8_e4m3_quantize_shuffled(x, config.tile_m)
    got = g.mxfp8_gemm_shuffled(a, sa, w, s, torch.bfloat16, m)
    plain = compile_mxfp8_gemm_decode(N=N, K=K, RT=config.tile_m // 16, NI=config.tile_n // 16, KW=2,
                                      OWNER_READ=False)
    monkeypatch.setattr(H, "decode_launcher", lambda config, a_bf16: plain)
    g._decode_launcher.cache_clear()
    try:
        expected = g.mxfp8_gemm_shuffled(a, sa, w, s, torch.bfloat16, m)
    finally:
        g._decode_launcher.cache_clear()
    assert torch.equal(got, expected)


@pytest.mark.parametrize("m", [3329, 4096, 5121, 6144])
@torch.inference_mode()
def test_persistent_256x192_dbo_grid_bitwise(m, monkeypatch):
    require_gfx950()
    assert g._select_config(m, N, K, torch.bfloat16).kernel == H._PERSISTENT_KERNEL
    x, _, _, (w, s) = make_inputs(m, N, K, seed=m)
    a, sa = g.mxfp8_e4m3_quantize_aiter(x)
    persistent = g.mxfp8_gemm(a, sa, w, s, torch.bfloat16, activation_scale_preshuffled=True)
    monkeypatch.setattr(g, "DBO", True)
    plain = g.mxfp8_gemm(a, sa, w, s, torch.bfloat16, activation_scale_preshuffled=True)
    assert torch.equal(persistent, plain)


@pytest.mark.parametrize("m", [769, 1024, 1281, 2048])
@torch.inference_mode()
def test_lds_ring_tiles_next_to_mfma(m):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    assert g._select_config(m, N, K, torch.bfloat16).kernel in (H._TILE_128_KERNEL, H._ONE_TILE_KERNEL)
    x, _, _, (w, s) = make_inputs(m, N, K, seed=m)
    a, sa = g.mxfp8_e4m3_quantize_aiter(x)
    run = lambda: g.mxfp8_gemm(a, sa, w, s, torch.bfloat16, activation_scale_preshuffled=True)  # noqa: E731
    expected = run()
    spin = mfma_spinner(32)
    side = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    spin(1_000_000, side)
    results = [run() for _ in range(16)]
    torch.cuda.synchronize()
    assert all(torch.equal(expected, got) for got in results)
