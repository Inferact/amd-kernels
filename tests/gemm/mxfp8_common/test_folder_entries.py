# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The shape folders with entries of their own serve the common entries: ``mxfp8_linear`` (bf16) and ``mxfp8_gemm``
(row-major codes) return what the folder's entry returns, bit for bit, where the folder's table has a row, and the
common dispatcher's result elsewhere."""

import importlib

import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G

from .test_gemm import require_gfx950

BF16 = torch.bfloat16
FOLDERS = ["mxfp8_n1792_k5120", "mxfp8_n4096_k1280", "mxfp8_n5120_k2304"]


def _weights(pkg):
    gen = torch.Generator(device="cuda").manual_seed(pkg.N + pkg.K)
    q, s = pkg.quantize_mxfp8((torch.randn(pkg.N, pkg.K, generator=gen, device="cuda") * 0.05).to(BF16))
    return pkg.preshuffle_weight(q, s)


@pytest.mark.parametrize("name", FOLDERS)
@pytest.mark.parametrize("m", [1, 16, 32, 100, 768, 1000])
def test_mxfp8_linear_runs_the_folder_entry(name, m):
    require_gfx950()
    pkg = importlib.import_module(f"amd_kernels.gemm.{name}")
    w, ws = _weights(pkg)
    x = torch.randn(m, pkg.K, device="cuda").to(BF16)
    got = G.mxfp8_linear(x, w, ws, BF16)
    if pkg.kernel.row(pkg.kernel.LINEAR, m) is not None:
        torch.testing.assert_close(got, pkg.mxfp8_linear(x, w, ws), rtol=0, atol=0)
    else:
        torch.testing.assert_close(got, G.mxfp8_linear(x, w, ws, BF16, config=G._select_config(m, pkg.N, pkg.K)),
                                   rtol=0, atol=0)


@pytest.mark.parametrize("m", [16, 32, 64, 128])
def test_mxfp8_gemm_codes_run_the_folder_entry(m):
    require_gfx950()
    for name in ("mxfp8_n4096_k1280", "mxfp8_n5120_k2304"):
        pkg = importlib.import_module(f"amd_kernels.gemm.{name}")
        w, ws = _weights(pkg)
        codes, scale = pkg.quantize_mxfp8(torch.randn(m, pkg.K, device="cuda").to(BF16))
        got = G.mxfp8_gemm(codes, scale, w, ws, BF16)
        if name == "mxfp8_n4096_k1280":
            want = pkg.mxfp8_linear_prequant(codes, scale, w, ws)
        else:
            row = pkg.host._codes_row(m)
            if row is None:
                continue
            want = pkg.host._decode(codes, scale, w, ws, row[3], key=pkg.host.KEY)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
