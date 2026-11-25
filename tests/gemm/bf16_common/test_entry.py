# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The latent down entry gives the routed package's bits (codes, scales, bf16 projection) at one row, at one launch's
largest row count and past it (two launches); its reason is None on those calls."""
import importlib

import pytest
import torch

from amd_kernels.gemm.bf16_common import (bf16_latent_down_mxfp4_quant, bf16_latent_down_mxfp4_quant_blocker,
                                          bf16_latent_down_mxfp4_quant_shapes,
                                          bf16_latent_down_mxfp4_quant_weight_reason, latent_down)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("key", sorted(bf16_latent_down_mxfp4_quant_shapes()))
@pytest.mark.parametrize("with_bf16", [False, True])
def test_entry_equals_package(key, with_bf16):
    n, k = key
    name = latent_down._ROUTES.hosts[key]
    host = importlib.import_module(name)
    direct = getattr(host, name.split(".")[-2])
    g = torch.Generator(device="cuda").manual_seed(n + k)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).bfloat16()
    assert bf16_latent_down_mxfp4_quant_weight_reason(w) is None
    for m in (1, host.MAX_M, host.MAX_M + 13):
        x = torch.randn(m, k, device="cuda", generator=g).bfloat16()
        ob = [torch.empty(m, n, dtype=torch.bfloat16, device="cuda") if with_bf16 else None for _ in range(2)]
        assert bf16_latent_down_mxfp4_quant_blocker(x, w, out_bf16=ob[0]) is None
        ref = direct(x.clone(), w, shuffled=False, out_bf16=ob[1])
        got = bf16_latent_down_mxfp4_quant(x, w, shuffled=False, out_bf16=ob[0])
        torch.cuda.synchronize()
        for a, b in zip(got, ref):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), m
        if with_bf16:
            assert torch.equal(ob[0].view(torch.int16), ob[1].view(torch.int16)), m
