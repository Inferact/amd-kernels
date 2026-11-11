# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Stable local queues retain strict-tree ties, unbiased weight bits and graph writes."""
import struct

import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_topk as rt
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_topk_mxfp4_mxfp8_quant_sorted_tasks as new
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_topk_sorted as plain_new
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import mxfp4_quant_ref
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from .workload import make_activation, require_gfx950


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33, 64, 65, 128, 255, 256])
@pytest.mark.parametrize("case", ["random", "ties", "biased_ties", "saturation", "no_choices", "few_choices", "nan_logits"])
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("plain", [False, True])
def test_sorted_queues_match_router_topk(m, case, token_major, plain):
    """Against ``router_topk``'s ids and weights; the tasks kernel's codes and scales against the reference MXFP4
    quant and ``shared_quant``."""
    require_gfx950()
    s = 48 if m <= 64 else 16
    gen = torch.Generator(device="cuda").manual_seed(531 + m)
    p = (torch.randn((s, m, 128), device="cuda", generator=gen) * 0.1).permute(1, 0, 2)
    bias = torch.randn(128, device="cuda", generator=gen) * 0.01
    if case == "ties":
        p.copy_(p[:, :, :8].repeat(1, 1, 16))
        bias.zero_()
    if case == "biased_ties":
        # At 2**24 every sigmoid in [0,1] rounds to the same choice, while
        # the unbiased weights differ. This checks the queue's carried sigmoid.
        p.mul_(10)
        bias.fill_(2**24)
    if case == "saturation":
        p[:, :, ::2] = 2.0
        p[:, :, 1::2] = -2.0
        bias.zero_()
    if case == "no_choices":
        bias.fill_(float("-inf"))
    if case == "few_choices":
        bias.fill_(float("nan"))
        bias[127] = 0.0
    if case == "nan_logits":
        p[:, :, ::3] = float("nan")
        p[:, :, 1::3] = float("inf")
    if token_major:
        if s != 48 and not plain:
            pytest.skip("constant offsets serve S48 only")
        p = p.contiguous()
    x = make_activation(m, seed=m)
    outputs = [torch.empty((m, 4), device="cuda"),
               torch.empty((m, 4), dtype=torch.int32, device="cuda"),
               *[torch.empty((m, h), dtype=torch.uint8, device="cuda") for h in (3072, 192, 6144, 192)]]
    actual_module = plain_new if plain else new
    guards = None
    if plain:
        guards = [torch.full((m, 7), -777., device="cuda"),
                  torch.full((m, 7), -777, device="cuda", dtype=torch.int32)]
        outputs = [t[:, :4] for t in guards]
    scale = struct.unpack("<i", struct.pack("<f", 2.0))[0]

    def topk_args():
        return (p.data_ptr(), bias.data_ptr(), outputs[0].data_ptr(), outputs[1].data_ptr(), p.numel() * 4,
                m, p.stride(0), p.stride(1), outputs[0].stride(0), scale)

    def run():
        kernel = actual_module._get_kernel(128, 4, s, token_major)
        if plain:
            rt._run_compiled(kernel, *topk_args(), torch.cuda.current_stream())
        else:
            rt._run_compiled(kernel, *topk_args(), x.data_ptr(),
                             *[t.data_ptr() for t in outputs[2:]], torch.cuda.current_stream())

    def reference():
        rt._run_compiled(rt._get_kernel(128, 4, s), *topk_args(), torch.cuda.current_stream())
        out = [t.clone() for t in outputs[:2]]
        return out if plain else out + [*mxfp4_quant_ref(x), *shared_quant(x)]

    expected = reference()
    run()
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(outputs, expected))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for change in (False, True, False):
        if change:
            p.copy_(torch.randn(p.shape, device="cuda", generator=gen))
            x.copy_(make_activation(m, seed=m + 19))
            expected = reference()
        for t in outputs:
            t.view(torch.uint8).fill_(0xA5)
        graph.replay()
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(outputs, expected))

    if guards is not None:
        assert all(torch.all(t[:, 4:] == -777) for t in guards)
