# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""reference.py against vLLM's unpack-combine on one GPU (skipped without vLLM)."""
import pytest
import torch

from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192.reference import combine_reference

from .common import has_vllm, make_sends, vllm_combine


@pytest.mark.skipif(not has_vllm(), reason="vLLM is not installed")
@pytest.mark.parametrize("t", [1, 17, 288])
def test_reference_matches_vllm(t):
    dev = torch.device("cuda", 0)
    sends = make_sends(t, 0, dev, empty_ranks=(5,))
    for rank in (0, 5):
        want = vllm_combine(sends, rank).float()
        got = combine_reference(sends, rank).float()
        assert not got.isnan().any()
        assert (got - want).abs().max().item() <= want.abs().max().item() * 2**-7
