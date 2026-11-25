# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The single variant compiles in compile-only mode (one GPU, no launch: the kernel needs the peers). Run with
``AKR_COMPILE_ONLY=1``."""
import pytest
import torch

from tests import compile_only

from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192 import host


@pytest.mark.parametrize("m", [1, 288, host.MAX_TOKENS])
def test_compiles(m):
    if not compile_only.ACTIVE:
        pytest.skip("compile-only mode (AKR_COMPILE_ONLY=1) only: the kernel needs 8 ranks")
    dev = torch.device("cuda")
    comm = host.DcpPackedCombine.__new__(host.DcpPackedCombine)
    comm.rank, comm.device, comm.max_tokens, comm.slot_bytes = 0, dev, host.MAX_TOKENS, 1 << 20
    comm.table = torch.zeros(16, dtype=torch.int64, device=dev)
    send = torch.zeros((8, m, 48, 66), dtype=torch.bfloat16, device=dev)
    out = torch.zeros((m, 16, 192), dtype=torch.bfloat16, device=dev)
    comm._launch(send, out, torch.cuda.current_stream())
