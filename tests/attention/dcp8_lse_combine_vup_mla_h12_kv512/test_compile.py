# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Every variant compiles: each of the 12 token tiers, plain and gated, in compile-only mode (one GPU, no launch: the
kernel needs the peers). Run with ``AKR_COMPILE_ONLY=1``."""
import pytest
import torch

from tests import compile_only

from amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512 import host


@pytest.mark.parametrize("tiles", range(1, 13))
def test_tier_compiles(tiles):
    if not compile_only.ACTIVE:
        pytest.skip("compile-only mode (AKR_COMPILE_ONLY=1) only: the kernel needs 8 ranks")
    dev = torch.device("cuda")
    comm = host.DcpVupCombine.__new__(host.DcpVupCombine)
    comm.rank, comm.device, comm.max_tokens, comm.base_e, comm.slot_bytes = 0, dev, 192, True, 1 << 20
    comm.table = torch.zeros(16, dtype=torch.int64, device=dev)
    m = 16 * tiles
    x = torch.zeros((m, 96, 512), dtype=torch.bfloat16, device=dev)
    lse = torch.zeros((m, 96), dtype=torch.float32, device=dev)
    wt = torch.zeros((96, 128, 512), dtype=torch.bfloat16, device=dev)
    out = torch.zeros((m, 12, 128), dtype=torch.bfloat16, device=dev)
    gate = torch.zeros((m, 1536), dtype=torch.bfloat16, device=dev)
    for g in (None, gate):
        comm._launch(x, lse, wt, out, None, None, torch.cuda.current_stream(), g)
