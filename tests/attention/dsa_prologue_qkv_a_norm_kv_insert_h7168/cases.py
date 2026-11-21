# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch

EPS = 1e-6
WEIGHT_SCALE = 32**-0.5 * 128**-0.5
# the served token counts: the critical M of the c4..c48 runs and CUDA-graph padding sizes
SERVED_M = [1, 6, 12, 24, 40, 65, 80, 128, 192]


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


def owned_slots(n, n_slots, pages, gen, owned=8):
    """Slots of the first ``n_slots`` tokens as one of ``owned`` DCP ranks sees them: 1 in ``owned`` tokens is its own."""
    perm = torch.randperm(pages * 64, generator=gen)[:n_slots]
    mine = torch.rand(n_slots, generator=gen) < 1.0 / owned
    return torch.where(mine, perm, torch.full_like(perm, -1)).cuda()
