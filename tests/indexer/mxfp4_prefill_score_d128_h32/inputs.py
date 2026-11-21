# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch

HEADS = 32


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("needs gfx950")


def make_inputs(rows: int, keys: int, windows=None, seed: int = 0, device="cuda"):
    """Random FP4 index tensors in the scorer's layouts; ``windows`` = [(ks, ke), ...] per row (default causal tail)."""
    g = torch.Generator(device=device).manual_seed(seed)
    pages = (keys + 63) // 64
    q_payload = torch.randint(0, 256, (rows, HEADS, 64), dtype=torch.uint8, device=device, generator=g)
    q_scale = torch.randint(118, 125, (rows, 1, 4, 16, 4), dtype=torch.uint8, device=device, generator=g)
    key_payload = torch.randint(0, 256, (pages, 1, 4, 64, 16), dtype=torch.uint8, device=device, generator=g)
    key_scale = torch.randint(118, 125, (pages, 1, 4, 64), dtype=torch.uint8, device=device, generator=g)
    weights = torch.randn(rows, HEADS, device=device, dtype=torch.bfloat16, generator=g)
    if windows is None:
        windows = [(0, keys - rows + r + 1) for r in range(rows)]
    ks = torch.tensor([w[0] for w in windows], dtype=torch.int32, device=device)
    ke = torch.tensor([w[1] for w in windows], dtype=torch.int32, device=device)
    return q_payload, q_scale, weights, key_payload, key_scale, ks, ke
