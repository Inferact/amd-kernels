# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``unsupported_reason``: None for the checkpoint layout ``MoeWeights.from_row_major`` takes, a reason for a wrong
shape, dtype or SwiGLU constant (meta tensors: no GPU)."""

import torch
from amd_kernels.moe import mxfp4_h6144_i768_e128_top4 as P

H, I, E = P.H, P.I, P.E  # noqa: E741


def _weights(**override):
    m = dict(device="meta")
    w = dict(
        w13=torch.empty((E, 2 * I, H // 2), dtype=torch.uint8, **m),
        w13_scale=torch.empty((E, 2 * I, H // 32), dtype=torch.uint8, **m),
        w2=torch.empty((E, H, I // 2), dtype=torch.uint8, **m),
        w2_scale=torch.empty((E, H, I // 32), dtype=torch.uint8, **m),
        shared_w13=torch.empty((2 * I, H), dtype=torch.float8_e4m3fn, **m),
        shared_s13=torch.empty((2 * I, H // 32), dtype=torch.uint8, **m),
        shared_w2=torch.empty((H, I), dtype=torch.float8_e4m3fn, **m),
        shared_s2=torch.empty((H, I // 32), dtype=torch.uint8, **m),
    )
    w.update(override)
    return w


def test_checkpoint_layout_is_supported():
    assert P.unsupported_reason(**_weights()) is None


def test_reasons():
    assert "w2:" in P.unsupported_reason(**_weights(w2=torch.empty((E, H, I), dtype=torch.uint8, device="meta")))
    bf16 = torch.empty((2 * I, H), dtype=torch.bfloat16, device="meta")
    assert "shared w13:" in P.unsupported_reason(**_weights(shared_w13=bf16))
    assert "SwiGLU" in P.unsupported_reason(**_weights(), swiglu=(1.0, 7.0, 1.0))
    assert "w13:" in P.unsupported_reason(**_weights(w13=None))
