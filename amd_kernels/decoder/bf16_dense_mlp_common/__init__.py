# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dense bf16 MLP shard entry (gate_up + SwiGLU, down on packed weights): reads (hidden, inter) from the down weight
and calls the shape package that serves it (imported on first use); a call no package covers gets a reason from
``bf16_dense_mlp_unsupported_reason``."""
import functools


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable."""
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


from .host import (bf16_dense_mlp, bf16_dense_mlp_shapes, bf16_dense_mlp_unsupported_reason,  # noqa: E402
                   dense_mlp_pack_down, dense_mlp_pack_gate_up, dense_mlp_packed_tiles, dense_mlp_rows_buffer,
                   dense_mlp_workspace, dense_mlp_workspace_unsupported_reason)

__all__ = ["available", "bf16_dense_mlp", "bf16_dense_mlp_shapes", "bf16_dense_mlp_unsupported_reason",
           "dense_mlp_pack_down", "dense_mlp_pack_gate_up", "dense_mlp_packed_tiles", "dense_mlp_rows_buffer",
           "dense_mlp_workspace", "dense_mlp_workspace_unsupported_reason"]
