# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dense bf16 MLP shard entry (gate_up + SwiGLU, down on packed weights): reads (hidden, inter) from the down weight
and calls the shape package that serves it (``host._ROUTES``, imported on first use); a call no package covers gets a
reason from ``bf16_dense_mlp_blocker``."""
from amd_kernels.routes import available

from .host import (bf16_dense_mlp, bf16_dense_mlp_blocker, bf16_dense_mlp_shapes, bf16_dense_mlp_unsupported_reason,
                   dense_mlp_pack_down, dense_mlp_pack_gate_up, dense_mlp_packed_tiles, dense_mlp_rows_buffer,
                   dense_mlp_workspace, dense_mlp_workspace_blocker, dense_mlp_workspace_unsupported_reason,
                   prepare_dense_mlp_workspace)

__all__ = ["available", "bf16_dense_mlp", "bf16_dense_mlp_blocker", "bf16_dense_mlp_shapes", "dense_mlp_pack_down",
           "dense_mlp_pack_gate_up", "dense_mlp_packed_tiles", "dense_mlp_rows_buffer", "dense_mlp_workspace",
           "dense_mlp_workspace_blocker", "prepare_dense_mlp_workspace"]
