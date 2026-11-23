# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 GEMM entries: each reads its shape from the operands and calls the shape package that serves it (``_ROUTES``
of each module, imported on first use); a call no package covers gets a reason from ``<op>_blocker``."""
from amd_kernels.routes import available

from .latent_down import (bf16_latent_down_mxfp4_quant, bf16_latent_down_mxfp4_quant_blocker,
                          bf16_latent_down_mxfp4_quant_shapes, bf16_latent_down_mxfp4_quant_unsupported_reason,
                          bf16_latent_down_mxfp4_quant_weight_reason, latent_down_swap_stream_workspaces,
                          latent_down_xcd_local)
from .linear_gather import (bf16_linear_gather, bf16_linear_gather_blocker, bf16_linear_gather_shapes,
                            bf16_linear_gather_unsupported_reason, linear_gather_comm, linear_gather_comm_blocker,
                            linear_gather_comm_unsupported_reason, linear_gather_pack_aux,
                            linear_gather_pack_aux_blocker, linear_gather_pack_aux_unsupported_reason,
                            linear_gather_pack_weight, linear_gather_packed_tiles, linear_gather_rows_buffer,
                            prepare_linear_gather_comm)

__all__ = ["available", "bf16_latent_down_mxfp4_quant", "bf16_latent_down_mxfp4_quant_blocker",
           "bf16_latent_down_mxfp4_quant_shapes", "bf16_latent_down_mxfp4_quant_weight_reason", "bf16_linear_gather",
           "bf16_linear_gather_blocker", "bf16_linear_gather_shapes", "latent_down_swap_stream_workspaces",
           "latent_down_xcd_local", "linear_gather_comm", "linear_gather_comm_blocker", "linear_gather_pack_aux",
           "linear_gather_pack_aux_blocker", "linear_gather_pack_weight", "linear_gather_packed_tiles",
           "linear_gather_rows_buffer", "prepare_linear_gather_comm"]
