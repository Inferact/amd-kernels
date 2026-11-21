# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 GEMM entries: each reads its shape from the operands and calls the shape package that serves it (imported on
first use); a call no package covers gets a reason from ``<op>_unsupported_reason``."""
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


from .latent_down import (bf16_latent_down_mxfp4_quant, bf16_latent_down_mxfp4_quant_shapes,  # noqa: E402
                          bf16_latent_down_mxfp4_quant_unsupported_reason, bf16_latent_down_mxfp4_quant_weight_reason,
                          latent_down_xcd_local)
from .linear_gather import (bf16_linear_gather, bf16_linear_gather_shapes,  # noqa: E402
                            bf16_linear_gather_unsupported_reason, linear_gather_comm,
                            linear_gather_comm_unsupported_reason, linear_gather_pack_aux,
                            linear_gather_pack_aux_unsupported_reason, linear_gather_pack_weight,
                            linear_gather_packed_tiles, linear_gather_rows_buffer)

__all__ = ["available", "bf16_latent_down_mxfp4_quant", "bf16_latent_down_mxfp4_quant_shapes",
           "bf16_latent_down_mxfp4_quant_unsupported_reason", "bf16_latent_down_mxfp4_quant_weight_reason",
           "bf16_linear_gather", "bf16_linear_gather_shapes", "bf16_linear_gather_unsupported_reason",
           "latent_down_xcd_local", "linear_gather_comm", "linear_gather_comm_unsupported_reason",
           "linear_gather_pack_aux", "linear_gather_pack_aux_unsupported_reason", "linear_gather_pack_weight",
           "linear_gather_packed_tiles", "linear_gather_rows_buffer"]
