# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The draft's input projection: MXFP8 GEMM, N 5120 x K 15360 (three tapped hidden states -> hidden size)."""


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


from .host import DISPATCH, MainProjWeights, config_for, main_proj, warmup_main_proj
from .reference import K, N, main_proj_reference

__all__ = ["DISPATCH", "K", "MainProjWeights", "N", "available", "config_for", "main_proj", "main_proj_reference",
           "warmup_main_proj"]
