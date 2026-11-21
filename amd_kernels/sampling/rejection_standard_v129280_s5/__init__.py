# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Standard speculative rejection sampling of 5 draft tokens over a 129280-token vocabulary, two launches."""


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


from .host import rejection_sample, warmup_rejection_sample
from .reference import NSPEC, ROWS, V, rejection_sample_reference

__all__ = ["NSPEC", "ROWS", "V", "available", "rejection_sample", "rejection_sample_reference",
           "warmup_rejection_sample"]
