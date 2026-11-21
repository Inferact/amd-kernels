# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Fused Markov draft chain of a Kimi-K3.1 DSpark draft (vocab 163840, Markov rank 512)."""


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


from .host import MAX_ROWS, markov_gumbel_fused, row_tiles, scratch_dwords
from .reference import RANK, V

__all__ = ["MAX_ROWS", "RANK", "V", "available", "markov_gumbel_fused", "row_tiles", "scratch_dwords"]
