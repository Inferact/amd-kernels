# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Fused sequential Markov draft sampling of a DSpark speculator: 5 steps, vocabulary 129280, rank 256, every row."""

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


from .host import MAX_ROWS, markov_chain, nrf_for, warmup_markov_chain
from .reference import NSPEC, R, V, markov_chain_reference

__all__ = ["MAX_ROWS", "NSPEC", "R", "V", "available", "markov_chain", "markov_chain_reference", "nrf_for",
           "warmup_markov_chain"]
