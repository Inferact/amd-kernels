# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Top-p mask of fp32 logits rows, vocab 200064, bit for bit as vLLM: its split top-p pipeline (<= 64 rows) in one
launch, and its one-program-per-row top-p kernel above."""


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


from .host import rows_grid, top_p_blocker, top_p_mask, warmup_top_p_mask
from .reference import SPLIT_MAX_BATCH, V, splits_for, top_p_mask_reference

__all__ = ["SPLIT_MAX_BATCH", "V", "available", "rows_grid", "splits_for", "top_p_blocker", "top_p_mask",
           "top_p_mask_reference", "warmup_top_p_mask"]
