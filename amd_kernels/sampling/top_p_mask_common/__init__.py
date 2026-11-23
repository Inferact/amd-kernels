# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Top-p mask of fp32 logits rows for any vocabulary with a package: one entry, routed by the vocabulary (``_ROUTES``
of ``host.py``); an uncovered call gets a reason from ``top_p_mask_blocker``."""

from .host import (available, top_p_mask, top_p_mask_blocker, top_p_mask_config_blocker, top_p_mask_shapes,
                   top_p_mask_unsupported_reason, warmup_top_p_mask)

__all__ = ["available", "top_p_mask", "top_p_mask_blocker", "top_p_mask_config_blocker", "top_p_mask_shapes",
           "warmup_top_p_mask"]
