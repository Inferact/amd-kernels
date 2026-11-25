# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Standard rejection sampling entry (``host._ROUTES``)."""

from .host import rejection_sample, rejection_sample_blocker, rejection_sample_shapes, warmup_rejection_sample

__all__ = ["rejection_sample", "rejection_sample_blocker", "rejection_sample_shapes", "warmup_rejection_sample"]
