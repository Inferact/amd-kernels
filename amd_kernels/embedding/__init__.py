# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Embedding op entries. Callers import ops from here only (``amd_kernels/README.md``, "Op entries"); each family's
``__all__`` is re-exported."""

from amd_kernels.routes import available

from . import vocab_gather_common  # noqa: F401
from .vocab_gather_common import *  # noqa: F403

__all__ = sorted({"available", *vocab_gather_common.__all__})
