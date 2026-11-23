# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Decoder-block entries: the parts of a decoder layer stack around its attention, sequenced on the kernel entries.
Callers import ops from here only (``amd_kernels/README.md``, "Op entries"); each family's ``__all__`` is re-
exported."""

from amd_kernels.routes import available

from . import bf16_dense_mlp_common, dense_mxfp8_decoder_common  # noqa: F401
from .bf16_dense_mlp_common import *  # noqa: F403
from .dense_mxfp8_decoder_common import *  # noqa: F403

__all__ = sorted({"available", *bf16_dense_mlp_common.__all__, *dense_mxfp8_decoder_common.__all__})
