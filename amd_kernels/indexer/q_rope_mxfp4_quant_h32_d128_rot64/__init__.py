# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Indexer query RoPE (last 64 of 128 dims) + MXFP4 quant with the scaled head weights, FlyDSL (gfx950). The contract
is ``reference.py``."""

from .host import q_rope_mxfp4_quant, supports_q_rope_mxfp4_quant

__all__ = ["q_rope_mxfp4_quant", "supports_q_rope_mxfp4_quant"]
