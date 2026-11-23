# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compressor tail (pool, norm, RoPE, e4m3 main-cache insert, ring save) and indexer key store (wk, norm, RoPE, MXFP4
insert) of a compress-ratio 1 or 2 layer, FlyDSL (gfx950), in one launch. The contract is ``reference.py``."""

from .host import compressor_insert, shuffle_wk, supports_compressor_insert

__all__ = ["compressor_insert", "shuffle_wk", "supports_compressor_insert"]
