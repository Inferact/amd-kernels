# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MLA decode entry (``host._ROUTES``): reads (heads, dim) from ``q`` and calls the shape package."""

from .host import sparse_mla_decode, sparse_mla_decode_blocker, sparse_mla_decode_shapes

__all__ = ["sparse_mla_decode", "sparse_mla_decode_blocker", "sparse_mla_decode_shapes"]
