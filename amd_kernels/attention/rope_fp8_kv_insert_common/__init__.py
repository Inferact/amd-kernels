# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""RoPE + fp8 KV insert entry (``host._ROUTES``): reads (heads, dim, rotary dim) from the tensors."""

from .host import rope_fp8_kv_insert, rope_fp8_kv_insert_blocker, rope_fp8_kv_insert_shapes

__all__ = ["rope_fp8_kv_insert", "rope_fp8_kv_insert_blocker", "rope_fp8_kv_insert_shapes"]
