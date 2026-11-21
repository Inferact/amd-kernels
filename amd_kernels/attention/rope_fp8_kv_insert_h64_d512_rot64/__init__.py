# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Query RoPE + e4m3 quant and KV RoPE + e4m3 cache insert, 64 heads of 512 (RoPE on the last 64 dims, GPT-J pairs),
FlyDSL (gfx950), in one launch. The contract is ``reference.py``."""

from .host import rope_fp8_kv_insert, supports_rope_fp8_kv_insert

__all__ = ["rope_fp8_kv_insert", "supports_rope_fp8_kv_insert"]
