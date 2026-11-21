# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Kimi K3.1 DSA decode prologue (FlyDSL, gfx950): the q_a / kv_a norms with the fp8_ds_mla KV insert
(``qkv_a_norm_kv_insert``) and the indexer's key LayerNorm + MXFP4 insert, query MXFP4 quant and head weights
(``indexer_prep``). The contract is ``reference.py``."""
