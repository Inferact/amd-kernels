# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dense MXFP8 decoder-stack entries on a TP group: the block (communicators of the fc, embedding, norm tails and
context K / V chain) and its layers' weights read their shape from their arguments and route to the shape package
(``_ROUTES`` / ``_LAYERS`` of ``stack.py``); an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .stack import (
    dense_decoder,
    dense_decoder_blocker,
    dense_decoder_context_kv_blocker,
    dense_decoder_layer,
    dense_decoder_shapes,
    dense_decoder_stack_blocker,
    prepare_dense_decoder,
)

__all__ = [
    "available",
    "dense_decoder",
    "dense_decoder_blocker",
    "dense_decoder_context_kv_blocker",
    "dense_decoder_layer",
    "dense_decoder_shapes",
    "dense_decoder_stack_blocker",
    "prepare_dense_decoder",
]
