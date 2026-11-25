# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sliding-window decode attention entries over e4m3 K / V pages: the op reads its shape from q and the cache and
routes to the shape package (``_ROUTES`` of ``decode.py``); an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .decode import (
    swa_decode_layer_blocker,
    swa_decode_max_query_len,
    swa_decode_oproj_input,
    swa_decode_oproj_input_blocker,
    swa_decode_oproj_input_shapes,
)

__all__ = [
    "available",
    "swa_decode_layer_blocker",
    "swa_decode_max_query_len",
    "swa_decode_oproj_input",
    "swa_decode_oproj_input_blocker",
    "swa_decode_oproj_input_shapes",
]
