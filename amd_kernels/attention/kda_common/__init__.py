# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA entries: one per op (decode, prefill, mono decode), each reads its shape from the tensors and routes to the
shape package (``_SHAPE_HOSTS`` of each module); an uncovered call gets a reason from ``*_unsupported_reason``."""

from .decode import kda_decode_fb_gate_spec, kda_decode_fb_gate_spec_shapes, kda_decode_fb_gate_spec_unsupported_reason
from .device import available
from .mono import (
    kda_mono_decode,
    kda_mono_decode_shapes,
    kda_mono_decode_unsupported_reason,
    kda_mono_workspace,
    kda_mono_workspace_unsupported_reason,
)
from .prefill import kda_prefill, kda_prefill_chunk_table, kda_prefill_shapes, kda_prefill_unsupported_reason

__all__ = [
    "available",
    "kda_decode_fb_gate_spec",
    "kda_decode_fb_gate_spec_shapes",
    "kda_decode_fb_gate_spec_unsupported_reason",
    "kda_mono_decode",
    "kda_mono_decode_shapes",
    "kda_mono_decode_unsupported_reason",
    "kda_mono_workspace",
    "kda_mono_workspace_unsupported_reason",
    "kda_prefill",
    "kda_prefill_chunk_table",
    "kda_prefill_shapes",
    "kda_prefill_unsupported_reason",
]
