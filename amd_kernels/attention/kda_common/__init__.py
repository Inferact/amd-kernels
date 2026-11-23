# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA entries: one per op (decode, prefill, mono decode), each reads its shape from the tensors and routes to the
shape package (``_ROUTES`` of each module); an uncovered call gets a reason from ``*_blocker``."""

from .decode import (
    kda_decode_fb_gate_spec,
    kda_decode_fb_gate_spec_blocker,
    kda_decode_fb_gate_spec_shapes,
    kda_decode_fb_gate_spec_unsupported_reason,
)
from .device import available
from .mono import (
    kda_mono_decode,
    kda_mono_decode_blocker,
    kda_mono_decode_shapes,
    kda_mono_decode_unsupported_reason,
    kda_mono_workspace,
    kda_mono_workspace_blocker,
    kda_mono_workspace_unsupported_reason,
    prepare_kda_mono_workspace,
)
from .prefill import (
    kda_prefill,
    kda_prefill_blocker,
    kda_prefill_chunk_table,
    kda_prefill_shapes,
    kda_prefill_unsupported_reason,
)

__all__ = [
    "available",
    "kda_decode_fb_gate_spec",
    "kda_decode_fb_gate_spec_blocker",
    "kda_decode_fb_gate_spec_shapes",
    "kda_mono_decode",
    "kda_mono_decode_blocker",
    "kda_mono_decode_shapes",
    "kda_mono_workspace",
    "kda_mono_workspace_blocker",
    "kda_prefill",
    "kda_prefill_blocker",
    "kda_prefill_chunk_table",
    "kda_prefill_shapes",
    "prepare_kda_mono_workspace",
]
