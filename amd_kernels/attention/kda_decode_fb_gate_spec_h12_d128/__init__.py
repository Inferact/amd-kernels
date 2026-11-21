# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA spec-verify decode for 12 heads of dim 128 (FlyDSL, gfx950): the f_b gate GEMM, causal conv + SiLU, delta-rule
recurrence with per-token state snapshots and the gated RMSNorm in one launch. The contract is ``reference.py``."""

from .host import blocker, kda_decode_fb_gate_spec

__all__ = ["blocker", "kda_decode_fb_gate_spec"]
