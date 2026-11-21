# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA chunked prefill for 12 heads of dim 128 (FlyDSL, gfx950): causal conv + SiLU, gate and q / k norms, the chunked delta-rule scan
over fp32 recurrent states and the gated RMSNorm, in one launch for short batches and three (four for long chains) otherwise. The
contract is ``reference.py``."""

from .host import chunk_scan, gated_norm, plan_groups, prefill, uses_mega
from .prologue import chunk_table

__all__ = ["chunk_scan", "chunk_table", "gated_norm", "plan_groups", "prefill", "uses_mega"]
