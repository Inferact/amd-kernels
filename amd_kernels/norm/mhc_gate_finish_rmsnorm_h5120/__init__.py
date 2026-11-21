# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""mHC seam finisher: reduces the producer GEMM's tile partials into the gates and the normed layer input."""

from .host import mhc_gate_finish

__all__ = ["mhc_gate_finish"]
