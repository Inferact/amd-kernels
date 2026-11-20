# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``allreduce_add_rmsnorm_ranks2.unsupported_reason``: the conditions a model integration asks the library about."""

from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm_ranks2 as AR


def test_reasons():
    assert "2 ranks" in AR.unsupported_reason(4)
    assert "2 ranks" in AR.unsupported_reason(1)
    assert "hidden size 6144" in AR.unsupported_reason(2, 4096)
    assert AR.unsupported_reason(2) in (None, "needs gfx950 with FlyDSL")
