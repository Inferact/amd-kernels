# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
from tests.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8.test_fused_ranks8 import worker
from tests.multi_gpu8 import run_on_8_gpus


def test_ep8_situv2():
    package = "fused_ep_mxfp4_h3584_i3072_e112_top16_ranks8"
    cases = [
        ("uniform", [1] * 8, True, False),
        ("skew", [32, 0, 8, 1, 16, 2, 3, 5], False, True),
        ("hot", [256, 0, 2, 5, 16, 1, 3, 8], True, True),
    ]
    run_on_8_gpus(worker, 112, cases, [[16] * 8], package, timeout=900)
