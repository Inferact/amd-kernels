# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# compile-only mode (AKR_COMPILE_ONLY=1, tests/compile_only.py); a test's own `python -m tests.<...>` subprocess
# passes here too
import os

if os.environ.get("AKR_COMPILE_ONLY") == "1":
    from . import compile_only

    compile_only.install()

# Tests call an entry at single sizes: the first-call warm-up (amd_kernels.warm_up) is off here; the warm-up tests
# turn it on in their own processes.
from amd_kernels import warm_up as _warm_up

_warm_up.set_auto(False)
