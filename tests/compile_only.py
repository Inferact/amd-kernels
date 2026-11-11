# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile-only mode (``AKR_COMPILE_ONLY=1``; ``tests/__init__.py`` calls ``install`` in pytest and in a test's own
``python -m tests.<...>`` subprocess): every FlyDSL launch compiles, as in a normal run, and none runs.

A compile test (root README rule 4: every variant compiles before serving) checks which kernels compile, not what
they compute. In this mode it still runs on a GPU, so torch, aiter, Triton and what a host asks the device (CU and
XCD counts) are real and pick the variants a normal run picks; only FlyDSL's launches are skipped (FlyDSL's
``COMPILE_ONLY=1``). Several such tests can then share one card (CI: ``compile`` in ``tools/ci/targets.tsv``). The
FlyDSL kernels' outputs are left unwritten: a host that read one back to choose a variant would choose from garbage.

FlyDSL's ``flyc.compile`` returns None in this mode, so a caller that keeps what it returns (``_run_compiled``
of ``amd_kernels.common.tensor_shim``) would compile again at every call; ``install`` makes it return a launcher that does nothing, so a
variant compiles once, as in a normal run.
"""

import os

ACTIVE = os.environ.get("AKR_COMPILE_ONLY") == "1"

_installed = False


class _Launcher:
    """What ``flyc.compile`` returns in this mode: a compiled launch that runs nothing."""

    def __call__(self, *args, **kwargs):
        return None


class _Compile:
    """``flyc.compile`` (also ``flyc.compile[hints]``) that returns ``_Launcher`` where FlyDSL returns None."""

    def __init__(self, real):
        self._real = real

    def __getitem__(self, hints):
        return _Compile(self._real[hints])

    def __call__(self, func, *args):
        out = self._real(func, *args)
        return _Launcher() if out is None and args else out


def install():
    """Compile-only mode for this process (and the subprocesses it starts)."""
    global _installed
    if _installed:
        return
    os.environ["COMPILE_ONLY"] = "1"  # FlyDSL: compile, write the cache, launch nothing
    import flydsl.compiler as flyc

    flyc.compile = _Compile(flyc.compile)
    _installed = True
