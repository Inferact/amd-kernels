# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Batched MXFP8 GEMM (B x 1024 x 4096, 32x32 weight blocks) with MXFP8 output, FlyDSL (gfx950). The torch
contract is ``reference.py``."""


def __getattr__(name):
    if name in __all__:
        from . import host

        return getattr(host, name)
    raise AttributeError(name)


__all__ = ["mxfp8_bmm_mxfp8_out"]
