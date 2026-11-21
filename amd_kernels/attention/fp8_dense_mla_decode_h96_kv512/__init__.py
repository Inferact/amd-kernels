# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DSpark draft dense MLA decode over the fp8 latent cache (FlyDSL, gfx950)."""

__all__ = ["fp8_dense_mla_decode", "supports_fp8_dense_mla_decode"]


def __getattr__(name):
    if name in __all__:
        from . import host

        return getattr(host, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
