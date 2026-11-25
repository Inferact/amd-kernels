# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What every module of the entry shares: the device check and the communicator's "built for" reason."""

from amd_kernels.routes import available

__all__ = ["available", "built_for"]


def built_for(op: str, names, key, call) -> str:
    built = ", ".join(f"{n}={v}" for n, v in zip(names, key))
    return f"{op}: communicator built for {built}, call is {', '.join(f'{n}={v}' for n, v in zip(names, call))}"
