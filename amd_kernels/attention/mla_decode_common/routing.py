# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What every module of the entry shares: the device check and the reason texts."""

import functools
import importlib.util

import torch


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable (the check every routed package makes)."""
    if not torch.cuda.is_available() or importlib.util.find_spec("flydsl") is None:
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"


def no_kernel(op: str, names, key, table) -> str:
    dims = ", ".join(f"{n}={v}" for n, v in zip(names, key))
    return f"{op}: no kernel for {dims} (covered: {', '.join(str(k) for k in sorted(table))})"


def owner_key(table, cls):
    """The key whose shape package defines ``cls`` (an object an entry built), or None."""
    mod = cls.__module__
    for key, name in table.items():
        if mod == name or mod.startswith(name + "."):
            return key
    return None


def built_for(op: str, names, key, call) -> str:
    built = ", ".join(f"{n}={v}" for n, v in zip(names, key))
    return f"{op}: communicator built for {built}, call is {', '.join(f'{n}={v}' for n, v in zip(names, call))}"
