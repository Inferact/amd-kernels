# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The device check and the reason texts every op of the family shares."""
import functools


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable."""
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


def dims(names, key) -> str:
    if len(names) != len(key):
        return f"shape {tuple(key)}"
    return ", ".join(f"{n}={v}" for n, v in zip(names, key))


def no_kernel(op, names, key, table) -> str:
    return f"{op}: no kernel for {dims(names, key)} (covered: {', '.join(str(k) for k in sorted(table))})"


def no_kernel_for_type(op, cls, table) -> str:
    return f"{op}: no kernel for a {cls.__qualname__} (covered: {', '.join(str(k) for k in sorted(table))})"


def built_for(op, names, key, call, what="communicator") -> str:
    return f"{op}: {what} built for {dims(names, key)}, call is {dims(names[:len(call)], call)}"


def key_of_type(table, cls):
    """The key whose module defines ``cls``, or None."""
    return next((k for k, name in table.items() if cls.__module__ == name), None)
