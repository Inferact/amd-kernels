# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The device check and the reason texts every op of the family shares."""
from amd_kernels.routes import Routes, available

__all__ = ["available", "blocker", "built_for", "not_owned", "swap_stream_workspaces"]


def blocker(routes, key) -> str | None:
    """``routes.blocker(key)``, the key printed as is when an operand of another rank gives it another length."""
    return (routes if len(key) == len(routes.names) else Routes(routes.op, routes.hosts)).blocker(key)


def not_owned(routes, cls) -> str:
    """The reason when no shape package of ``routes`` built the object of type ``cls``."""
    return f"{routes.op}: no kernel for a {cls.__qualname__} (covered: {sorted(routes.hosts)})"


def built_for(routes, key, call, what="communicator") -> str:
    return f"{routes.op}: {what} built for {routes.describe(key)}, call is {routes.describe(call)}"


def swap_stream_keys(workspaces: dict, device_index: int, a: int, b: int) -> None:
    """Exchange the entries of streams ``a`` and ``b`` on a device in a dict keyed (device index, stream, ...)."""
    moved = {}
    for key in [k for k in workspaces if k[0] == device_index and k[1] in (a, b)]:
        moved[(key[0], b if key[1] == a else a, *key[2:])] = workspaces.pop(key)
    workspaces.update(moved)


def loaded_hosts(routes) -> list:
    """The shape-package hosts of ``routes`` imported so far (a package nobody called has no workspace)."""
    import sys

    return [sys.modules[name] for name in routes.hosts.values() if name in sys.modules]


def swap_stream_workspaces(device_index: int, stream_a: int, stream_b: int) -> None:
    """Exchange the router's and the experts' per-stream workspaces of two streams on a device (dict keys only:
    nothing is allocated, copied or freed). A graph captured on ``stream_b`` while it holds ``stream_a``'s workspaces
    bakes their addresses, so one set serves a stream's eager calls and the graphs it replays; the calls on one set
    must stay in one stream order (the kernels keep their epochs and tickets on the device)."""
    from . import experts, router

    for host in loaded_hosts(router._ROUTES) + loaded_hosts(experts._ROUTES):
        for name in ("_WORK", "_AUX_WORK"):
            if isinstance(getattr(host, name, None), dict):
                swap_stream_keys(getattr(host, name), device_index, stream_a, stream_b)
