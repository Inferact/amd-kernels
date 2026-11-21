# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shape-agnostic op entries (one per op contract): a call's shape key -> the shape package that serves it.

The category modules (``amd_kernels.<category>``) hold the entries: ``<op>`` reads the key from its tensors and calls
the shape package, ``<op>_blocker`` answers on host metadata alone (None: covered, else the reason a caller logs before
it falls back to its own path), ``<op>_shapes`` lists the covered keys. Shape packages are imported on first use.
"""

import functools
import importlib


@functools.cache
def _import(name: str):
    return importlib.import_module(name)


class Routes:
    """One op's registry: shape key -> the shape package (dotted module name) that serves it."""

    def __init__(self, op: str, hosts: dict):
        self.op = op
        self.hosts = dict(hosts)

    def shapes(self) -> frozenset:
        return frozenset(self.hosts)

    def blocker(self, key) -> str | None:
        if key in self.hosts:
            return None
        return f"{self.op}: no shape package for {key} (covered: {sorted(self.hosts)})"

    def package(self, key):
        """The shape package of ``key``; ValueError with the blocker's reason when none serves it."""
        name = self.hosts.get(key)
        if name is None:
            raise ValueError(self.blocker(key))
        return _import(name)


def first(*reasons: str | None) -> str | None:
    """The first reason that is not None (blockers checked in order, cheapest first)."""
    return next((r for r in reasons if r is not None), None)


def check(reason: str | None) -> None:
    if reason is not None:
        raise ValueError(reason)
