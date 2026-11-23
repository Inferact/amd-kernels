# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Op entries: the one way callers reach a kernel (``amd_kernels/README.md``, "Op entries").

Callers import ops from ``amd_kernels.<category>`` only. An op's implementation lives in a ``<family>_common``
package: ``<op>`` reads its shape key from its tensors and calls the shape package that ``Routes`` names for it,
``<op>_blocker`` takes the same arguments and answers on host metadata alone (None: served, else the reason a caller
logs before it falls back to its own path), ``<op>_shapes`` lists the covered keys. An op that binds weights or builds
state at load time has ``prepare_<op>``: a ``Prepared`` handle or the reason it cannot serve. Shape packages are
imported on first use.
"""

import functools
import importlib
import importlib.util
import warnings


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable (what every shape package needs)."""
    import torch

    if not torch.cuda.is_available() or importlib.util.find_spec("flydsl") is None:
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"


@functools.cache
def _import(name: str):
    return importlib.import_module(name)


class Routes:
    """One op's registry: shape key -> the shape package (dotted module name) that serves it.

    ``names`` labels the key's fields in the reason text; without it the key is printed as is.
    """

    def __init__(self, op: str, hosts: dict, names: tuple[str, ...] | None = None):
        self.op = op
        self.hosts = dict(hosts)
        self.names = names

    def shapes(self) -> frozenset:
        return frozenset(self.hosts)

    def describe(self, key) -> str:
        if self.names is None:
            return str(key)
        return ", ".join(f"{n}={v}" for n, v in zip(self.names, key))

    def blocker(self, key) -> str | None:
        if key in self.hosts:
            return None
        return f"{self.op}: no kernel for {self.describe(key)} (covered: {sorted(self.hosts)})"

    def host(self, key):
        """The shape package of ``key``, or None."""
        name = self.hosts.get(key)
        return None if name is None else _import(name)

    def package(self, key):
        """The shape package of ``key``; ValueError with the blocker's reason when none serves it."""
        name = self.hosts.get(key)
        if name is None:
            raise ValueError(self.blocker(key))
        return _import(name)

    def module(self, key, name: str):
        """Module ``name`` inside the shape package of ``key``."""
        self.package(key)
        return _import(f"{self.hosts[key]}.{name}")

    def owner(self, cls) -> object | None:
        """The key whose shape package defines ``cls`` (the type of an object an entry built), or None."""
        mod = cls.__module__
        for key, name in self.hosts.items():
            if mod == name or mod.startswith(name + "."):
                return key
        return None


class Rows:
    """The row counts at which a kernel beat the path a caller serves otherwise (``spans``: inclusive ranges)."""

    def __init__(self, *spans: tuple[int, int]):
        self.spans = spans

    def __contains__(self, m: int) -> bool:
        return any(lo <= m <= hi for lo, hi in self.spans)

    def blocker(self, op: str, m: int) -> str | None:
        if m in self:
            return None
        wins = ", ".join(f"{lo}-{hi}" for lo, hi in self.spans)
        return f"{op}: at {m} rows the served path is faster (this kernel wins at {wins})"


class Prepared:
    """What ``prepare_<op>`` returns: the op bound to its load-time state (weights in the kernel's layout, workspaces,
    communicators), compiled. ``blocker`` takes the call's arguments, as ``<op>_blocker`` does."""

    op = ""

    def blocker(self, *args, **kwargs) -> str | None:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def try_call(self, *args, **kwargs):
        """The op's result, or None (nothing launched) when ``blocker`` declines the call."""
        if self.blocker(*args, **kwargs) is not None:
            return None
        return self(*args, **kwargs)


def first(*reasons: str | None) -> str | None:
    """The first reason that is not None (blockers checked in order, cheapest first)."""
    return next((r for r in reasons if r is not None), None)


def check(reason: str | None) -> None:
    if reason is not None:
        raise ValueError(reason)


def prepare(reason: str | None, build):
    """``reason`` when it is set, else ``build()``: the body of a ``prepare_<op>`` over a blocker and a constructor."""
    return reason if reason is not None else build()


def renamed(fn, old: str):
    """``fn`` under its previous name ``old``, warning once per process that callers should switch."""

    @functools.wraps(fn)
    def alias(*args, **kwargs):
        _warn_once(f"amd_kernels: {old} is renamed {fn.__name__}")
        return fn(*args, **kwargs)

    alias.__name__ = alias.__qualname__ = old
    alias.renamed_to = fn.__name__
    return alias


def integer_form(legacy):
    """Let a blocker that now takes its op's arguments also answer a call in its previous integer form (the first
    argument an int), through ``legacy``, warning once per process."""

    def wrap(fn):
        @functools.wraps(fn)
        def blocker(*args, **kwargs):
            if args and type(args[0]) is int:
                _warn_once(f"amd_kernels: {fn.__name__} takes its op's arguments; the integer form is deprecated")
                return legacy(*args, **kwargs)
            return fn(*args, **kwargs)

        blocker.integer_form = legacy
        return blocker

    return wrap


@functools.cache
def _warn_once(message: str) -> None:
    warnings.warn(message, DeprecationWarning, stacklevel=3)
