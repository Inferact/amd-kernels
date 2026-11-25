# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The op-entry convention (``amd_kernels/README.md``, "Op entries") holds for every category: callers import from
``amd_kernels.<category>`` only, every exported op lives in a ``*_common`` family, a blocker takes its op's
arguments, deprecated names stay out of ``__all__``, and importing the categories imports no shape package. Host
only: no GPU."""

import importlib
import inspect
import pathlib
import subprocess
import sys

import pytest

import amd_kernels
from amd_kernels.routes import Prepared, Routes

ROOT = pathlib.Path(amd_kernels.__file__).parent
CATEGORIES = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and any(p.glob("*_common/__init__.py")))
# load-time checks named for what they check rather than for an op
LOAD_TIME = ("_rope_blocker", "_layer_blocker", "_weight_blocker", "_config_blocker", "_world_blocker",
             "_stack_blocker", "_context_kv_blocker")
# the object an op runs through, built by ``prepare_*`` (its blocker takes it first)
OBJECTS = {"comm", "ws", "gather", "sampling", "runner", "block", "handoffs", "layer"}


def category(name):
    return importlib.import_module(f"amd_kernels.{name}")


def params(fn):
    return list(inspect.signature(fn).parameters.values())


@pytest.mark.parametrize("name", CATEGORIES)
def test_exports_resolve_once_from_families(name):
    mod = category(name)
    assert len(mod.__all__) == len(set(mod.__all__))
    for export in mod.__all__:
        obj = getattr(mod, export)
        owner = getattr(obj, "__module__", None)
        if export == "available" or owner is None or not callable(obj):
            continue
        assert owner.startswith(f"amd_kernels.{name}.") and "_common" in owner, (export, owner)
        assert not hasattr(obj, "renamed_to"), f"deprecated alias {export} in __all__"


@pytest.mark.parametrize("name", CATEGORIES)
def test_blockers_take_their_op_arguments(name):
    mod = category(name)
    for export in mod.__all__:
        if not export.endswith("_blocker"):
            continue
        blocker = getattr(mod, export)
        op = getattr(mod, export.removesuffix("_blocker"), None)
        if op is None or not callable(op):
            first = params(blocker)[0].name
            assert export.endswith(LOAD_TIME) or first in OBJECTS, (export, first)
            continue
        want = params(op.__init__)[1:] if inspect.isclass(op) else params(op)
        got = params(blocker)
        positional = [p.name for p in want if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        assert [p.name for p in got][: len(positional)] == positional, (export, [p.name for p in got], positional)
        extra = {p.name for p in got} - {p.name for p in want}
        assert all(p.kind == p.KEYWORD_ONLY for p in got if p.name in extra), (export, extra)


@pytest.mark.parametrize("name", CATEGORIES)
def test_prepare_returns_a_handle_or_a_reason(name):
    mod = category(name)
    for export in mod.__all__:
        if export.startswith("prepare_"):
            hint = inspect.signature(getattr(mod, export)).return_annotation
            assert hint is inspect.Signature.empty or "str" in str(hint), export


def test_prepared_handles_try_call():
    class Half(Prepared):
        op = "half"

        def blocker(self, x):
            return None if x % 2 == 0 else "half: odd"

        def __call__(self, x):
            return x // 2

    assert Half().try_call(4) == 2 and Half().try_call(3) is None


def test_no_shape_table_outside_routes():
    left = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py") if "_SHAPE_HOSTS" in p.read_text()]
    assert not left, left


def test_every_routes_table_names_an_existing_package():
    seen = 0
    for path in ROOT.glob("*/*_common/*.py"):
        mod = importlib.import_module(".".join(path.relative_to(ROOT.parent).with_suffix("").parts))
        for routes in vars(mod).values():
            if isinstance(routes, Routes):
                seen += 1
                for key, module in routes.hosts.items():
                    assert importlib.util.find_spec(module) is not None, (routes.op, key, module)
                    assert routes.blocker(key) is None
                assert "no kernel for" in routes.blocker(("uncovered",))
    assert seen > 40


def test_importing_the_categories_imports_no_shape_package():
    code = ("import sys, importlib\n"
            f"for c in {CATEGORIES!r}: importlib.import_module('amd_kernels.' + c)\n"
            "print(sorted(m for m in sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2\n"
            "             and '_common' not in m))\n"
            "print('flydsl' in sys.modules)")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split("\n")[:2] == ["[]", "False"], out.stdout


def test_previous_names_warn_and_answer():
    import warnings

    import torch

    from amd_kernels import gemm, indexer, moe, routes

    routes._warn_once.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert gemm.mxfp8_postmix_blocker(16, 5120, 8192, False) is None
        assert gemm.mxfp8_grouped_gemm_blocker(16, 8, 1024, 4096) is None
        assert gemm.mxfp8_gate_up_swiglu_mxfp8_blocker(16, 4608, 5120, False) is None
        assert moe.expert_parallel_blocker(8, 5120, 2304, 6, 48, 1024) is None
        assert indexer.mxfp4_indexer_blocker(64, 128, 64, 4096, 1).startswith("mxfp4_indexer_topk: no kernel")
        assert gemm.bf16_gemm_blocker(torch.empty(999, 5120, device="meta")).startswith("bf16_gemm")
    assert caught and all(issubclass(w.category, DeprecationWarning) for w in caught)
