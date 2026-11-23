# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the logits exchange entry on meta tensors (no GPU): every table row imports and exports
what the entry calls, the factory's key comes from its dims and the group, a call's reason routes by the
communicator and compares the weight with it."""
import ast
import importlib
import inspect
import pathlib
import subprocess
import sys
import types

import pytest
import torch

from amd_kernels import routes
from amd_kernels.collectives import logits_exchange_common as E
from amd_kernels.collectives.logits_exchange_common import host as entry

BF16 = torch.bfloat16


def meta(*shape, dtype=BF16):
    return torch.empty(*shape, dtype=dtype, device="meta")


class _Dist:
    def __init__(self, world, rank=0):
        self.world, self.rank = world, rank

    def get_world_size(self, group):
        return self.world

    def get_rank(self, group):
        return self.rank


@pytest.fixture(autouse=True)
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def _comm(key, max_rows=256, closed=False):
    comm = object.__new__(entry.LogitsExchange)
    comm.key, comm.max_rows, comm.device = key, max_rows, torch.device("meta")
    comm.exchange = types.SimpleNamespace(closed=closed)
    return comm


def test_table_modules_export_what_the_entry_calls():
    assert entry._GEMMS.keys() == entry._ROUTES.hosts.keys() == entry._GEMM_ROUTES.hosts.keys()
    for key, name in entry._ROUTES.hosts.items():
        host = importlib.import_module(name)
        for method in ("send_args", "wait", "recv", "warmup", "close", "__call__"):
            assert callable(getattr(host.RowAllToAll, method))
        assert host.BUFFERS >= 1 and host.RECV_ROWS >= 1
        gemm_name, op, serial_above = entry._GEMMS[key]
        gemm = importlib.import_module(gemm_name)
        assert callable(getattr(gemm, op)) and callable(gemm.prepare)
        assert set(serial_above) == {True, False}


def test_exchange_packages_import_only_themselves():
    """No GEMM package import in an exchange package (module level or in a function): the entry composes them."""
    for name in entry._ROUTES.hosts.values():
        package = name.rpartition(".")[0]
        for path in pathlib.Path(importlib.import_module(package).__file__).parent.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
                for mod in names:
                    assert not mod.startswith("amd_kernels.") or mod.startswith((package, "amd_kernels.common")), \
                        (path.name, mod)


def test_shapes_are_the_table_keys():
    assert entry.logits_gemm_exchange_shapes() == frozenset(entry._ROUTES.hosts)


def test_old_names_are_aliases():
    for new in ("logits_exchange_comm_blocker", "logits_gemm_exchange_blocker"):
        old = new.removesuffix("_blocker") + "_unsupported_reason"
        assert getattr(E, old).renamed_to == new and getattr(entry, old).renamed_to == new
        assert getattr(E, new) is getattr(entry, new) and old not in E.__all__


def test_reason_signatures():
    sig = inspect.signature
    params = sig(entry.logits_exchange_comm).parameters
    assert sig(entry.logits_exchange_comm_blocker).parameters == params
    assert sig(entry.prepare_logits_exchange_comm).parameters == params
    method = list(sig(entry.LogitsExchange.gemm_exchange).parameters)
    assert list(sig(entry.logits_gemm_exchange_blocker).parameters) == ["comm"] + method[1:]


def test_routes(monkeypatch):
    for key, name in entry._ROUTES.hosts.items():
        n, k, world = key
        host = importlib.import_module(name)
        monkeypatch.setattr(entry, "dist", _Dist(world, rank=5))
        for max_rows in (1, host.RECV_ROWS):
            assert entry.logits_exchange_comm_blocker(5, None, "cuda", n_shard=n, k=k, max_rows=max_rows) is None
        comm = _comm(key)
        for rows in (1, comm.max_rows):
            idx = meta(rows, dtype=torch.int64)
            for counts in (None, meta(world, dtype=torch.int32)):
                for buf in range(host.BUFFERS):
                    assert entry.logits_gemm_exchange_blocker(comm, meta(rows, k), meta(n, k), idx, counts,
                                                              buf) is None


def test_reasons(monkeypatch):
    (key, name), = entry._ROUTES.hosts.items()
    n, k, world = key
    host = importlib.import_module(name)
    f = entry.logits_exchange_comm_blocker
    monkeypatch.setattr(entry, "dist", _Dist(world - 1))
    reason = f(0, None, "cuda", n_shard=n, k=k, max_rows=8)
    assert "no kernel" in reason and f"world={world - 1}" in reason and str(key) in reason
    with pytest.raises(ValueError) as e:
        entry.logits_exchange_comm(0, None, "cuda", n_shard=n, k=k, max_rows=8)
    assert str(e.value) == reason
    assert entry.prepare_logits_exchange_comm(0, None, "cuda", n_shard=n, k=k, max_rows=8) == reason
    monkeypatch.setattr(entry, "dist", _Dist(world, rank=2))
    assert "no kernel" in f(2, None, "cuda", n_shard=n, k=k + 1, max_rows=8)
    assert "max_rows 0" in f(2, None, "cuda", n_shard=n, k=k, max_rows=0)
    assert "max_rows" in f(2, None, "cuda", n_shard=n, k=k, max_rows=host.RECV_ROWS + 1)
    assert "rank 1" in f(1, None, "cuda", n_shard=n, k=k, max_rows=8)

    r = entry.logits_gemm_exchange_blocker
    comm, x, w, idx = _comm(key, max_rows=64), meta(8, k), meta(n, k), meta(8, dtype=torch.int64)
    assert "no kernel for communicator object" in r(object(), x, w, idx)
    assert "no kernel for communicator LogitsExchange" in r(_comm((n, k, world - 1)), x, w, idx)
    assert "communicator built for" in r(comm, x, meta(n // 2, k), idx)
    assert "communicator built for" in r(comm, meta(8, 2 * k), meta(n, 2 * k), idx)
    assert "closed" in r(_comm(key, closed=True), x, w, idx)
    assert "w must be" in r(comm, x, meta(n, k, dtype=torch.float16), idx)
    assert "x must be" in r(comm, meta(8, k, dtype=torch.float16), w, idx)
    assert "x must be" in r(comm, meta(8, 2 * k)[:, :k], w, idx)
    assert "row_indices must be" in r(comm, x, w, meta(8, dtype=torch.int32))
    assert "row_indices must be" in r(comm, x, w, meta(2, 4, dtype=torch.int64))
    assert "0 rows" in r(comm, x, w, meta(0, dtype=torch.int64))
    assert "65 rows" in r(comm, x, w, meta(65, dtype=torch.int64))
    assert "counts must be" in r(comm, x, w, idx, meta(world, dtype=torch.int64))
    assert "counts must be" in r(comm, x, w, idx, meta(world + 1, dtype=torch.int32))
    assert "buffer" in r(comm, x, w, idx, None, host.BUFFERS)
    monkeypatch.setattr(routes, "available", lambda: False)
    assert "gfx950" in r(comm, x, w, idx) and "gfx950" in f(2, None, "cuda", n_shard=n, k=k, max_rows=8)


def test_import_loads_no_shape_package():
    code = ("import sys; from amd_kernels.collectives.logits_exchange_common import host as e; "
            "names = [v.rpartition('.')[0] for v in e._ROUTES.hosts.values()]; "
            "names.append('amd_kernels.gemm.bf16_n20480_k7168'); "
            "print(sorted(n for n in names if n in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out
