# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry's routing and reasons on meta tensors (no GPU): every table row imports and exports the communicator, the
factory key comes from ``hidden`` and the group, a call routes by its communicator, and every reason branch."""
import importlib
import inspect
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from amd_kernels import routes
from amd_kernels.norm import attnres_rmsnorm_common as N
from amd_kernels.norm.attnres_rmsnorm_common import host as entry

KEY = (7168, 8)


def meta(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


@pytest.fixture
def comm():
    c = object.__new__(importlib.import_module(entry._ROUTES.hosts[KEY]).AttnResComm)
    c.__dict__.update(bufs=[1], device=torch.device("meta"))
    return c


def args(m=12, h=7168, nb=8):
    return dict(partial=meta(m, h), prefix=meta(m, h), bank=meta(m, nb, h), norm_weight=meta(h), qk_weight=meta(h),
                num_blocks=3)


def test_every_row_imports_and_exports_the_communicator():
    for name in entry._ROUTES.hosts.values():
        host = importlib.import_module(name)
        assert all(hasattr(host, a) for a in ("AttnResComm", "MAX_ROWS", "BANK_ROWS")), name
    assert N.allreduce_attnres_rmsnorm_shapes() == frozenset(entry._ROUTES.hosts)


def test_importing_the_entry_imports_no_shape_package():
    code = ("import sys; import amd_kernels.norm.attnres_rmsnorm_common; "
            "print(sorted(m for m in sys.modules if m.startswith('amd_kernels.norm.') and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "[]"


def test_reason_takes_the_method_arguments():
    host = importlib.import_module(entry._ROUTES.hosts[KEY])
    method = list(inspect.signature(host.AttnResComm.allreduce_attnres_rmsnorm).parameters.values())[1:]
    reason = list(inspect.signature(N.allreduce_attnres_rmsnorm_blocker).parameters.values())
    assert [(p.name, p.default) for p in reason[1:]] == [(p.name, p.default) for p in method]
    factory = inspect.signature(N.attnres_comm).parameters
    assert factory == inspect.signature(N.attnres_comm_blocker).parameters


def test_factory_key(gfx950, monkeypatch):
    monkeypatch.setattr(entry.dist, "get_world_size", lambda group: 8)
    assert N.attnres_comm_blocker(0, None, "meta", hidden=7168) is None
    reason = N.attnres_comm_blocker(0, None, "meta", hidden=6144)
    assert reason == "attnres_comm: no kernel for hidden=6144, world=8 (covered: [(7168, 8)])"
    with pytest.raises(ValueError) as e:
        N.attnres_comm(0, None, "meta", hidden=6144)
    assert str(e.value) == reason
    monkeypatch.setattr(entry.dist, "get_world_size", lambda group: 4)
    assert "world=4" in N.attnres_comm_blocker(0, None, "meta", hidden=7168)


def test_served_calls(gfx950, comm):
    r = N.allreduce_attnres_rmsnorm_blocker
    for m in (1, 256):
        assert r(comm, **args(m)) is None
        assert r(comm, **{**args(m), "prefix": None}, block_write_idx=4, out_norm_weight=meta(7168)) is None
    assert r(comm, **{**args(), "bank": None}) is None
    assert r(comm, **{**args(), "prefix": meta(12, 7176)[:, :7168]}) is None


def test_reasons(gfx950, comm, monkeypatch):
    r = N.allreduce_attnres_rmsnorm_blocker
    want = "allreduce_attnres_rmsnorm: communicator built for hidden=7168, world=8, call is hidden=6144"
    assert r(comm, **args(h=6144)) == want
    assert "257 rows (the kernel takes 1..256)" in r(comm, **args(257))
    assert "0 rows" in r(comm, **args(0))
    assert "partial must be" in r(comm, **{**args(), "partial": meta(12, 7168, dtype=torch.float32)})
    assert "prefix must be" in r(comm, **{**args(), "prefix": meta(12, 7170)[:, :7168]})
    assert "bank must be" in r(comm, **args(nb=9))
    assert "a bank needs norm_weight and qk_weight" in r(comm, **{**args(), "norm_weight": None})
    assert "a bank needs norm_weight and qk_weight" in r(comm, **{**args(), "qk_weight": None})
    assert "bank must be" in r(comm, **{**args(), "bank": meta(12, 8, 7172)[:, :, :7168]})
    assert "num_blocks" in r(comm, **{**args(), "num_blocks": 9})
    assert "num_blocks" in r(comm, **args(), block_write_idx=2)
    assert "weights must be" in r(comm, **{**args(), "qk_weight": meta(7168, dtype=torch.float32)})
    assert "weights must be" in r(comm, **args(), out_norm_weight=meta(6144))
    other = SimpleNamespace(bufs=[1], device=torch.device("meta"))
    assert "no kernel for a SimpleNamespace (covered: [(7168, 8)])" in r(other, **args())
    comm.bufs = []
    assert "closed" in r(comm, **args())
    monkeypatch.setattr(routes, "available", lambda: False)
    assert r(comm, **args()) == "allreduce_attnres_rmsnorm: needs gfx950 with FlyDSL"


def test_prepare_returns_the_reason(gfx950, monkeypatch):
    monkeypatch.setattr(entry.dist, "get_world_size", lambda group: 8)
    assert N.prepare_attnres_comm(0, None, "meta", hidden=6144) == N.attnres_comm_blocker(0, None, "meta",
                                                                                          hidden=6144)


def test_old_names_alias_the_blockers():
    from amd_kernels.norm.attnres_rmsnorm_common import (
        allreduce_attnres_rmsnorm_unsupported_reason,
        attnres_comm_unsupported_reason,
    )
    assert allreduce_attnres_rmsnorm_unsupported_reason.renamed_to == "allreduce_attnres_rmsnorm_blocker"
    assert attnres_comm_unsupported_reason.renamed_to == "attnres_comm_blocker"
