# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the gather + merge entry on one process (host only, no GPU): the table rows import and
export the communicator, the factory refuses a world size off the table, the call's reason routes by the
communicator's type and every branch fires on meta operands."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.collectives import topk_gather_merge_common as E
from amd_kernels.collectives.topk_gather_merge_common import host

K = 2048


@pytest.fixture(scope="module")
def group():
    own = not dist.is_initialized()
    if own:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield dist.group.WORLD
    if own:
        dist.destroy_process_group()


@pytest.fixture
def on_gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def fake_comm(key=(K, 8), closed=False):
    """A communicator object of the routed class without the collective construction."""
    cls = importlib.import_module(host._ROUTES.hosts[key]).TopkGatherMerge
    comm = object.__new__(cls)
    comm.closed, comm.device = closed, torch.device("meta")
    return comm


def call_args(rows=12, topk=K, blocks=64):
    z = lambda *s, dtype=torch.int32: torch.empty(*s, dtype=dtype, device="meta")  # noqa: E731
    return dict(scores=z(rows, topk, dtype=torch.float32), local_idx=z(rows, topk), block_table=z(rows, blocks),
                out_slots=z(rows * topk), out_indptr=z(rows + 1), page_size=64, block_stride_rows=72)


def test_table_imports_and_exports():
    for (topk, world), name in host._ROUTES.hosts.items():
        mod = importlib.import_module(name)
        assert {"rank", "group", "device"} <= set(inspect.signature(mod.TopkGatherMerge).parameters)
        assert callable(mod.TopkGatherMerge.warmup) and callable(mod.TopkGatherMerge.close)
        assert mod.host.MAX_ROWS >= 1 and mod.kernel.K == topk and mod.kernel.WORLD == world
    assert host.topk_gather_merge_shapes() == frozenset(host._ROUTES.hosts)


def test_call_reason_takes_the_call_arguments():
    params = list(inspect.signature(host.topk_gather_merge_blocker).parameters)
    call = importlib.import_module(host._ROUTES.hosts[(K, 8)]).TopkGatherMerge.__call__
    assert params == ["comm"] + list(inspect.signature(call).parameters)[1:]
    params = inspect.signature(host.topk_gather_merge_comm).parameters
    assert params == inspect.signature(host.topk_gather_merge_comm_blocker).parameters
    assert params == inspect.signature(host.prepare_topk_gather_merge_comm).parameters


def test_old_names_are_aliases():
    for new in ("topk_gather_merge_comm_blocker", "topk_gather_merge_blocker"):
        old = new.removesuffix("_blocker") + "_unsupported_reason"
        assert getattr(E, old).renamed_to == new and getattr(host, old).renamed_to == new
        assert getattr(E, new) is getattr(host, new) and old not in E.__all__


def test_factory_off_table(group, on_gfx950):
    why = host.topk_gather_merge_comm_blocker(0, group, "cpu", topk=K)
    assert why == f"topk_gather_merge_comm: no kernel for topk={K}, world=1 (covered: [({K}, 8)])"
    with pytest.raises(ValueError) as err:
        host.topk_gather_merge_comm(0, group, "cpu", topk=K)
    assert str(err.value) == why
    assert host.prepare_topk_gather_merge_comm(0, group, "cpu", topk=K) == why
    assert "topk=1024, world=1" in host.topk_gather_merge_comm_blocker(0, group, "cpu", topk=1024)


def test_factory_rank_and_platform(group, monkeypatch):
    monkeypatch.setitem(host._ROUTES.hosts, (K, 1), host._ROUTES.hosts[(K, 8)])
    monkeypatch.setattr(routes, "available", lambda: True)
    assert host.topk_gather_merge_comm_blocker(0, group, "cpu", topk=K) is None
    assert "rank 3 is not" in host.topk_gather_merge_comm_blocker(3, group, "cpu", topk=K)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    why = host.topk_gather_merge_comm_blocker(0, None, "cpu", topk=K)
    assert why == "topk_gather_merge_comm: torch.distributed is not initialized"
    monkeypatch.setattr(routes, "available", lambda: False)
    why = host.topk_gather_merge_comm_blocker(0, group, "cpu", topk=K)
    assert why == "topk_gather_merge_comm: needs gfx950 with FlyDSL"


def test_call_reason_branches():
    comm = fake_comm()
    a = call_args()
    assert host.topk_gather_merge_blocker(comm, **a) is None
    assert host.topk_gather_merge_blocker(comm, **call_args(rows=1)) is None
    assert host.topk_gather_merge_blocker(comm, **call_args(rows=256)) is None
    z = lambda *s, dtype=torch.int32: torch.empty(*s, dtype=dtype, device="meta")  # noqa: E731
    cases = [
        (object(), a, "object is not a communicator of this entry"),
        (fake_comm(closed=True), a, "communicator closed"),
        (comm, call_args(topk=1024), f"communicator built for topk={K}, world=8, call is topk=1024"),
        (comm, {**a, "scores": z(12 * K, dtype=torch.float32)}, "call is topk=None"),
        (comm, call_args(rows=257), "257 rows (the kernel takes 1..256)"),
        (comm, call_args(rows=0), "0 rows"),
        (comm, {**a, "scores": z(12, K, dtype=torch.bfloat16)}, "scores must be fp32"),
        (comm, {**a, "scores": z(K, 12, dtype=torch.float32).t()}, "scores must be fp32"),
        (comm, {**a, "local_idx": z(12, K, dtype=torch.int64)}, "local_idx"),
        (comm, {**a, "local_idx": z(11, K)}, "local_idx"),
        (comm, {**a, "block_table": z(11, 64)}, "block_table"),
        (comm, {**a, "block_table": z(64, 12).t()}, "block_table"),
        (comm, {**a, "block_table": z(12, 64, dtype=torch.int64)}, "block_table"),
        (comm, {**a, "out_slots": z(12 * K - 1)}, "out_slots"),
        (comm, {**a, "out_indptr": z(12)}, "out_indptr"),
        (comm, {**a, "out_indptr": z(13, dtype=torch.int64)}, "out_indptr"),
        (comm, {**a, "block_table": torch.empty(12, 64, dtype=torch.int32)}, "communicator's device"),
    ]
    for c, args, want in cases:
        why = host.topk_gather_merge_blocker(c, **args)
        assert why is not None and why.startswith("topk_gather_merge: ") and want in why, (want, why)


def test_import_loads_no_shape_package():
    code = (
        "import sys; import amd_kernels.collectives.topk_gather_merge_common as e; "
        "print(sorted(m for m in sys.modules if m.startswith('amd_kernels.collectives.') and 'common' not in m))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
