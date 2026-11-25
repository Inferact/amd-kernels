# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the dense MLP entry on meta tensors (no GPU): every table row imports and exports what the
entry calls, the key comes from the down weight, an uncovered call gets a reason and the op raises it."""
import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.decoder import bf16_dense_mlp_common as E
from amd_kernels.decoder.bf16_dense_mlp_common import host as entry

BF16 = torch.bfloat16


def meta(*shape, dtype=BF16):
    return torch.empty(*shape, dtype=dtype, device="meta")


@pytest.fixture(autouse=True)
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def _ws(host):
    ws = object.__new__(host.MlpWorkspace)
    ws.slab = meta(1, dtype=torch.int32)
    return ws


def test_table_modules_export_what_the_entry_calls():
    for name in entry._ROUTES.hosts.values():
        host = importlib.import_module(name)
        for attr in ("MlpWorkspace", "dense_mlp", "pack_gate_up", "pack_down", "packed_tiles", "packed_rows_buffer",
                     "MAX_ROWS"):
            assert hasattr(host, attr), (name, attr)


def test_shapes_are_the_table_keys():
    assert entry.bf16_dense_mlp_shapes() == frozenset(entry._ROUTES.hosts)


def test_old_names_are_aliases():
    for new in ("dense_mlp_workspace_blocker", "bf16_dense_mlp_blocker"):
        old = new.removesuffix("_blocker") + "_unsupported_reason"
        assert getattr(E, old).renamed_to == new and getattr(entry, old).renamed_to == new
        assert getattr(E, new) is getattr(entry, new) and old not in E.__all__


def test_reason_signatures():
    sig = inspect.signature
    assert sig(entry.bf16_dense_mlp_blocker).parameters == sig(entry.bf16_dense_mlp).parameters
    assert sig(entry.dense_mlp_workspace_blocker).parameters == sig(entry.dense_mlp_workspace).parameters
    assert sig(entry.prepare_dense_mlp_workspace).parameters == sig(entry.dense_mlp_workspace).parameters
    for name in entry._ROUTES.hosts.values():
        assert list(sig(entry.bf16_dense_mlp).parameters) == \
            list(sig(importlib.import_module(name).dense_mlp).parameters)


def test_routes():
    for key, name in entry._ROUTES.hosts.items():
        hidden, inter = key
        host = importlib.import_module(name)
        down, gate_up = meta(hidden, inter), meta(2 * inter, hidden)
        assert entry._ROUTES.host(tuple(down.shape)) is host
        assert entry._ROUTES.owner(host.MlpWorkspace) == key
        assert entry.dense_mlp_workspace_blocker("cuda", hidden=hidden, inter=inter) is None
        for rows in (1, host.MAX_ROWS):
            tiles = host.packed_tiles(rows)
            assert entry.dense_mlp_packed_tiles(rows, down=down) == tiles
            xp = meta(tiles, hidden // 32, 64, 8)
            assert entry.bf16_dense_mlp_blocker(_ws(host), xp, gate_up, down, rows) is None
            assert entry.bf16_dense_mlp_blocker(_ws(host), xp, gate_up, down, rows, meta(rows, hidden)) is None


def test_reasons(monkeypatch):
    (key, name), = entry._ROUTES.hosts.items()
    hidden, inter = key
    host = importlib.import_module(name)
    r = entry.bf16_dense_mlp_blocker
    ws, down, gate_up = _ws(host), meta(hidden, inter), meta(2 * inter, hidden)
    xp = meta(host.packed_tiles(16), hidden // 32, 64, 8)
    off = meta(hidden, inter * 2)
    reason = r(ws, xp, gate_up, off, 16)
    assert "no kernel" in reason and f"inter={inter * 2}" in reason and str(key) in reason
    with pytest.raises(ValueError) as e:
        entry.bf16_dense_mlp(ws, xp, gate_up, off, 16)
    assert str(e.value) == reason
    assert "no kernel" in entry.dense_mlp_workspace_blocker("cuda", hidden=hidden, inter=inter * 2)
    for f, arg in ((entry.dense_mlp_pack_down, off), (entry.dense_mlp_pack_gate_up, meta(4 * inter, hidden)),
                   (entry.dense_mlp_pack_gate_up, meta(3, hidden))):
        with pytest.raises(ValueError, match="no kernel"):
            f(arg)
    with pytest.raises(ValueError, match="no kernel"):
        entry.dense_mlp_workspace("cuda", hidden=hidden, inter=inter * 2)
    assert entry.prepare_dense_mlp_workspace("cuda", hidden=hidden, inter=inter * 2) == \
        entry.dense_mlp_workspace_blocker("cuda", hidden=hidden, inter=inter * 2)
    with pytest.raises(ValueError, match="no kernel"):
        entry.dense_mlp_packed_tiles(16, down=off)
    with pytest.raises(ValueError, match="no kernel"):
        entry.dense_mlp_rows_buffer(16, "meta", down=off)
    assert "no kernel for workspace object" in r(object(), xp, gate_up, down, 16)
    assert "down must be" in r(ws, xp, gate_up, meta(hidden, inter, dtype=torch.float16), 16)
    assert "down must be" in r(ws, xp, gate_up, meta(inter, hidden).t(), 16)
    assert "gate_up must be" in r(ws, xp, meta(inter, hidden), down, 16)
    assert "gate_up must be" in r(ws, xp, meta(2 * inter, hidden, dtype=torch.float32), down, 16)
    other = _ws(host)
    other.slab = torch.empty(1, dtype=torch.int32)
    assert "workspace on cpu" in r(other, xp, gate_up, down, 16)
    assert "0 rows" in r(ws, xp, gate_up, down, 0)
    assert f"{host.MAX_ROWS + 1} rows" in r(ws, xp, gate_up, down, host.MAX_ROWS + 1)
    assert "x_packed must be" in r(ws, xp, gate_up, down, host.MAX_ROWS)
    assert "x_packed must be" in r(ws, meta(xp.shape[0], hidden // 64, 64, 8), gate_up, down, 16)
    assert "x_packed must be" in r(ws, xp.float(), gate_up, down, 16)
    assert "out must be" in r(ws, xp, gate_up, down, 16, meta(16, inter))
    monkeypatch.setattr(routes, "available", lambda: False)
    assert "gfx950" in r(ws, xp, gate_up, down, 16)
    assert "gfx950" in entry.dense_mlp_workspace_blocker("cuda", hidden=hidden, inter=inter)


def test_import_loads_no_shape_package():
    code = ("import sys; from amd_kernels.decoder.bf16_dense_mlp_common import host as e; "
            "names = [v.rpartition('.')[0] for v in e._ROUTES.hosts.values()]; "
            "print(sorted(n for n in names if n in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out
