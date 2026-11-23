# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the bf16 GEMM entries on meta tensors (no GPU): every table row imports and exports what
the entry calls, keys come from the operands, an uncovered call gets a reason and the op raises it."""
import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.gemm import bf16_common as E
from amd_kernels.gemm.bf16_common import latent_down, linear_gather

BF16 = torch.bfloat16


def meta(*shape, dtype=BF16):
    return torch.empty(*shape, dtype=dtype, device="meta")


@pytest.fixture(autouse=True)
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


class _Dist:
    def __init__(self, world, rank=0):
        self.world, self.rank = world, rank

    def get_world_size(self, group):
        return self.world

    def get_rank(self, group):
        return self.rank


def _comm(host, key, closed=False):
    comm = object.__new__(host.ProjComm)
    comm.bufs, comm.device = ([] if closed else [0] * key[2]), torch.device("meta")
    return comm


def test_table_modules_export_what_the_entries_call():
    for name in latent_down._ROUTES.hosts.values():
        host = importlib.import_module(name)
        assert callable(getattr(host, name.split(".")[-2])) and callable(host.supports) and host.MAX_M >= 1
        assert isinstance(host._XCD_LOCAL, dict)
    for name in linear_gather._ROUTES.hosts.values():
        host = importlib.import_module(name)
        for attr in ("ProjComm", "context_proj_gather_norm", "pack_weight", "pack_aux", "packed_tiles",
                     "packed_rows_buffer", "MAX_ROWS", "TIERS"):
            assert hasattr(host, attr), (name, attr)
        assert importlib.import_module(host.__package__ + ".pack_rows").LAYERS >= 1


def test_shapes_are_the_table_keys():
    assert latent_down.bf16_latent_down_mxfp4_quant_shapes() == frozenset(latent_down._ROUTES.hosts)
    assert linear_gather.bf16_linear_gather_shapes() == frozenset(linear_gather._ROUTES.hosts)


def test_old_names_are_aliases():
    for mod, new in ((latent_down, "bf16_latent_down_mxfp4_quant_blocker"),
                     (linear_gather, "bf16_linear_gather_blocker"), (linear_gather, "linear_gather_comm_blocker"),
                     (linear_gather, "linear_gather_pack_aux_blocker")):
        old = new.removesuffix("_blocker") + "_unsupported_reason"
        assert getattr(E, old).renamed_to == new and getattr(mod, old).renamed_to == new
        assert getattr(E, new) is getattr(mod, new) and old not in E.__all__


def test_reason_signatures_are_the_ops():
    sig = inspect.signature
    assert sig(latent_down.bf16_latent_down_mxfp4_quant_blocker).parameters == \
        sig(latent_down.bf16_latent_down_mxfp4_quant).parameters
    assert sig(linear_gather.bf16_linear_gather_blocker).parameters == \
        sig(linear_gather.bf16_linear_gather).parameters
    assert sig(linear_gather.linear_gather_pack_aux_blocker).parameters == \
        sig(linear_gather.linear_gather_pack_aux).parameters
    assert sig(linear_gather.linear_gather_comm_blocker).parameters == \
        sig(linear_gather.linear_gather_comm).parameters == sig(linear_gather.prepare_linear_gather_comm).parameters
    for name in latent_down._ROUTES.hosts.values():
        host = importlib.import_module(name)
        assert list(sig(latent_down.bf16_latent_down_mxfp4_quant_weight_reason).parameters) == \
            list(sig(host.supports).parameters)
        assert list(sig(latent_down.bf16_latent_down_mxfp4_quant).parameters) == \
            list(sig(getattr(host, name.split(".")[-2])).parameters)
    for name in linear_gather._ROUTES.hosts.values():
        host = importlib.import_module(name)
        assert list(sig(linear_gather.bf16_linear_gather).parameters) == \
            list(sig(host.context_proj_gather_norm).parameters)


def test_latent_down_routes_by_weight():
    for (n, k), name in latent_down._ROUTES.hosts.items():
        w = meta(n, k)
        assert latent_down._ROUTES.host(tuple(w.shape)) is importlib.import_module(name)
        host = latent_down._ROUTES.host((n, k))
        for m in (1, host.MAX_M):
            x = meta(m, k)
            out = (meta(m, n // 2, dtype=torch.uint8), meta(m, n // 32, dtype=torch.uint8))
            assert latent_down.bf16_latent_down_mxfp4_quant_blocker(x, w) is None
            assert latent_down.bf16_latent_down_mxfp4_quant_blocker(x, w, out, out_bf16=meta(m, n)) is None
        assert latent_down.bf16_latent_down_mxfp4_quant_weight_reason(w) is None


def test_latent_down_reasons(monkeypatch):
    (n, k), = latent_down._ROUTES.hosts
    host = latent_down._ROUTES.host((n, k))
    w, x = meta(n, k), meta(4, k)
    r = latent_down.bf16_latent_down_mxfp4_quant_blocker
    off = meta(n, k + 64)
    reason = r(x, off)
    assert "no kernel" in reason and str((n, k)) in reason
    with pytest.raises(ValueError) as e:
        latent_down.bf16_latent_down_mxfp4_quant(x, off)
    assert str(e.value) == reason
    with pytest.raises(ValueError, match="no kernel"):
        latent_down.latent_down_xcd_local(0, False, w=off)
    assert "supports" in latent_down.bf16_latent_down_mxfp4_quant_weight_reason(w, meta(n))
    assert "supports" in r(x, meta(n, k, dtype=torch.float16))
    assert "supports" in r(x, meta(k, n).t())
    assert "x must be" in r(meta(4, k, dtype=torch.float16), w)
    assert "x must be" in r(meta(4, 2 * k)[:, :k], w)
    assert "x must be" in r(meta(4, k - 32), w)
    assert r(meta(0, k), w) is None and r(meta(2 * host.MAX_M + 1, k), w) is None
    assert "out must be" in r(x, w, (meta(4, n // 2, dtype=torch.uint8), meta(4, n // 16, dtype=torch.uint8)))
    assert "out must be" in r(x, w, (meta(4, n // 2, dtype=BF16), meta(4, n // 32, dtype=torch.uint8)))
    assert "out_bf16 must be" in r(x, w, out_bf16=meta(4, n, dtype=torch.float32))
    assert "out_bf16 must be" in r(x, w, out_bf16=meta(5, n))
    monkeypatch.setattr(routes, "available", lambda: False)
    assert "gfx950" in r(x, w) and "gfx950" in latent_down.bf16_latent_down_mxfp4_quant_weight_reason(w)


def test_latent_down_xcd_switch():
    (n, k), = latent_down._ROUTES.hosts
    host = latent_down._ROUTES.host((n, k))
    saved = dict(host._XCD_LOCAL)
    try:
        latent_down.latent_down_xcd_local(7, False, w=meta(n, k))
        assert host._XCD_LOCAL[7] is False
    finally:
        host._XCD_LOCAL.clear()
        host._XCD_LOCAL.update(saved)


def test_latent_down_swap_stream_workspaces():
    (n, k), = latent_down._ROUTES.hosts
    host = latent_down._ROUTES.host((n, k))
    saved = dict(host._WORK)
    try:
        host._WORK.clear()
        host._WORK.update({(0, 1): "a", (0, 2): "b", (1, 1): "c"})
        latent_down.latent_down_swap_stream_workspaces(0, 1, 5)
        assert host._WORK == {(0, 5): "a", (0, 2): "b", (1, 1): "c"}
    finally:
        host._WORK.clear()
        host._WORK.update(saved)


def test_linear_gather_routes(monkeypatch):
    for key, name in linear_gather._ROUTES.hosts.items():
        n_shard, k, world = key
        host = importlib.import_module(name)
        monkeypatch.setattr(linear_gather, "dist", _Dist(world, rank=3))
        assert linear_gather.linear_gather_comm_blocker(3, None, "cuda", n_shard=n_shard, k=k) is None
        comm = _comm(host, key)
        assert linear_gather._comm_key(type(comm)) == key
        w = meta(n_shard, k)
        assert linear_gather._weight_key(tuple(w.shape)) == key
        layers = linear_gather._layers(host)
        for rows in (1, linear_gather._max_rows(host)):
            tiles = host.packed_tiles(rows)
            assert linear_gather.linear_gather_packed_tiles(rows, w_packed=w) == tiles
            xp = meta(tiles, k // 32, 64, 8)
            aux = [meta(rows, k // layers) for _ in range(layers)]
            assert linear_gather.linear_gather_pack_aux_blocker(
                aux, meta(tiles * (k // 32) * 512), rows, w_packed=w) is None
            assert linear_gather.bf16_linear_gather_blocker(
                comm, xp, w, meta(n_shard * world), rows, out=meta(rows, n_shard * world)) is None


def test_linear_gather_reasons(monkeypatch):
    (key, name), = linear_gather._ROUTES.hosts.items()
    n_shard, k, world = key
    host = importlib.import_module(name)
    r = linear_gather.bf16_linear_gather_blocker
    comm, w, nw = _comm(host, key), meta(n_shard, k), meta(n_shard * world)
    xp = meta(host.packed_tiles(16), k // 32, 64, 8)
    hi = linear_gather._max_rows(host)
    monkeypatch.setattr(linear_gather, "dist", _Dist(world + 1))
    reason = linear_gather.linear_gather_comm_blocker(0, None, "cuda", n_shard=n_shard, k=k)
    assert "no kernel" in reason and f"world={world + 1}" in reason and str(key) in reason
    with pytest.raises(ValueError, match="no kernel"):
        linear_gather.linear_gather_comm(0, None, "cuda", n_shard=n_shard, k=k)
    assert linear_gather.prepare_linear_gather_comm(0, None, "cuda", n_shard=n_shard, k=k) == reason
    monkeypatch.setattr(linear_gather, "dist", _Dist(world, rank=1))
    assert "rank 0" in linear_gather.linear_gather_comm_blocker(0, None, "cuda", n_shard=n_shard, k=k)
    stranger = object()
    assert "no kernel for communicator object" in r(stranger, xp, w, nw, 16)
    with pytest.raises(ValueError, match="no kernel for communicator"):
        linear_gather.bf16_linear_gather(stranger, xp, w, nw, 16)
    assert "communicator built for" in r(comm, xp, meta(n_shard * 2, k), nw, 16)
    assert "closed" in r(_comm(host, key, closed=True), xp, w, nw, 16)
    assert "w_packed must be" in r(comm, xp, meta(n_shard, k, dtype=torch.float16), nw, 16)
    assert "0 rows" in r(comm, xp, w, nw, 0)
    assert f"{hi + 1} rows" in r(comm, xp, w, nw, hi + 1)
    assert "x_packed must be" in r(comm, xp, w, nw, hi)
    assert "x_packed must be" in r(comm, meta(xp.shape[0], k // 64, 64, 8), w, nw, 16)
    assert "norm_weight must be" in r(comm, xp, w, meta(n_shard), 16)
    assert "out must be" in r(comm, xp, w, nw, 16, out=meta(15, n_shard * world))
    off = meta(n_shard, k + 32)
    with pytest.raises(ValueError, match="no kernel"):
        linear_gather.linear_gather_pack_weight(off)
    with pytest.raises(ValueError, match="no kernel"):
        linear_gather.linear_gather_packed_tiles(16, w_packed=off)
    with pytest.raises(ValueError, match="no kernel"):
        linear_gather.linear_gather_rows_buffer(16, "meta", w_packed=off)
    with pytest.raises(ValueError, match="no kernel"):
        linear_gather.linear_gather_pack_aux([], off, 16, w_packed=off)
    pa = linear_gather.linear_gather_pack_aux_blocker
    layers = linear_gather._layers(host)
    aux = [meta(16, k // layers) for _ in range(layers)]
    out = meta(host.packed_tiles(16) * (k // 32) * 512)
    assert "no kernel" in pa(aux, out, 16, w_packed=meta(n_shard, 2 * k))
    assert "0 rows" in pa(aux, out, 0, w_packed=w)
    assert "aux must be" in pa(aux[:-1], out, 16, w_packed=w)
    assert "aux must be" in pa(aux, out, 17, w_packed=w)
    assert "out must be" in pa(aux, meta(out.numel() + 1), 16, w_packed=w)
    monkeypatch.setattr(routes, "available", lambda: False)
    assert "gfx950" in r(comm, xp, w, nw, 16) and "gfx950" in pa(aux, out, 16, w_packed=w)


def test_import_loads_no_shape_package():
    code = ("import sys, amd_kernels.gemm.bf16_common as e; "
            "from amd_kernels.gemm.bf16_common import latent_down as a, linear_gather as b; "
            "names = [v.rpartition('.')[0] for t in (a._ROUTES.hosts, b._ROUTES.hosts) for v in t.values()]; "
            "print(sorted(n for n in names if n in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out
