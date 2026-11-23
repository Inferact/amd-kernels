# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entries' routing and reasons on meta tensors (no GPU): every table row imports and exports what the entry
calls, keys come from the operands, an off-table call gets a reason (the op raises it), and every reason branch."""
import importlib
import inspect
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from amd_kernels import routes
from amd_kernels.moe import latent_moe_common as E
from amd_kernels.moe.latent_moe_common import (
    experts,
    latent_tail,
    layer_tail,
    router,
)

bf, u8, f32, i32 = torch.bfloat16, torch.uint8, torch.float32, torch.int32
ROUTED = {
    router: ("router_sigmoid_bias_top16_e896_h7168", "make_aux", "supports", "MAX_M", "aux_capacity"),
    experts: ("MoeWeights", "Sort", "moe_sort", "mxfp4_moe_experts", "quantize_activations", "sort_capacity", "MAX_M",
              "BM", "TOPK", "E", "H"),
    latent_tail: ("LatentTailComm", "allreduce_latent_rmsnorm_up_proj", "allreduce_slots_rmsnorm_up_proj", "MAX_ROWS",
                  "T"),
    layer_tail: ("LayerTailRunner", "K2Build", "k2_launch", "SERVED_ROWS"),
}
PAIRS = [
    (E.router_sigmoid_bias_topk, E.router_sigmoid_bias_topk_blocker),
    (E.mxfp4_moe_experts, E.mxfp4_moe_experts_blocker),
    (E.moe_weights, E.moe_weights_blocker),
    (E.moe_sort_from_aux, E.moe_sort_from_aux_blocker),
    (E.quantize_moe_activations, E.quantize_moe_activations_blocker),
    (E.latent_tail_comm, E.latent_tail_comm_blocker),
    (E.allreduce_latent_rmsnorm_up_proj, E.allreduce_latent_rmsnorm_up_proj_blocker),
    (E.allreduce_slots_rmsnorm_up_proj, E.allreduce_slots_rmsnorm_up_proj_blocker),
    (E.layer_tail_runner, E.layer_tail_runner_blocker),
    (E.layer_tail_launch, E.layer_tail_launch_blocker),
]


def meta(*shape, dtype=bf):
    return torch.empty(*shape, dtype=dtype, device="meta")


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


@pytest.mark.parametrize("module", list(ROUTED), ids=lambda m: m.__name__.rsplit(".", 1)[1])
def test_every_row_imports_and_exports_the_routed_names(module):
    for name in module._ROUTES.hosts.values():
        host = importlib.import_module(name)
        assert all(hasattr(host, a) for a in ROUTED[module]), name


def test_shapes_are_the_tables():
    assert E.router_sigmoid_bias_topk_shapes() == frozenset(router._ROUTES.hosts)
    assert E.mxfp4_moe_experts_shapes() == frozenset(experts._ROUTES.hosts)
    assert E.allreduce_latent_rmsnorm_up_proj_shapes() == frozenset(latent_tail._ROUTES.hosts)
    assert E.allreduce_slots_rmsnorm_up_proj_shapes() == frozenset(latent_tail._ROUTES.hosts)
    assert E.layer_tail_launch_shapes() == frozenset(layer_tail._ROUTES.hosts)


def test_importing_the_entries_imports_no_shape_package():
    code = ("import sys; import amd_kernels.moe.latent_moe_common; "
            "print(sorted(m for m in sys.modules if m.startswith('amd_kernels.moe.') and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "[]"


@pytest.mark.parametrize("op,reason", PAIRS, ids=lambda f: getattr(f, "__name__", ""))
def test_reason_takes_the_op_arguments(op, reason):
    assert list(inspect.signature(op).parameters.values()) == list(inspect.signature(reason).parameters.values())


def test_off_gfx950_every_reason_says_so(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    w = meta(896, 7168)
    assert E.router_sigmoid_bias_topk_weight_blocker(w).endswith("needs gfx950 with FlyDSL")
    assert "needs gfx950" in E.moe_weights_blocker(meta(896, 768, 1792, dtype=u8), w, w, w)
    assert "needs gfx950" in E.layer_tail_launch_blocker(None, {}, **{k: None for k in LAUNCH_KW})


# router


def router_args(m=12, e=896, h=7168, **kw):
    return dict(x=meta(m, h), w=meta(e, h), bias=meta(e, dtype=f32), **kw)


def test_router_routes_by_gate_shape_and_topk(gfx950):
    host = router._ROUTES.package((896, 7168, 16))
    assert host.__name__ == router._ROUTES.hosts[(896, 7168, 16)]
    for m in (1, 192):
        aux = E.router_aux(meta(896, 7168), m, torch.device("meta"))
        assert E.router_sigmoid_bias_topk_blocker(**router_args(m), aux=aux) is None
        assert E.router_sigmoid_bias_topk_blocker(**router_args(m)) is None


@pytest.mark.parametrize("kw", [dict(e=384), dict(h=6144), dict(topk=8)])
def test_router_off_table(gfx950, kw):
    topk = kw.pop("topk", 16)
    args = router_args(**kw)
    reason = E.router_sigmoid_bias_topk_blocker(**args, topk=topk)
    assert "no kernel for" in reason and "(896, 7168, 16)" in reason
    with pytest.raises(ValueError) as e:
        E.router_sigmoid_bias_topk(**args, topk=topk)
    assert str(e.value) == reason


def test_router_reasons(gfx950):
    r = E.router_sigmoid_bias_topk_blocker
    w = meta(896, 7168)
    assert "operand layout" in E.router_sigmoid_bias_topk_weight_blocker(meta(896, 7168, dtype=f32))
    assert "operand layout" in E.router_sigmoid_bias_topk_weight_blocker(w, meta(896, dtype=bf))
    assert E.router_sigmoid_bias_topk_weight_blocker(w) is None
    assert "correction bias" in r(meta(4, 7168), w, None)
    assert "193 rows" in r(**router_args(193))
    assert "0 rows" in r(**router_args(0))
    assert "x must be" in r(meta(4, 7168).t().contiguous().t(), w, meta(896, dtype=f32))
    assert "x must be" in r(meta(4, 7168, dtype=f32), w, meta(896, dtype=f32))
    good = (meta(4, 16, dtype=f32), meta(4, 16, dtype=i32))
    assert r(**router_args(4), out=good) is None
    assert "out must be" in r(**router_args(4), out=(meta(4, 16, dtype=f32), meta(4, 16, dtype=f32)))
    assert "out must be" in r(**router_args(4), out=(meta(4, 16, dtype=f32), meta(5, 16, dtype=i32)))
    aux = E.router_aux(w, 4, torch.device("meta"), zero_moe_buf=False)
    assert "aux is not" in r(**router_args(8), aux=aux)
    assert "aux is not" in r(**router_args(4), aux={k: v for k, v in aux.items() if k != "m_indices"})
    assert "moe_buf" in r(**router_args(4), aux={**aux, "moe_buf": meta(5, 3584)})
    assert "moe_buf" in r(**router_args(4), aux={**aux, "moe_buf": meta(4, 2048)})


# experts


def weights(e=896):
    return E.moe_weights(meta(e, 768, 1792, dtype=u8), meta(e * 768, 112, dtype=u8), meta(e, 3584, 192, dtype=u8),
                         meta(e * 3584, 16, dtype=u8))


def sort_of(m, w):
    return E.moe_sort_from_aux(E.router_aux(meta(896, 7168), m, torch.device("meta")), weights=w)


def test_experts_route_by_w13(gfx950):
    w = weights()
    assert experts._ROUTES.package(tuple(w.w13.shape)).__name__ == experts._ROUTES.hosts[(896, 768, 1792)]
    assert E.moe_weights_blocker(w.w13, w.s13, w.w2, w.s2) is None
    for m in (1, 192):
        aux = E.router_aux(meta(896, 7168), m, torch.device("meta"))
        assert E.moe_sort_from_aux_blocker(aux, weights=w) is None
        args = (meta(m, 1792, dtype=u8), meta(m, 112, dtype=u8), w, E.moe_sort_from_aux(aux, weights=w))
        assert E.mxfp4_moe_experts_blocker(*args) is None
        assert E.mxfp4_moe_experts_blocker(*args, out=meta(m, 3584)) is None
        assert E.quantize_moe_activations_blocker(meta(m, 3584), weights=w) is None
    host = importlib.import_module(experts._ROUTES.hosts[(896, 768, 1792)])
    assert E.moe_sort_capacity(12, weights=w) == host.sort_capacity(12)


def test_experts_off_table(gfx950):
    args = (meta(128, 768, 1792, dtype=u8), meta(128 * 768, 112, dtype=u8), meta(128, 3584, 192, dtype=u8),
            meta(128 * 3584, 16, dtype=u8))
    reason = E.moe_weights_blocker(*args)
    assert "no kernel for E=128, 2I=768, H/2=1792" in reason and "(896, 768, 1792)" in reason
    with pytest.raises(ValueError) as e:
        E.moe_weights(*args)
    assert str(e.value) == reason
    fake = SimpleNamespace(w13=meta(128, 768, 1792, dtype=u8))
    reason = E.mxfp4_moe_experts_blocker(meta(4, 1792, dtype=u8), meta(4, 112, dtype=u8), fake, None)
    assert "no kernel for" in reason
    with pytest.raises(ValueError, match="no kernel for"):
        E.mxfp4_moe_experts(meta(4, 1792, dtype=u8), meta(4, 112, dtype=u8), fake, None)


def test_experts_reasons(gfx950):
    w = weights()
    assert "operand layout" in E.moe_weights_blocker(w.w13, w.s13, w.w2, meta(896 * 3584, 12, dtype=u8))
    assert "operand layout" in E.moe_weights_blocker(w.w13, meta(896 * 768, 56, dtype=bf), w.w2, w.s2)
    r = E.mxfp4_moe_experts_blocker
    sort = sort_of(12, w)
    codes, scales = meta(12, 1792, dtype=u8), meta(12, 112, dtype=u8)
    fake = SimpleNamespace(w13=w.w13, s13=w.s13, w2=w.w2, s2=w.s2)
    assert "not built by moe_weights" in r(codes, scales, fake, sort)
    assert "sort not built" in r(codes, scales, w, SimpleNamespace(**vars(sort)))
    assert "193 rows" in r(meta(193, 1792, dtype=u8), meta(193, 112, dtype=u8), w, sort_of(192, w))
    assert "x_codes / x_scales" in r(codes, meta(12, 56, dtype=u8), w, sort)
    assert "x_codes / x_scales" in r(meta(12, 1792, dtype=bf), scales, w, sort)
    assert "smaller than the sort" in r(meta(40, 1792, dtype=u8), meta(40, 112, dtype=u8), w, sort)
    assert "out must be" in r(codes, scales, w, sort, meta(12, 3584, dtype=f32))
    assert r(codes, scales, w, sort, meta(12, 3584, dtype=f32), reduce=False) is None
    q = E.quantize_moe_activations_blocker
    assert "x must be" in q(meta(12, 7168), weights=w)
    assert "codes / scales" in q(meta(12, 3584), meta(12, 1792, dtype=u8), meta(6, 112, dtype=u8), weights=w)
    assert "together" in q(meta(12, 3584), meta(12, 1792, dtype=u8), weights=w)
    assert q(meta(12, 3584), meta(12, 1792, dtype=u8), meta(12, 112, dtype=u8), weights=w) is None
    assert "together" in q(meta(12, 3584), None, meta(12, 112, dtype=u8), weights=w)
    assert "weights' device" in q(torch.empty(12, 3584, dtype=bf), weights=w)


def test_aux_hand_off_mismatch(gfx950):
    w = weights()
    aux = E.router_aux(meta(896, 7168), 12, torch.device("meta"))
    s = E.moe_sort_from_aux_blocker
    assert "not a top-16 sort" in s({**aux, "reverse_sorted": meta(12 * 8, dtype=i32)}, weights=w)
    assert "not a top-16 sort" in s({**aux, "moe_buf": meta(12, 7168)}, weights=w)
    assert "not a top-16 sort" in s({**aux, "sorted_token_ids": meta(64, dtype=i32)}, weights=w)
    assert "lacks the sort tensors" in s({k: v for k, v in aux.items() if k != "num_valid_ids"}, weights=w)
    assert "weights' device" in s({**aux, "reverse_sorted": torch.empty(12 * 16, dtype=i32)}, weights=w)
    no_buf = E.router_aux(meta(896, 7168), 12, torch.device("meta"), zero_moe_buf=False)
    assert s(no_buf, weights=w) is None


# latent tail


def fake_comm(module, cls_name, **attrs):
    comm = object.__new__(getattr(importlib.import_module(module), cls_name))
    comm.__dict__.update(bufs=[1], device=torch.device("meta"), **attrs)
    return comm


def tail_args(m=12, n=896, k=3584):
    return dict(partial=meta(m, k), norm_weight=meta(k), up_weight=meta(n, k), shared=meta(m, n))


@pytest.fixture
def comm():
    return fake_comm(latent_tail._ROUTES.hosts[(896, 3584, 8)], "LatentTailComm")


def test_latent_tail_routes_by_comm(gfx950, comm, monkeypatch):
    for m in (1, 256):
        assert E.allreduce_latent_rmsnorm_up_proj_blocker(comm, **tail_args(m)) is None
    for m in (1, 64):
        assert E.allreduce_slots_rmsnorm_up_proj_blocker(
            comm, meta(200, 3584), meta(m * 16, dtype=i32), meta(3584), meta(896, 3584), meta(m, 896)) is None
    shared = meta(12, 7168)[:, 896:1792]
    assert E.allreduce_latent_rmsnorm_up_proj_blocker(comm, **{**tail_args(), "shared": shared}) is None
    monkeypatch.setattr(latent_tail.dist, "get_world_size", lambda group: 8)
    assert E.latent_tail_comm_blocker(0, None, "meta", up_rows=896, latent=3584) is None
    monkeypatch.setattr(latent_tail.dist, "get_world_size", lambda group: 4)
    reason = E.latent_tail_comm_blocker(0, None, "meta", up_rows=896, latent=3584)
    assert "no kernel for up_rows=896, latent=3584, world=4" in reason
    with pytest.raises(ValueError) as e:
        E.latent_tail_comm(0, None, "meta", up_rows=896, latent=3584)
    assert str(e.value) == reason


def test_latent_tail_reasons(gfx950, comm):
    r = E.allreduce_latent_rmsnorm_up_proj_blocker
    assert "communicator built for up_rows=896, latent=3584, world=8, call is up_rows=1024, latent=3584" in r(
        comm, **tail_args(n=1024))
    assert "257 rows" in r(comm, **tail_args(257))
    assert "0 rows" in r(comm, **tail_args(0))
    assert "communicator built for" in r(comm, **{**tail_args(), "partial": meta(12, 7168)})
    assert "bf16" in r(comm, **{**tail_args(), "norm_weight": meta(3584, dtype=f32)})
    assert "row strides" in r(comm, **{**tail_args(), "partial": meta(12, 3586)[:, :3584]})
    assert "row strides" in r(comm, **{**tail_args(), "shared": meta(12, 898)[:, :896]})
    assert "norm weight" in r(comm, **{**tail_args(), "norm_weight": meta(7168)})
    assert "shared and out" in r(comm, **tail_args(), out=meta(13, 896))
    s = E.allreduce_slots_rmsnorm_up_proj_blocker
    args = (meta(3584), meta(896, 3584), meta(65, 896))
    assert "65 rows (the kernel takes 1..64)" in s(comm, meta(2000, 3584), meta(65 * 16, dtype=i32), *args)
    assert "reverse_sorted" in s(comm, meta(200, 3584), meta(12 * 16, dtype=torch.int64), *args[:2], meta(12, 896))
    assert "needs reverse_sorted" in s(comm, meta(200, 3584), None, *args)
    other = SimpleNamespace(bufs=[1], device=torch.device("meta"))
    reason = r(other, **tail_args())
    assert "no kernel for a SimpleNamespace" in reason
    with pytest.raises(ValueError) as e:
        E.allreduce_latent_rmsnorm_up_proj(other, **tail_args())
    assert str(e.value) == reason
    comm.bufs = []
    assert "closed" in r(comm, **tail_args())


# layer tail

LAUNCH_KW = ("core", "w_o", "prefix", "blocks", "ares_nw", "ares_qk", "in_nw", "w_gate", "bias", "w_ld", "w_sgu",
             "w_sd", "w13", "w13s", "w2", "w2s", "ln_w", "w_up", "out", "layer")


def launch_kw(s=12, hidden=7168, k=1536, e=896):
    return dict(core=meta(s, k), w_o=meta(hidden, k), prefix=meta(s, hidden), blocks=meta(s, 8, hidden),
                ares_nw=meta(hidden), ares_qk=meta(hidden), in_nw=meta(hidden), w_gate=meta(e, hidden),
                bias=meta(e, dtype=f32), w_ld=meta(3584, hidden), ln_w=meta(3584), w_sgu=meta(1536, hidden),
                w_sd=meta(hidden, 768), w_up=meta(896, 3584), w13=meta(e, 768, 1792, dtype=u8),
                w2=meta(e, 3584, 192, dtype=u8), w13s=meta(1, dtype=u8), w2s=meta(1, dtype=u8), out=meta(s, hidden),
                layer=3)


@pytest.fixture
def runner():
    return fake_comm(layer_tail._ROUTES.hosts[(7168, 1536, 896, 8)], "LayerTailRunner", scratch=meta(1, dtype=u8))


def test_layer_tail_routes_by_runner(gfx950, runner, monkeypatch):
    r = E.layer_tail_launch_blocker
    for s in (4, 16):
        for reset in (False, True):
            assert r(runner, {"tokens": s, "nblocks": 4 + reset, "reset": reset, "m1b_qlen": 0}, **launch_kw(s)) is None
    _, fields = layer_tail._runner_host(type(runner))
    assert {"tokens", "nblocks", "reset", "eps"} <= fields and "m1b_qlen" not in fields
    monkeypatch.setattr(layer_tail.dist, "get_world_size", lambda group: 8)
    assert E.layer_tail_runner_blocker("meta", None, 0, hidden=7168, o_proj_k=1536, experts=896) is None
    reason = E.layer_tail_runner_blocker("meta", None, 0, hidden=6144, o_proj_k=1536, experts=896)
    assert "no kernel for hidden=6144, o_proj_k=1536, experts=896, world=8" in reason
    with pytest.raises(ValueError) as e:
        E.layer_tail_runner("meta", None, 0, hidden=6144, o_proj_k=1536, experts=896)
    assert str(e.value) == reason


def test_layer_tail_reasons(gfx950, runner):
    r = E.layer_tail_launch_blocker
    build = {"tokens": 12, "nblocks": 4}
    assert "runner built for hidden=7168, o_proj_k=1536, experts=896, world=8, call is " in r(
        runner, build, **launch_kw(e=384))
    assert "5 tokens" in r(runner, {"tokens": 5, "nblocks": 4}, **launch_kw(5))
    assert "12 tokens, core has 16 rows" in r(runner, build, **{**launch_kw(), "core": meta(16, 1536)})
    assert "9 block sources" in r(runner, {"tokens": 12, "nblocks": 9}, **launch_kw())
    assert "needs tokens and nblocks" in r(runner, {"tokens": 12}, **launch_kw())
    assert "prefix and out" in r(runner, build, **{**launch_kw(), "out": meta(16, 7168)})
    assert "hashable" in r(runner, {**build, "extra": [1]}, **launch_kw())
    assert "must be int" in r(runner, {"tokens": 12.0, "nblocks": 4}, **launch_kw())
    assert "does not fuse ['dsa', 'm1b_qlen']" in r(runner, {**build, "m1b_qlen": 2, "dsa": 2}, **launch_kw())
    assert "does not fuse ['m1c']" in r(runner, {**build, "m1c": True}, **launch_kw())
    stock_off = {"m1b_qlen": 0, "m1b_ranks": 1, "m1c": False, "dsa": 0, "dsa_tile": 128, "uv_bf16": False}
    assert r(runner, {**build, **stock_off}, **launch_kw()) is None
    assert "contiguous bf16" in r(runner, build, **{**launch_kw(), "core": meta(12, 1536, dtype=f32)})
    assert "contiguous bf16" in r(runner, build, **{**launch_kw(), "out": meta(12, 7168, dtype=f32)})
    assert "contiguous bf16" in r(runner, build, **{**launch_kw(), "prefix": meta(12, 7170)[:, :7168]})
    assert "blocks must be" in r(runner, build, **{**launch_kw(), "blocks": meta(12, 2, 7168)})
    assert "blocks must be" in r(runner, build, **{**launch_kw(), "blocks": meta(12, 8, 7168, dtype=f32)})
    # the weights' other dims: a model with the same hidden, o_proj K and experts but another intermediate size
    assert "w2 (896, 3584, 2048) (the kernel takes (896, 3584, 192))" in r(
        runner, build, **{**launch_kw(), "w2": meta(896, 3584, 2048, dtype=u8)})
    assert "w13 (896, 8192, 1792)" in r(runner, build, **{**launch_kw(), "w13": meta(896, 8192, 1792, dtype=u8)})
    assert "w_gate must be contiguous" in r(runner, build, **{**launch_kw(), "w_gate": meta(896, 7168, dtype=f32)})
    assert "bias must be contiguous" in r(runner, build, **{**launch_kw(), "bias": meta(896)})
    other = SimpleNamespace()
    reason = r(other, build, **launch_kw())
    assert "no kernel for a SimpleNamespace" in reason
    with pytest.raises(ValueError) as e:
        E.layer_tail_launch(other, build, **launch_kw())
    assert str(e.value) == reason


def test_swap_stream_workspaces_moves_keys_only(monkeypatch):
    """Lending exchanges the two streams' entries of one device in every loaded host's dicts, objects unchanged, and
    twice is the identity; another device's and another stream's entries stay."""
    work = {(0, 1, 4, ()): "a4", (0, 2, 4, ()): "b4", (0, 1, 8, ()): "a8", (1, 1, 4, ()): "d1", (0, 3, 4, ()): "c4"}
    aux = {(0, 1, 16): "x", (0, 2, 32): "y"}
    slots = {(0, 1): "s"}
    hosts = {router._ROUTES.hosts[(896, 7168, 16)]: SimpleNamespace(_WORK=work, _AUX_WORK=aux),
             experts._ROUTES.hosts[(896, 768, 1792)]: SimpleNamespace(_WORK=slots)}
    for name, host in hosts.items():
        monkeypatch.setitem(sys.modules, name, host)
    before = (dict(work), dict(aux), dict(slots))
    E.swap_stream_workspaces(0, 1, 2)
    assert work == {(0, 2, 4, ()): "a4", (0, 1, 4, ()): "b4", (0, 2, 8, ()): "a8", (1, 1, 4, ()): "d1",
                    (0, 3, 4, ()): "c4"}
    assert aux == {(0, 2, 16): "x", (0, 1, 32): "y"} and slots == {(0, 2): "s"}
    E.swap_stream_workspaces(0, 2, 1)
    assert (work, aux, slots) == before


def test_prepare_returns_the_object_or_the_reason(gfx950, monkeypatch):
    args = (meta(128, 768, 1792, dtype=u8), meta(128 * 768, 112, dtype=u8), meta(128, 3584, 192, dtype=u8),
            meta(128 * 3584, 16, dtype=u8))
    assert E.prepare_moe_weights(*args) == E.moe_weights_blocker(*args)
    w = weights()
    assert type(E.prepare_moe_weights(w.w13, w.s13, w.w2, w.s2)) is type(w)
    monkeypatch.setattr(latent_tail.dist, "get_world_size", lambda group: 4)
    assert "world=4" in E.prepare_latent_tail_comm(0, None, "meta", up_rows=896, latent=3584)
    monkeypatch.setattr(layer_tail.dist, "get_world_size", lambda group: 8)
    reason = E.prepare_layer_tail_runner("meta", None, 0, hidden=6144, o_proj_k=1536, experts=896)
    assert reason == E.layer_tail_runner_blocker("meta", None, 0, hidden=6144, o_proj_k=1536, experts=896)


def test_off_rank_operand_keys_print_as_is(gfx950):
    reason = E.router_sigmoid_bias_topk_weight_blocker(meta(2, 896, 7168))
    assert "no kernel for (2, 896, 7168, 16)" in reason, reason


OLD_NAMES = ["router_sigmoid_bias_topk", "router_sigmoid_bias_topk_weight", "moe_weights", "moe_sort_from_aux",
             "quantize_moe_activations", "mxfp4_moe_experts", "latent_tail_comm", "allreduce_latent_rmsnorm_up_proj",
             "allreduce_slots_rmsnorm_up_proj", "layer_tail_runner", "layer_tail_launch"]


@pytest.mark.parametrize("op", OLD_NAMES)
def test_old_names_alias_the_blockers(op):
    old = op + ("_reason" if op.endswith("_weight") else "_unsupported_reason")
    alias = getattr(importlib.import_module("amd_kernels.moe.latent_moe_common"), old)
    assert alias.renamed_to == op + "_blocker" and alias.__wrapped__ is getattr(E, op + "_blocker")
