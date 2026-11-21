# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of ``mla_decode_common`` on meta tensors (no GPU): every table row imports its shape package,
each covered key routes to it, an off-table shape gives a "no kernel" reason and the op raises it, the reasons take
the op's arguments, and importing the entry imports no shape package."""

import importlib
import inspect
import json
import socket
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from amd_kernels.attention import mla_decode_common as E
from amd_kernels.attention.mla_decode_common import combine, dense, prologue, query, sparse

TABLES = [
    (prologue._SHAPE_HOSTS, ("IndexSide", "qkv_a_norm_kv_insert", "supports_qkv_a_norm_kv_insert")),
    (prologue._INDEX_HOSTS, ("indexer_prep", "supports_indexer_prep")),
    (query._SHAPE_HOSTS, ("QueryGatherComm", "pack_q_b", "pack_w_uk", "q_b_allgather_wuk_all_heads")),
    (sparse._SHAPE_HOSTS, ("fp8_sparse_mla_decode", "supports_fp8_sparse_mla_decode")),
    (combine._SHAPE_HOSTS, ("DcpVupCombine", "dcp8_lse_combine_vup", "prepare_w_uv")),
    (dense._SHAPE_HOSTS, ("fp8_dense_mla_decode", "supports_fp8_dense_mla_decode")),
]
SHAPES = [
    (E.qkv_a_norm_kv_insert_shapes, prologue._SHAPE_HOSTS),
    (E.indexer_prep_shapes, prologue._INDEX_HOSTS),
    (E.q_bproj_wuk_gather_shapes, query._SHAPE_HOSTS),
    (E.fp8_sparse_mla_decode_shapes, sparse._SHAPE_HOSTS),
    (E.lse_combine_vup_shapes, combine._SHAPE_HOSTS),
    (E.fp8_dense_mla_decode_shapes, dense._SHAPE_HOSTS),
]
PAIRS = [
    (E.qkv_a_norm_kv_insert, E.qkv_a_norm_kv_insert_unsupported_reason),
    (E.indexer_prep, E.indexer_prep_unsupported_reason),
    (E.q_gather_comm, E.q_gather_comm_unsupported_reason),
    (E.q_bproj_wuk_gather, E.q_bproj_wuk_gather_unsupported_reason),
    (E.fp8_sparse_mla_decode, E.fp8_sparse_mla_decode_unsupported_reason),
    (E.lse_combine_comm, E.lse_combine_comm_unsupported_reason),
    (E.lse_combine_vup, E.lse_combine_vup_unsupported_reason),
    (E.fp8_dense_mla_decode, E.fp8_dense_mla_decode_unsupported_reason),
]
MODULES = (prologue, query, sparse, combine, dense)


def meta(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "routed"


@pytest.fixture
def gpu_ok(monkeypatch):
    for m in MODULES:
        monkeypatch.setattr(m, "available", lambda: True)


@pytest.fixture(scope="module")
def group():
    """A one-rank gloo group: the factories read the world size from it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    yield dist.group.WORLD
    dist.destroy_process_group()


@pytest.mark.parametrize("table,names", TABLES, ids=lambda x: str(x)[:40])
def test_every_row_imports_and_exports(table, names):
    for name in table.values():
        mod = importlib.import_module(name)
        assert all(hasattr(mod, n) for n in names), (name, [n for n in names if not hasattr(mod, n)])


@pytest.mark.parametrize("shapes,table", SHAPES, ids=lambda x: getattr(x, "__name__", ""))
def test_shapes_are_the_table_keys(shapes, table):
    assert shapes() == frozenset(table)


@pytest.mark.parametrize("op,reason", PAIRS, ids=lambda f: f.__name__)
def test_reason_takes_the_op_arguments(op, reason):
    assert list(inspect.signature(op).parameters.values()) == list(inspect.signature(reason).parameters.values())


def test_import_loads_no_shape_package():
    code = (
        "import sys, json; import amd_kernels.attention.mla_decode_common as m; "
        "print(json.dumps(sorted(k for k in sys.modules if k.startswith('amd_kernels.attention.') "
        "and not k.startswith('amd_kernels.attention.mla_decode_common'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def insert_args(n, q_lora=1536, kv_lora=512, index=True):
    side = dict(ln_weight=meta(128), ln_bias=meta(128), cache=meta(4, 64, 68, dtype=torch.uint8),
                slot_mapping=meta(n, dtype=torch.int64), weights=meta(n, 32, dtype=torch.float32), weight_scale=0.5)
    return (meta(n, 2272), meta(q_lora), meta(kv_lora), 1e-6, meta(n, q_lora), meta(n, kv_lora),
            meta(4, 64, 656, dtype=torch.uint8), meta(n, dtype=torch.int64), side if index else None)


def prep_args(n, heads=32, d=128):
    return (meta(n, heads, d), None, None, None, None, 1e-6, None, meta(4, 64, 68, dtype=torch.uint8),
            meta(n, 2048, dtype=torch.int32), 0.5, meta(n, heads, d // 2, dtype=torch.uint8),
            meta(n, 1, 4, 16, 4, dtype=torch.uint8), None)


def sparse_args(n, heads=96, q8=False):
    q = None if q8 else meta(n, heads, 576)
    kw = {"q8": meta(n, heads, 1168, dtype=torch.uint8)} if q8 else {}
    return (q, meta(64, 656, dtype=torch.uint8), meta(n + 1, dtype=torch.int32), meta(9, dtype=torch.int32), 0.1), kw


def dense_args(n, heads=96):
    fp8 = torch.float8_e4m3fn
    return (meta(n, heads, 576, dtype=fp8), meta(64, 576, dtype=fp8), meta(n // 4 + 1, dtype=torch.int32),
            meta(9, dtype=torch.int32), meta(n // 4 + 1, dtype=torch.int32), meta(1, dtype=torch.float32),
            meta(1, dtype=torch.float32), 0.1, 4)


def comm_of(table, cls, mode=None):
    obj = object.__new__(getattr(importlib.import_module(next(iter(table.values()))), cls))
    if mode is not None:
        obj.mode = mode
    obj.max_tokens = 192
    return obj


def query_args(comm, n, q_lora=1536):
    return (comm, meta(n, q_lora), meta(2304, 1536), meta(96, 8, 4, 4, 4, 16, 8)), {"out": meta(n, 96, 576)}


def combine_args(comm, n, heads=96):
    return (comm, meta(n, 96, 512), meta(n, 96, dtype=torch.float32), meta(heads, 128, 512), meta(n, 12, 128),
            meta(5, dtype=torch.int32), meta(6, dtype=torch.int32), meta(n, 1536)), {}


def test_routes_each_covered_key(monkeypatch):
    for n in (1, 192):
        for index in (True, False):
            pkg = prologue._shape_host((1536, 512))
            rec = Recorder()
            monkeypatch.setattr(pkg, "qkv_a_norm_kv_insert", rec)
            args = insert_args(n, index=index)
            E.qkv_a_norm_kv_insert(*args)
            (got, _), = rec.calls
            assert got[:8] == args[:8]
            assert (got[8] is None) if not index else (got[8] == pkg.IndexSide(**args[8]))
        pkg = prologue._index_host((32, 128))
        monkeypatch.setattr(pkg, "indexer_prep", rec := Recorder())
        E.indexer_prep(*prep_args(n))
        assert len(rec.calls) == 1
        for q8 in (False, True):
            pkg = sparse._shape_host((96, 576, 656))
            monkeypatch.setattr(pkg, "fp8_sparse_mla_decode", rec := Recorder())
            args, kw = sparse_args(n, q8=q8)
            assert E.fp8_sparse_mla_decode(*args, **kw) == "routed" and rec.calls[0][0][-1] is kw.get("q8")
        pkg = dense._shape_host((96, 576))
        monkeypatch.setattr(pkg, "fp8_dense_mla_decode", rec := Recorder())
        assert E.fp8_dense_mla_decode(*dense_args(4 * n)) == "routed"
        pkg = query._shape_host((1536, 2304, 96, 8))
        monkeypatch.setattr(pkg, "q_b_allgather_wuk_all_heads", rec := Recorder())
        args, kw = query_args(comm_of(query._SHAPE_HOSTS, "QueryGatherComm", "q"), n)
        assert E.q_bproj_wuk_gather(*args, **kw) == "routed" and rec.calls[0] == (args, {**kw, "out_q8": None})
        pkg = combine._shape_host((96, 128, 512, 8))
        monkeypatch.setattr(pkg, "dcp8_lse_combine_vup", rec := Recorder())
        args, _ = combine_args(comm_of(combine._SHAPE_HOSTS, "DcpVupCombine"), n)
        assert E.lse_combine_vup(*args) == "routed" and rec.calls[0] == (args, {})


def test_kernel_lookup_returns_the_routed_entry():
    pkg = prologue._shape_host((1536, 512))
    run = E.qkv_a_norm_kv_insert_kernel(*insert_args(2))
    assert run.entry.name == pkg.qkv_a_norm_kv_insert.entry.name
    inputs = run.entry.inputs(run.entry.mode(*insert_args(2)), 1, *insert_args(2))
    assert isinstance(inputs[0][8], pkg.IndexSide)  # the warm-up takes the dict side
    assert E.indexer_prep_kernel(*prep_args(2)) is prologue._index_host((32, 128)).indexer_prep
    args, kw = sparse_args(2)
    assert E.fp8_sparse_mla_decode_kernel(*args, **kw) is sparse._shape_host((96, 576, 656)).fp8_sparse_mla_decode
    args, kw = query_args(comm_of(query._SHAPE_HOSTS, "QueryGatherComm", "q"), 2)
    routed = query._shape_host((1536, 2304, 96, 8)).q_b_allgather_wuk_all_heads
    assert E.q_bproj_wuk_gather_kernel(*args, **kw) is routed
    args, _ = combine_args(comm_of(combine._SHAPE_HOSTS, "DcpVupCombine"), 2)
    assert E.lse_combine_vup_kernel(*args) is combine._shape_host((96, 128, 512, 8)).dcp8_lse_combine_vup


def off_table_cases():
    sa, skw = sparse_args(4, heads=64)
    qa, qkw = sparse_args(4, heads=64, q8=True)
    return [
        ("qkv_a_norm_kv_insert", insert_args(4, q_lora=1024), {}, "q_lora=1024, kv_lora=512"),
        ("indexer_prep", prep_args(4, heads=16), {}, "heads=16, d=128"),
        ("fp8_sparse_mla_decode", sa, skw, "heads=64, d_qk=576, record=656"),
        ("fp8_sparse_mla_decode", qa, qkw, "heads=64, d_qk=q8, record=656"),
        ("fp8_dense_mla_decode", dense_args(4, heads=128), {}, "heads=128, d_qk=576"),
        ("q_bproj_wuk_gather", query_args(object(), 4)[0], {"out": None}, "communicator object"),
        ("lse_combine_vup", combine_args(object(), 4)[0], {}, "communicator object"),
    ]


@pytest.mark.parametrize("name,args,kw,dims", off_table_cases(), ids=lambda x: x if isinstance(x, str) else "")
def test_off_table_gives_the_reason_and_the_op_and_kernel_raise_it(name, args, kw, dims):
    why = getattr(E, name + "_unsupported_reason")(*args, **kw)
    assert why.startswith(f"{name}: no kernel for {dims}") and "covered: (" in why, why
    for fn in (getattr(E, name), getattr(E, name + "_kernel", None)):
        if fn is not None:
            with pytest.raises(ValueError) as e:
                fn(*args, **kw)
            assert str(e.value) == why


def test_factories_read_the_world_from_the_group(group):
    why = E.q_gather_comm_unsupported_reason(0, group, "cpu", "q", q_lora=1536, q_b_rows=2304, heads=96)
    assert why.startswith("q_gather_comm: no kernel for q_lora=1536, q_b_rows=2304, heads=96, world=1"), why
    with pytest.raises(ValueError, match="world=1"):
        E.q_gather_comm(0, group, "cpu", "q", q_lora=1536, q_b_rows=2304, heads=96)
    why = E.lse_combine_comm_unsupported_reason(0, group, "cpu", heads=96, v_head=128, kv_lora=512)
    assert why.startswith("lse_combine_comm: no kernel for heads=96, v_head=128, kv_lora=512, world=1"), why
    with pytest.raises(ValueError, match="world=1"):
        E.lse_combine_comm(0, group, "cpu", heads=96, v_head=128, kv_lora=512)


def test_factory_reasons_without_a_process_group(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    dims = dict(q_lora=1536, q_b_rows=2304, heads=96)
    assert "needs an initialized" in E.q_gather_comm_unsupported_reason(0, None, "cpu", "q", **dims)
    dims = dict(heads=96, v_head=128, kv_lora=512)
    assert "needs an initialized" in E.lse_combine_comm_unsupported_reason(0, None, "cpu", **dims)


def test_factory_reasons_past_the_key(monkeypatch, gpu_ok):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)
    ok = dict(q_lora=1536, q_b_rows=2304, heads=96)
    assert E.q_gather_comm_unsupported_reason(0, None, "cpu", "q", **ok) is None
    assert "mode 'x'" in E.q_gather_comm_unsupported_reason(0, None, "cpu", "x", **ok)
    ok = dict(heads=96, v_head=128, kv_lora=512)
    assert E.lse_combine_comm_unsupported_reason(0, None, "cpu", **ok) is None
    assert "max_tokens 193 (the kernel takes 1..192)" in E.lse_combine_comm_unsupported_reason(
        0, None, "cpu", **ok, max_tokens=193)
    monkeypatch.setattr(combine, "available", lambda: False)
    assert E.lse_combine_comm_unsupported_reason(0, None, "cpu", **ok) == "lse_combine_comm: needs gfx950 with FlyDSL"


def test_needs_gfx950(monkeypatch):
    for m in MODULES:
        monkeypatch.setattr(m, "available", lambda: False)
    args, kw = sparse_args(4)
    cases = [
        (E.qkv_a_norm_kv_insert_unsupported_reason, insert_args(4), {}),
        (E.indexer_prep_unsupported_reason, prep_args(4), {}),
        (E.fp8_sparse_mla_decode_unsupported_reason, args, kw),
        (E.fp8_dense_mla_decode_unsupported_reason, dense_args(4), {}),
        (E.q_bproj_wuk_gather_unsupported_reason, *query_args(comm_of(query._SHAPE_HOSTS, "QueryGatherComm", "q"), 4)),
        (E.lse_combine_vup_unsupported_reason, *combine_args(comm_of(combine._SHAPE_HOSTS, "DcpVupCombine"), 4)),
    ]
    for reason, a, k in cases:
        assert reason(*a, **k).endswith(": needs gfx950 with FlyDSL"), reason.__name__


def mutate(args, kw, i, value):
    args = list(args)
    if isinstance(i, str):
        kw = {**kw, i: value}
    else:
        args[i] = value
    return tuple(args), kw


QUERY_BRANCHES = [
    (None, None, None),
    (0, comm_of(query._SHAPE_HOSTS, "QueryGatherComm", "latent"), "communicator mode 'latent'"),
    (1, meta(4, 1024), "communicator built for q_lora=1536, q_b_rows=2304, heads=96, world=8, call is q_lora=1024"),
    (2, meta(4608, 1536), "call is q_lora=1536, q_b_rows=4608"),
    (1, meta(0, 1536), "0 rows (the kernel takes 1..256)"),
    (1, meta(257, 1536), "257 rows (the kernel takes 1..256)"),
    (1, meta(4, 1536, dtype=torch.float16), "operand layout (q_a bf16"),
    (1, meta(4, 1540)[:, :1536], "operand layout (q_a bf16"),
    (2, meta(1536, 2304).t(), "operand layout (w_qb, w_uk96"),
    (3, meta(96, 128, 256), "operand layout (w_qb, w_uk96"),
    ("out", meta(4, 96, 512), "operand layout (out bf16"),
    ("out_q8", meta(4, 96, 1152, dtype=torch.uint8), "operand layout (out_q8 uint8"),
]


@pytest.mark.parametrize("i,value,text", QUERY_BRANCHES)
def test_query_reasons(gpu_ok, i, value, text):
    args, kw = query_args(comm_of(query._SHAPE_HOSTS, "QueryGatherComm", "q"), 4)
    if i is not None:
        args, kw = mutate(args, kw, i, value)
    why = E.q_bproj_wuk_gather_unsupported_reason(*args, **kw)
    assert (why is None) if text is None else (text in why), why


COMBINE_BRANCHES = [
    (None, None, None),
    (3, meta(64, 128, 512), "communicator built for heads=96, v_head=128, kv_lora=512, world=8, call is heads=64"),
    (1, meta(4, 96 * 512), "operand layout (attn_out [rows"),
    (1, meta(0, 96, 512), "0 rows (the kernel takes 1..192)"),
    (1, meta(193, 96, 512), "193 rows (the kernel takes 1..192)"),
    (1, meta(4, 95, 512), "operand layout (attn_out bf16"),
    (1, meta(4, 96, 576)[:, :, :512], None),
    (1, meta(4, 96, 1024)[:, :, ::2], "operand layout (attn_out bf16"),
    (2, meta(4, 96, dtype=torch.bfloat16), "operand layout (lse fp32"),
    (2, meta(4, 128, dtype=torch.float32), None),
    (3, meta(96, 512, 128).transpose(1, 2), "operand layout (w_uv_t bf16 contiguous"),
    (4, meta(4, 96, 128), "operand layout (out bf16 contiguous [rows, 12, 128])"),
    (7, meta(4, 1024), "operand layout (gate bf16"),
    (7, None, None),
    (5, meta(5, dtype=torch.int64), "seq_lens and query_start_loc int32"),
    (6, None, "seq_lens and query_start_loc int32"),
]


@pytest.mark.parametrize("i,value,text", COMBINE_BRANCHES)
def test_combine_reasons(gpu_ok, i, value, text):
    args, kw = combine_args(comm_of(combine._SHAPE_HOSTS, "DcpVupCombine"), 4)
    if i is not None:
        args, kw = mutate(args, kw, i, value)
    why = E.lse_combine_vup_unsupported_reason(*args, **kw)
    assert (why is None) if text is None else (text in why), why


def test_combine_rows_capped_by_the_communicator(gpu_ok):
    comm = comm_of(combine._SHAPE_HOSTS, "DcpVupCombine")
    comm.max_tokens = 64
    args, _ = combine_args(comm, 65)
    assert "65 rows (the kernel takes 1..64)" in E.lse_combine_vup_unsupported_reason(*args)


def test_weight_packers_route_by_shape():
    assert E.pack_q_b(meta(2304, 1536)).shape == (2304, 1536)
    assert E.pack_w_uk(meta(96, 128, 512)).shape[0] == 96
    assert E.prepare_w_uv(meta(96, 512, 128)).shape == (96, 128, 512)
    off = ((E.pack_q_b, meta(4608, 1536)), (E.pack_w_uk, meta(64, 128, 512)), (E.prepare_w_uv, meta(64, 512, 128)))
    for fn, w in off:
        with pytest.raises(ValueError, match="no kernel for"):
            fn(w)


def test_shape_reasons_before_the_package(gpu_ok):
    args = insert_args(4)
    flat = mutate(args, {}, 4, meta(4 * 1536))[0]
    assert "operand layout (q_out, kv_out" in E.qkv_a_norm_kv_insert_unsupported_reason(*flat)
    flat = mutate(prep_args(4), {}, 0, meta(4, 4096))[0]
    assert "operand layout (q [rows" in E.indexer_prep_unsupported_reason(*flat)
    d = dense_args(8)
    assert "query_len 5 (the kernel takes 1..4)" in E.fp8_dense_mla_decode_unsupported_reason(*mutate(d, {}, 8, 5)[0])
    assert "operand layout (7 rows" in E.fp8_dense_mla_decode_unsupported_reason(*mutate(d, {}, 0, meta(7, 96, 576))[0])
    many = mutate(d, {}, 0, meta(256, 96, 576, dtype=torch.float8_e4m3fn))[0]
    assert "64 requests (the kernel takes 0..63)" in E.fp8_dense_mla_decode_unsupported_reason(*many)
    assert "cp_rank 8 of cp_world 8" in E.fp8_dense_mla_decode_unsupported_reason(*d, cp_rank=8)
    sa, skw = sparse_args(4)
    assert "operand layout (q or q8" in E.fp8_sparse_mla_decode_unsupported_reason(None, *sa[1:])
