# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the NVFP4 sparse attention entries on the host (meta tensors, no GPU)."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.attention import nvfp4_sparse_attention_common as E
from amd_kernels.attention.nvfp4_sparse_attention_common import layer

BF, F32, I32, U8, E4 = torch.bfloat16, torch.float32, torch.int32, torch.uint8, torch.float8_e4m3fn
KEY = (16, 128, 128, 16)
PKG = layer._DECODE.hosts[KEY]


def t(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


def attn(rows=8, nq=16, nkv=1, d=128, page=128, k=16):
    return dict(q=t(rows, nq, d), kv_cache=t(10, 2 * nkv, page, 72, dtype=U8), topk=t(nkv, rows, k, dtype=I32))


def decode_args(**s):
    a = attn(**s)
    rows = a["q"].shape[0]
    return dict(q=a["q"], kv_cache=a["kv_cache"], topk_idx=a["topk"], block_table=t(rows, 4, dtype=I32),
                seq_lens=t(rows, dtype=I32), sm_scale=0.088, output=t(*a["q"].shape), k_scale=t((), dtype=F32),
                v_scale=t((), dtype=F32), decode_query_len=1)


def index_args(**s):
    a = decode_args(**s)
    rows = a["q"].shape[0]
    a.pop("decode_query_len")
    return dict(index=None, iq=t(rows, 1, 128, dtype=E4), index_cache=t(10, 128, 72, dtype=U8),
                index_block_table=t(rows, 4, dtype=I32), index_seq_lens=t(rows, dtype=I32), init_blocks=1,
                local_blocks=4, **a, decode_query_len=1, max_seq_len=4096)


def prefill_args(**s):
    a = attn(**s)
    rows = a["q"].shape[0]
    return dict(q=a["q"], kv_cache=a["kv_cache"], topk=a["topk"], block_table=t(2, 4, dtype=I32),
                query_req_id=t(rows, dtype=I32), query_abs_pos=t(rows, dtype=I32),
                num_kv_heads=a["kv_cache"].shape[1] // 2, sm_scale=0.088, out=t(*a["q"].shape))


OPS = [("decode", E.nvfp4_sparse_decode, E.nvfp4_sparse_decode_blocker, decode_args, "sparse_decode"),
       ("index_decode", E.nvfp4_index_sparse_decode, E.nvfp4_index_sparse_decode_blocker, index_args,
        "index_sparse_decode"),
       ("prefill", E.nvfp4_sparse_prefill, E.nvfp4_sparse_prefill_blocker, prefill_args, "sparse_prefill")]
IDS = [o[0] for o in OPS]


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


@pytest.mark.parametrize("name,op,blocker,args,fn", OPS, ids=IDS)
def test_table_and_signature(name, op, blocker, args, fn):
    assert hasattr(importlib.import_module(PKG), fn)
    assert inspect.signature(blocker).parameters == inspect.signature(op).parameters
    assert E.nvfp4_sparse_attention_shapes() == {KEY}


@pytest.mark.parametrize("name,op,blocker,args,fn", OPS, ids=IDS)
def test_covered_routes_to_package(gfx950, monkeypatch, name, op, blocker, args, fn):
    a = args()
    assert blocker(**a) is None
    seen = []
    monkeypatch.setattr(importlib.import_module(PKG), fn, lambda *x, **k: seen.append((x, k)) or "route")
    op(**a)
    assert len(seen) == 1 and any(x is a["q"] for x in seen[0][0])


@pytest.mark.parametrize("shape", [dict(nq=8), dict(page=64), dict(k=32)], ids=["gqa8", "page64", "top32"])
@pytest.mark.parametrize("name,op,blocker,args,fn", OPS, ids=IDS)
def test_uncovered_reason_and_raise(gfx950, name, op, blocker, args, fn, shape):
    a = args(**shape)
    why = blocker(**a)
    assert "no kernel" in why and f"(covered: [{KEY}])" in why, why
    with pytest.raises(ValueError) as e:
        op(**a)
    assert str(e.value) == why


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    for _, _, blocker, args, _ in OPS:
        assert blocker(**args()).endswith("needs gfx950 with FlyDSL")


BRANCHES = [
    ("q", {"q": t(8, 16, 128, dtype=F32)}, "q "),
    ("cache", {"kv_cache": t(10, 2, 128, 72, dtype=BF)}, "KV cache"),
    ("topk", {"topk_idx": t(1, 4, 16, dtype=I32)}, "top-k"),
    ("tables", {"seq_lens": t(8, dtype=torch.int64)}, "seq lens"),
    ("output", {"output": t(8, 16, 64)}, "output"),
    ("scales", {"k_scale": t((), dtype=BF)}, "k / v scales"),
    ("query_len", {"decode_query_len": 0}, "decode_query_len"),
]


@pytest.mark.parametrize("label,edit,want", BRANCHES, ids=[b[0] for b in BRANCHES])
def test_decode_reasons(gfx950, label, edit, want):
    a = decode_args()
    a.update(edit)
    why = E.nvfp4_sparse_decode_blocker(**a)
    assert why is not None and why.startswith("nvfp4_sparse_decode: ") and want in why, why


def test_index_and_prefill_reasons(gfx950):
    a = index_args()
    a["iq"] = t(8, 1, 128)
    assert "index q" in E.nvfp4_index_sparse_decode_blocker(**a)
    a = index_args()
    a["index_cache"] = t(10, 128, 64, dtype=U8)
    assert "index cache" in E.nvfp4_index_sparse_decode_blocker(**a)
    a = prefill_args()
    a["num_kv_heads"] = 2
    assert "num_kv_heads" in E.nvfp4_sparse_prefill_blocker(**a)
    a = prefill_args()
    a["q"] = t(8, 16, 128, dtype=torch.float16)
    assert E.nvfp4_sparse_prefill_blocker(**a) is None


LAYER = dict(num_heads=16, num_kv_heads=1, num_index_heads=1, head_dim=128, block_size=128, topk_blocks=16,
             query_len=4, max_reqs=64, max_seq_len=65536)


def test_layer_routes_and_prepares(gfx950, monkeypatch):
    seen = []
    monkeypatch.setattr(importlib.import_module(PKG), "init", lambda device, **k: seen.append(k) or "indexer")
    assert inspect.signature(E.nvfp4_sparse_layer_blocker).parameters == \
        inspect.signature(E.nvfp4_sparse_layer).parameters == inspect.signature(E.prepare_nvfp4_sparse_layer).parameters
    lay = E.prepare_nvfp4_sparse_layer("cuda", **LAYER)
    assert isinstance(lay, E.Nvfp4SparseLayer) and lay.indexer == "indexer" and lay.key == KEY
    assert seen[0]["num_heads"] == 16 and "head_dim" not in seen[0]
    why = E.prepare_nvfp4_sparse_layer("cuda", **{**LAYER, "topk_blocks": 32})
    assert isinstance(why, str) and "no kernel" in why
    assert "rank and group" in E.nvfp4_sparse_layer_blocker("cuda", **LAYER, rank=0)
    assert "query_len" in E.nvfp4_sparse_layer_blocker("cuda", **{**LAYER, "query_len": 0})


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.attention.nvfp4_sparse_attention_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
