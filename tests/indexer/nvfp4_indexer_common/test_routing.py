# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the NVFP4 indexer prefill top-k entry on the host (meta tensors, no GPU)."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.indexer import nvfp4_indexer_common as E
from amd_kernels.indexer.nvfp4_indexer_common import prefill

I32, U8, E4 = torch.int32, torch.uint8, torch.float8_e4m3fn
KEY = (128, 128, 16)


def t(*shape, dtype=I32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def args(rows=40, heads=1, d=128, page=128, k=16):
    return dict(iq=t(rows, heads, d, dtype=E4), index_cache=t(10, page, 72, dtype=U8), block_table=t(2, 8),
                cu_seqlens_q=t(3), seq_lens=t(2), init_blocks=1, local_blocks=4, topk_idx=t(heads, rows, k),
                max_query_len=32, max_seq_len=8192, num_valid_pages=t(rows))


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)
    gfx = importlib.import_module(prefill._ROUTES.hosts[KEY] + ".gfx950")
    monkeypatch.setattr(gfx, "on_gfx950", lambda: True)


def test_table_and_signature():
    mod = prefill._module(KEY)
    assert all(hasattr(mod, n) for n in ("index_prefill_top16", "index_prefill_top16_blocker", "warm_up"))
    assert inspect.signature(E.index_prefill_top16_blocker).parameters == \
        inspect.signature(E.index_prefill_top16).parameters
    assert E.index_prefill_top16_shapes() == {KEY}


def test_covered_routes_to_package(gfx950, monkeypatch):
    a = args()
    assert E.index_prefill_top16_blocker(**a) is None
    seen = []
    monkeypatch.setattr(prefill._module(KEY), "index_prefill_top16", lambda *x, **k: seen.append((x, k)) or "out")
    assert E.index_prefill_top16(**a) == "out"
    assert seen[0][0][0] is a["iq"] and seen[0][1]["max_seq_len"] == 8192


@pytest.mark.parametrize("shape", [dict(d=64), dict(page=64), dict(k=32)], ids=["d64", "page64", "top32"])
def test_uncovered_reason_and_raise(gfx950, shape):
    a = args(**shape)
    why = E.index_prefill_top16_blocker(**a)
    assert "no kernel" in why and f"(covered: [{KEY}])" in why, why
    with pytest.raises(ValueError) as e:
        E.index_prefill_top16(**a)
    assert str(e.value) == why


def test_no_device_reason(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: False)
    assert E.index_prefill_top16_blocker(**args()).endswith("needs gfx950 with FlyDSL")


BRANCHES = [
    ("heads", {"iq": t(40, 3, 128, dtype=E4), "topk_idx": t(3, 40, 16)}, "index heads"),
    ("iq", {"iq": t(40, 1, 128, dtype=torch.bfloat16)}, "iq"),
    ("cache", {"index_cache": t(10, 128, 64, dtype=U8)}, "index cache"),
    ("topk", {"topk_idx": t(1, 20, 16)}, "top-k"),
    ("tables", {"seq_lens": t(2, dtype=torch.int64)}, "seq lens"),
    ("valid", {"num_valid_pages": t(10)}, "num_valid_pages"),
    ("seq_len", {"max_seq_len": 1 << 21}, "max_seq_len"),
]


@pytest.mark.parametrize("label,edit,want", BRANCHES, ids=[b[0] for b in BRANCHES])
def test_reasons(gfx950, label, edit, want):
    a = args()
    a.update(edit)
    why = E.index_prefill_top16_blocker(**a)
    assert why is not None and why.startswith("index_prefill_top16: ") and want in why, why


def test_layer_reason(gfx950):
    kw = dict(index_head_dim=128, block_size=128, topk_blocks=16)
    assert E.index_prefill_top16_layer_blocker(num_index_heads=1, **kw) is None
    assert "index heads" in E.index_prefill_top16_layer_blocker(num_index_heads=3, **kw)
    assert "no kernel" in E.index_prefill_top16_layer_blocker(num_index_heads=1, **{**kw, "topk_blocks": 8})


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.indexer.nvfp4_indexer_common; print(sorted(m for m in "
            "sys.modules if m.startswith('amd_kernels.') and m.count('.') >= 2 and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
