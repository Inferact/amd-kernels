# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the indexer entries on meta operands (host only, no GPU): every table row imports and
exports what the entry calls, each covered key routes to its package, an off-table shape gets a reason and a
ValueError, every reason branch fires, and importing the entry imports no shape package."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels.indexer.mxfp4_indexer_common import paged, prefill

META = "meta"


def z(*shape, dtype=torch.uint8):
    return torch.empty(*shape, dtype=dtype, device=META)


def paged_args(seqs=3, qlen=4, blocks=128, heads=32, d=128, topk=2048, page=64):
    rows = seqs * qlen
    cache = torch.empty_strided((40, page, d // 2 + d // 32), (4352, d // 2 + d // 32, 1), dtype=torch.uint8,
                                device=META)
    return dict(
        q_payload=z(rows, heads, d // 2), q_scales=z(rows, 1, 4, 16, 4), weights=z(rows, heads, dtype=torch.float32),
        cache=cache, block_table=z(seqs, blocks, dtype=torch.int32), query_start_loc=z(seqs + 1, dtype=torch.int32),
        row_ends=z(rows, dtype=torch.int32), max_query_len=qlen, out_indices=z(rows, topk, dtype=torch.int32),
        out_scores=z(rows, topk, dtype=torch.float32),
    )


def score_args(rows=16, pages=64, heads=32, d=128):
    return dict(
        q_payload=z(rows, heads, d // 2), q_scale=z(rows, 1, 4, 16, 4), weights=z(rows, heads, dtype=torch.bfloat16),
        key_payload=z(pages, 1, 4, 64, d // 8), key_scale=z(pages, 1, 4, 64), ks=z(rows, dtype=torch.int32),
        ke=z(rows, dtype=torch.int32), out=z(rows, pages * 64, dtype=torch.float32),
    )


def topk_args(rows=16, width=8192, topk=2048):
    return dict(logits=z(rows, width, dtype=torch.float32), ks=z(rows, dtype=torch.int32),
                ke=z(rows, dtype=torch.int32), out=z(rows, topk, dtype=torch.int32))


@pytest.fixture
def on_gfx950(monkeypatch):
    monkeypatch.setattr(paged, "available", lambda: True)
    monkeypatch.setattr(prefill, "available", lambda: True)


def test_tables_import_and_export():
    for name in paged._SHAPE_HOSTS.values():
        mod = importlib.import_module(name)
        assert callable(mod.mxfp4_paged_score_topk) and hasattr(mod.mxfp4_paged_score_topk, "entry")
        assert callable(mod.prefer_fused) and callable(mod.host.plan)
        assert callable(mod.host._scratch) and callable(mod.host._scratch_set)
        assert mod.geometry.COUNTER_BYTES and mod.geometry.MAX_ROWS and mod.geometry.QMAX
        assert mod.geometry.COUNTER_GROUPS and mod.host.TIER_WIDTHS
    for name in prefill._SCORE_HOSTS.values():
        mod = importlib.import_module(name)
        assert callable(mod.score_rows) and importlib.import_module(name + ".kernel").MAX_OUT_BYTES
    for name in prefill._TOPK_HOSTS.values():
        mod = importlib.import_module(name)
        assert callable(mod.topk_rows) and callable(mod.workspace_bytes)


def test_shapes_are_the_tables():
    assert paged.mxfp4_paged_score_topk_shapes() == frozenset(paged._SHAPE_HOSTS)
    assert prefill.mxfp4_prefill_score_shapes() == frozenset(prefill._SCORE_HOSTS)
    assert prefill.mxfp4_prefill_topk_shapes() == frozenset(prefill._TOPK_HOSTS)


def test_covered_keys_route_to_their_package():
    for key, name in paged._SHAPE_HOSTS.items():
        heads, d, topk, page = key
        a = paged_args(heads=heads, d=d, topk=topk, page=page)
        assert paged._route(a["q_payload"], a["out_indices"], a["cache"]).__name__ == name
        assert paged.mxfp4_paged_score_topk_kernel(**a) is importlib.import_module(name).mxfp4_paged_score_topk
        assert paged._scratch_host(a["cache"], a["out_indices"], need=True).__name__ == name
    for (heads, d), name in prefill._SCORE_HOSTS.items():
        assert prefill._score_route(score_args(heads=heads, d=d)["q_payload"]).__name__ == name
    for (topk,), name in prefill._TOPK_HOSTS.items():
        assert prefill._topk_route(topk_args(topk=topk)["out"]).__name__ == name
        assert prefill.prefill_topk_workspace_bytes(64, 8192, topk=topk) == importlib.import_module(
            name).workspace_bytes(64, 8192)


def test_signatures_match_the_ops():
    pairs = [
        (paged.mxfp4_paged_score_topk, paged.mxfp4_paged_score_topk_unsupported_reason),
        (paged.mxfp4_paged_score_topk, paged.mxfp4_paged_score_topk_kernel),
        (prefill.mxfp4_prefill_score, prefill.mxfp4_prefill_score_unsupported_reason),
        (prefill.mxfp4_prefill_topk, prefill.mxfp4_prefill_topk_unsupported_reason),
    ]
    for op, other in pairs:
        assert inspect.signature(op).parameters == inspect.signature(other).parameters, other.__name__
    routed = [
        (paged.mxfp4_paged_score_topk, paged._SHAPE_HOSTS, "mxfp4_paged_score_topk", ("dbg", "split")),
        (prefill.mxfp4_prefill_score, prefill._SCORE_HOSTS, "score_rows", ()),
        (prefill.mxfp4_prefill_topk, prefill._TOPK_HOSTS, "topk_rows", ()),
    ]
    for op, hosts, name, extra in routed:
        for mod in hosts.values():
            params = inspect.signature(getattr(importlib.import_module(mod), name)).parameters
            assert list(inspect.signature(op).parameters) == [p for p in params if p not in extra], mod


def test_off_table_shapes(on_gfx950):
    cases = [
        (paged.mxfp4_paged_score_topk, paged.mxfp4_paged_score_topk_unsupported_reason, paged_args(heads=16),
         "heads=16, d=128, topk=2048, page_rows=64", "(32, 128, 2048, 64)"),
        (paged.mxfp4_paged_score_topk, paged.mxfp4_paged_score_topk_unsupported_reason, paged_args(topk=1024),
         "topk=1024", "(32, 128, 2048, 64)"),
        (prefill.mxfp4_prefill_score, prefill.mxfp4_prefill_score_unsupported_reason, score_args(heads=64),
         "heads=64, d=128", "(32, 128)"),
        (prefill.mxfp4_prefill_topk, prefill.mxfp4_prefill_topk_unsupported_reason, topk_args(topk=512),
         "topk=512", "(2048,)"),
    ]
    for op, reason, args, dims, covered in cases:
        why = reason(**args)
        assert why is not None and "no kernel" in why and dims in why and f"(covered: {covered})" in why, why
        with pytest.raises(ValueError) as err:
            op(**args)
        assert str(err.value) == why
    a = paged_args(topk=1024)
    assert paged.paged_score_topk_prefer(4, 8192, cache=a["cache"], out_indices=a["out_indices"]) is False
    with pytest.raises(ValueError, match="no kernel"):
        paged.paged_score_topk_preallocate("cpu", [], cache=a["cache"], out_indices=a["out_indices"])
    with pytest.raises(ValueError, match="no kernel"):
        prefill.prefill_topk_workspace_bytes(8, 8192, topk=512)


def test_served_shapes_have_no_reason(on_gfx950):
    assert paged.mxfp4_paged_score_topk_unsupported_reason(**paged_args()) is None
    assert paged.mxfp4_paged_score_topk_unsupported_reason(**paged_args(seqs=64, qlen=4, blocks=2048)) is None
    assert paged.mxfp4_paged_score_topk_unsupported_reason(**paged_args(seqs=1, qlen=1, blocks=1)) is None
    assert prefill.mxfp4_prefill_score_unsupported_reason(**score_args()) is None
    assert prefill.mxfp4_prefill_score_unsupported_reason(**score_args(rows=0)) is None
    assert prefill.mxfp4_prefill_topk_unsupported_reason(**topk_args()) is None
    a = topk_args()
    assert prefill.mxfp4_prefill_topk_unsupported_reason(**a, order=z(16, dtype=torch.int32)) is None


def _with(args, **changes):
    return {**args, **changes}


def test_paged_reason_branches(on_gfx950, monkeypatch):
    a = paged_args()
    i32, f32, u8 = torch.int32, torch.float32, torch.uint8
    bad_cache = torch.empty_strided((40, 64, 68), (4096, 68, 1), dtype=u8, device=META)
    cases = [
        (paged_args(seqs=65, qlen=4), "260 rows (the kernel takes 1..256)"),
        (_with(a, max_query_len=7), "max_query_len 7"),
        (_with(a, max_query_len=0), "max_query_len 0"),
        (_with(a, query_start_loc=z(1, dtype=i32)), "query_start_loc"),
        (_with(a, query_start_loc=z(4, dtype=torch.int64)), "query_start_loc"),
        (_with(a, block_table=z(3, 128, dtype=torch.int64)), "block_table"),
        (_with(a, block_table=z(2, 128, dtype=i32)), "block_table"),
        (_with(a, block_table=z(128, 3, dtype=i32).t()), "block_table"),
        (_with(a, block_table=z(3, 2049, dtype=i32)), "block table of 131136 keys"),
        (_with(a, cache=bad_cache), "index cache"),
        (_with(a, cache=z(40, 64, 68, dtype=torch.int8)), "index cache"),
        (_with(a, q_payload=z(12, 64, 32).transpose(1, 2)), "q_payload"),
        (_with(a, q_scales=z(12, 1, 4, 16, 4, dtype=torch.int8)), "q_scales"),
        (_with(a, q_scales=z(2, 1, 4, 16, 4)), "q_scales"),
        (_with(a, q_scales=z(12, 1, 4, 16, 2)), "q_scales"),
        (_with(a, query_start_loc=z(4, 1, dtype=i32)), "query_start_loc"),
        (_with(a, row_ends=z(24, dtype=i32)[::2]), "row_ends"),
        (_with(a, out_scores=torch.empty(12, 2048, dtype=f32)), "GPU"),
        (_with(a, **{k: torch.empty(v.shape, dtype=v.dtype) for k, v in a.items() if isinstance(v, torch.Tensor)}),
         "GPU"),
        (_with(a, weights=z(12, 32, dtype=torch.bfloat16)), "weights"),
        (_with(a, weights=z(32, 12, dtype=f32).t()), "weights"),
        (_with(a, row_ends=z(11, dtype=i32)), "row_ends"),
        (_with(a, out_indices=z(12, 2048, dtype=f32)), "out_indices"),
        (_with(a, out_scores=z(12, 1024, dtype=f32)), "out_scores"),
        (_with(a, out_indices=z(2048, 12, dtype=i32).t()), "out_indices"),
    ]
    for args, want in cases:
        why = paged.mxfp4_paged_score_topk_unsupported_reason(**args)
        assert why is not None and why.startswith("mxfp4_paged_score_topk: ") and want in why, (want, why)
    many = paged_args(seqs=256, qlen=1)
    many["max_query_len"] = 6  # 256 sequences x 2 row tiles of 3 within 256 rows
    monkeypatch.setattr(importlib.import_module(paged._SHAPE_HOSTS[(32, 128, 2048, 64)]).geometry, "COUNTER_GROUPS",
                        511)
    assert "row tiles" in paged.mxfp4_paged_score_topk_unsupported_reason(**many)
    monkeypatch.setattr(paged, "available", lambda: False)
    assert paged.mxfp4_paged_score_topk_unsupported_reason(**a) == "mxfp4_paged_score_topk: needs gfx950 with FlyDSL"


def test_prefill_reason_branches(on_gfx950, monkeypatch):
    s = score_args()
    i32, f32, bf16 = torch.int32, torch.float32, torch.bfloat16
    cases = [
        (_with(s, q_payload=z(16, 64, 32).transpose(1, 2)), "q_payload"),
        (_with(s, q_scale=z(16, 1, 4, 16, 4, dtype=torch.int8)), "q_scale"),
        (_with(s, weights=z(16, 32, dtype=f32)), "weights"),
        (_with(s, weights=z(15, 32, dtype=bf16)), "weights"),
        (_with(s, key_payload=z(64, 1, 4, 64, 8)), "key_payload"),
        (_with(s, key_scale=z(64, 1, 64, 4).transpose(2, 3)), "key_scale"),
        (_with(s, key_scale=z(32, 1, 4, 64)), "key_scale"),
        (_with(s, q_scale=z(16, 1, 4, 16, 2)), "q_scale"),
        (_with(s, ks=z(16, dtype=torch.int64)), "ks and ke"),
        (_with(s, ks=z(32, dtype=i32)[::2]), "ks and ke"),
        (_with(s, weights=torch.empty(16, 32, dtype=bf16)), "GPU"),
        (_with(s, ke=z(15, dtype=i32)), "ks and ke"),
        (_with(s, out=z(16, 4096, dtype=bf16)), "out must be fp32"),
        (_with(s, out=z(4096, 16, dtype=f32).t()), "out must be fp32"),
        (_with(s, out=z(16, 4032, dtype=f32)), ">= pages * 64"),
        (_with(score_args(rows=8192, pages=1100)), "logits of"),
    ]
    for args, want in cases:
        why = prefill.mxfp4_prefill_score_unsupported_reason(**args)
        assert why is not None and why.startswith("mxfp4_prefill_score: ") and want in why, (want, why)
    t = topk_args()
    cases = [
        (_with(t, logits=z(16, 8192, dtype=bf16)), "logits must be fp32"),
        (_with(t, logits=z(8192, 16, dtype=f32).t()), "logits must be fp32"),
        (_with(t, logits=z(16, 8194, dtype=f32)[:, :8192]), "logits must be fp32"),
        (_with(t, ks=z(16, dtype=torch.int64)), "ks and ke"),
        (_with(t, ke=z(32, dtype=i32)[::2]), "ks and ke"),
        (_with(t, out=z(16, 2048, dtype=torch.int64)), "out must be int32"),
        (_with(t, out=z(8, 2048, dtype=i32)), "out must be int32"),
        (_with(t, out=z(2048, 16, dtype=i32).t()), "out must be int32"),
        (_with(t, order=z(16, dtype=torch.int64)), "order"),
        (_with(t, order=torch.empty(16, dtype=i32)), "GPU"),
        (_with(t, ks=torch.empty(16, dtype=i32)), "GPU"),
    ]
    for args, want in cases:
        why = prefill.mxfp4_prefill_topk_unsupported_reason(**args)
        assert why is not None and why.startswith("mxfp4_prefill_topk: ") and want in why, (want, why)
    misaligned = torch.empty(16 * 8192 + 1, dtype=torch.float32)[1:].view(16, 8192)
    why = prefill.mxfp4_prefill_topk_unsupported_reason(misaligned, *list(topk_args().values())[1:])
    assert why == "mxfp4_prefill_topk: logits must be 16-byte aligned"
    monkeypatch.setattr(prefill, "available", lambda: False)
    assert prefill.mxfp4_prefill_score_unsupported_reason(**s) == "mxfp4_prefill_score: needs gfx950 with FlyDSL"
    assert prefill.mxfp4_prefill_topk_unsupported_reason(**t) == "mxfp4_prefill_topk: needs gfx950 with FlyDSL"


def test_import_loads_no_shape_package():
    code = (
        "import sys; import amd_kernels.indexer.mxfp4_indexer_common as e; "
        "print(sorted(m for m in sys.modules if m.startswith('amd_kernels.indexer.') and 'common' not in m))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
