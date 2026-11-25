# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routing and reasons of the KDA entries on the host (meta tensors, no GPU): each covered key routes to its shape
package, an uncovered call gets a reason and the op raises it, every reason branch is reachable, and importing the
entry imports no shape package."""

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

from amd_kernels import routes
from amd_kernels.attention import kda_common as E
from amd_kernels.attention.kda_common import decode, mono, prefill

BF, F32, I32 = torch.bfloat16, torch.float32, torch.int32
META = "meta"


def t(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device=META)


def decode_args(heads=12, d=128, rows=12, tokens=6):
    n = -(-rows // tokens)
    lp = heads * d
    return dict(
        proj=t(rows, 6288), f_b_weight=t(lp, d), A_log=t(heads, dtype=F32), dt_bias=t(lp, dtype=F32),
        lower_bound=-5.0, conv_weight=t(3, 4, lp, dtype=F32),
        conv_state=t(n * tokens + 8, tokens + 3, 3 * lp).transpose(1, 2),
        state=t(n * tokens + 8, heads, d, d, dtype=F32),
        cu_seqlens=t(n + 1, dtype=I32), state_indices=t(n, tokens, dtype=I32), num_accepted=t(n, dtype=I32),
        norm_weight=t(d), norm_eps=1e-5, out=t(rows, heads, d),
    )  # fmt: skip


def prefill_args(heads=12, d=128, tokens=300, seqs=2, chunks=6):
    lp = heads * d
    proj = t(tokens, 4 * lp + d + 16)
    return dict(
        mixed=proj[:, : 3 * lp], raw_g=t(tokens, heads, d), beta_logit=proj[:, 4 * lp + d : 4 * lp + d + heads],
        gate=proj[:, 3 * lp : 4 * lp].view(tokens, heads, d), A_log=t(heads, dtype=F32), dt_bias=t(lp, dtype=F32),
        conv_weight=t(3, 4, lp, dtype=F32), conv_state=t(seqs + 2, 3 * lp, 3),
        state_cache=t(seqs + 2, heads, d, d, dtype=F32), state_indices=t(seqs, dtype=I32),
        has_initial_state=t(seqs, dtype=torch.bool), cu_seqlens=t(seqs + 1, dtype=I32),
        chunk_indices=t(chunks, 2, dtype=I32), chunk_offsets=t(seqs + 1, dtype=torch.int64),
        norm_weight=t(d, dtype=F32), norm_eps=1e-6, out=t(tokens, heads, d), lower_bound=-5.0,
    )  # fmt: skip


def mono_args(hidden=7168, heads=12, d=128, rows=12, qlen=6, nb=4):
    lp = heads * d
    return dict(
        ws=E.kda_mono_workspace(META, hidden=7168, heads=12, head_dim=128), layer=0, prefix=t(rows, hidden),
        delta=t(rows, hidden), blocks=t(rows, nb + 2, hidden), ares_nw=t(hidden), ares_qk=t(hidden), in_nw=t(hidden),
        w_in=t(6288, hidden), w_fb=t(lp, d), conv_w=t(3 * lp, 4, dtype=F32), conv_state=t(rows + 4, 3 * lp, qlen + 2),
        a_log=t(heads, dtype=F32), dt_bias=t(lp, dtype=F32), on_w=t(d), rstate=t(rows + 4, heads, d, d, dtype=F32),
        st_idx=t(rows // qlen, qlen, dtype=I32), num_acc=t(rows // qlen, dtype=I32), qlen=qlen, nblocks=nb,
    )  # fmt: skip


OPS = [
    ("decode", decode, E.kda_decode_fb_gate_spec, E.kda_decode_fb_gate_spec_blocker, decode_args,
     "kda_decode_fb_gate_spec", ("kda_decode_fb_gate_spec", "blocker", "host")),
    ("prefill", prefill, E.kda_prefill, E.kda_prefill_blocker, prefill_args, "prefill",
     ("prefill", "chunk_table")),
    ("mono", mono, E.kda_mono_decode, E.kda_mono_decode_blocker, mono_args, "kda_mono_decode_k1",
     ("kda_mono_decode_k1", "Workspace", "host")),
]  # fmt: skip
IDS = [o[0] for o in OPS]


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(routes, "available", lambda: True)


def call(fn, a):
    """``fn(**a)`` with the positional-or-keyword arguments passed positionally (the mono entries' first is ``ws``)."""
    params = inspect.signature(fn).parameters
    pos = [a[k] for k, p in params.items() if p.kind == p.POSITIONAL_OR_KEYWORD and k in a]
    return fn(*pos, **{k: v for k, v in a.items() if params[k].kind == inspect.Parameter.KEYWORD_ONLY})


@pytest.mark.parametrize("name,mod,op,reason,args,fn,exports", OPS, ids=IDS)
def test_table_imports_and_exports(name, mod, op, reason, args, fn, exports):
    for key, path in mod._ROUTES.hosts.items():
        pkg = importlib.import_module(path)
        assert all(hasattr(pkg, e) for e in exports), (path, exports)
        assert mod._ROUTES.host(key) is pkg


@pytest.mark.parametrize("name,mod,op,reason,args,fn,exports", OPS, ids=IDS)
def test_reason_takes_the_op_arguments(name, mod, op, reason, args, fn, exports):
    assert list(inspect.signature(reason).parameters.values()) == list(inspect.signature(op).parameters.values())
    assert inspect.signature(E.kda_mono_workspace_blocker).parameters == inspect.signature(
        E.kda_mono_workspace).parameters


def test_shapes_are_the_table_keys():
    assert E.kda_decode_fb_gate_spec_shapes() == frozenset(decode._ROUTES.hosts) == {(12, 128)}
    assert E.kda_prefill_shapes() == frozenset(prefill._ROUTES.hosts) == {(12, 128)}
    assert E.kda_mono_decode_shapes() == frozenset(mono._ROUTES.hosts) == {(7168, 12, 128)}


@pytest.mark.parametrize("name,mod,op,reason,args,fn,exports", OPS, ids=IDS)
def test_covered_shape_routes_and_has_no_reason(gfx950, name, mod, op, reason, args, fn, exports):
    a = args()
    assert call(reason, a) is None
    if name == "mono":
        key = mono._ROUTES.owner(type(a["ws"]))
        assert key == mono._key(a["prefix"], a["rstate"])
    else:
        key = mod._key(a["state"] if name == "decode" else a["state_cache"])
    assert mod._ROUTES.host(key).__name__ == mod._ROUTES.hosts[key]


@pytest.mark.parametrize("name,mod,op,reason,args,fn,exports", OPS[:2], ids=IDS[:2])
@pytest.mark.parametrize("heads,d", [(16, 128), (12, 64)])
def test_uncovered_shape_reason_and_raise(gfx950, name, mod, op, reason, args, fn, exports, heads, d):
    a = args(heads=heads, d=d)
    why = call(reason, a)
    assert "no kernel" in why and str(sorted(mod._ROUTES.hosts)) in why, why
    with pytest.raises(ValueError) as e:
        call(op, a)
    assert str(e.value) == why


def test_uncovered_mono(gfx950):
    why = E.kda_mono_workspace_blocker(META, hidden=6144, heads=12, head_dim=128)
    assert "no kernel" in why and str(sorted(mono._ROUTES.hosts)) in why
    with pytest.raises(ValueError) as e:
        E.kda_mono_workspace(META, hidden=6144, heads=12, head_dim=128)
    assert str(e.value) == why
    a = mono_args(hidden=6144)
    assert "no kernel" in call(E.kda_mono_decode_blocker, a)
    a = mono_args()
    a["ws"] = object()
    assert "is not a kda_mono_workspace" in call(E.kda_mono_decode_blocker, a)
    with pytest.raises(ValueError, match="is not a kda_mono_workspace"):
        call(E.kda_mono_decode, a)


def test_mono_workspace_of_another_key(gfx950, monkeypatch):
    """A call whose tensors give another covered key than its workspace's gets the "built for" reason."""
    other = (6144, 12, 128)
    monkeypatch.setitem(mono._ROUTES.hosts, other, mono._ROUTES.hosts[(7168, 12, 128)])
    a = mono_args(hidden=6144)
    assert call(E.kda_mono_decode_blocker, a) == (
        "kda_mono_decode: workspace built for (7168, 12, 128), call is (6144, 12, 128)")


def test_no_device_reason():
    if routes.available():
        pytest.skip("gfx950 present")
    for _, _, _, reason, args, fn, _ in OPS:
        assert call(reason, args()).endswith("needs gfx950 with FlyDSL")
    assert E.kda_mono_workspace_blocker(META, hidden=7168, heads=12, head_dim=128).endswith(
        "needs gfx950 with FlyDSL")


def _set(k, v):
    return lambda a: a.update({k: v})


DECODE_BRANCHES = [
    ("state_3d", _set("state", t(8, 12, 128 * 128, dtype=F32)), "fp32 [slots, heads"),
    ("proj_1d", _set("proj", t(12 * 6288)), "(2-D)"),
    ("tokens", _set("state_indices", t(2, 9, dtype=I32)), "9 tokens a sequence (the kernel takes 1..8)"),
    ("blocker", _set("proj", t(12, 6288, dtype=F32)), "kda_decode_fb_gate_spec: proj"),
    ("cu_dtype", _set("cu_seqlens", t(3, dtype=torch.int64)), "(int32)"),
    ("out_rows", _set("out", t(5, 12, 128)), "for 12 rows"),
    ("uniform", lambda a: a.update(uniform=True, cu_seqlens=t(4, dtype=I32)), "uniform with 12 rows for 3 sequences"),
    ("weights", _set("A_log", t(12)), "A_log"),
    ("conv_weight", _set("conv_weight", t(4608, 4, dtype=F32)), "conv weight"),
]
PREFILL_BRANCHES = [
    ("state_3d", _set("state_cache", t(4, 12, 128 * 128, dtype=F32)), "fp32 [slots, heads"),
    ("state_dtype", _set("state_cache", t(4, 12, 128, 128, dtype=BF)), "state cache"),
    ("mixed", _set("mixed", t(300, 4608, dtype=F32)), "mixed"),
    ("tokens", lambda a: a.update(mixed=t(0, 6288)[:, :4608]), "0 tokens"),
    ("raw_g", _set("raw_g", t(300, 12, 64)), "raw_g"),
    ("beta", _set("beta_logit", t(300, 13)), "beta_logit"),
    ("gate", _set("gate", t(300, 12, 128, dtype=F32)), "gate"),
    ("out", _set("out", t(299, 12, 128)), "out"),
    ("conv_state", _set("conv_state", t(4, 4608, 3, dtype=F32)), "conv state"),
    ("weights", _set("A_log", t(12)), "A_log"),
    ("conv_weight", _set("conv_weight", t(4608, 4, dtype=F32)), "conv weight"),
    ("norm_weight", _set("norm_weight", t(64)), "norm weight"),
    ("cu", _set("cu_seqlens", t(3, dtype=torch.int64)), "cu_seqlens"),
    ("has", _set("has_initial_state", t(2, dtype=torch.uint8)), "has_initial_state"),
    ("state_indices", _set("state_indices", t(2, 1, dtype=I32)), "state indices"),
    ("chunks", _set("chunk_offsets", t(3, dtype=I32)), "chunk indices"),
    ("table", _set("table", t(6, 3, dtype=I32)), "table"),
    ("counts", _set("chunk_counts", [3]), "1 chunk counts for 2 sequences"),
    ("ckpt", _set("checkpoint_state", t(4, 12, 64, 64, dtype=F32)), "checkpoint state"),
    ("ckpt_offsets", _set("checkpoint_state", t(4, 12, 128, 128, dtype=F32)), "checkpoint offsets"),
    ("impl", _set("impl", "fast"), "impl"),
]
MONO_BRANCHES = [
    ("rows", lambda a: a.update(mono_args(rows=18, qlen=6)), "18 rows (the kernel takes 1..16)"),
    ("qlen", _set("qlen", 5), "rows of 5-token requests"),
    ("scratch", _set("ws", E.kda_mono_workspace(META, hidden=7168, heads=12, head_dim=128, max_tokens=4)),
     "workspace scratch"),
    ("layer", _set("layer", 256), "layer slot 256"),
    ("prefix", _set("prefix", t(12, 7168, dtype=F32)), "prefix"),
    ("delta", _set("delta", t(12, 2 * 7168)[:, :7168]), "prefix"),
    ("blocks", _set("blocks", t(12, 6, 7168)[1:].repeat(1, 1, 1)[:, :, :7168]), "blocks"),
    ("blocks_stride", _set("blocks", t(12, 6, 7172)[:, :, 2:7170]), "blocks"),
    ("blocks_dtype", _set("blocks", t(12, 6, 7168, dtype=F32)), "blocks"),
    ("delta_dtype", _set("delta", t(12, 7168, dtype=F32)), "prefix"),
    ("nblocks", _set("nblocks", 9), "nblocks 9"),
    ("write_idx", _set("write_idx", 2), "write_idx 2"),
    ("w_in", _set("w_in", t(6284, 7168)), "w_in"),
    ("w_fb", _set("w_fb", t(1536, 128).t()), "w_fb"),
    ("w_in_dtype", _set("w_in", t(6288, 7168, dtype=torch.float8_e4m3fn)), "w_in"),
    ("a_log", _set("a_log", t(12)), "a_log"),
    ("conv_w", _set("conv_w", t(4608, 4)), "conv_w"),
    ("on_w", _set("on_w", t(128, dtype=F32)), "on_w"),
    ("conv_state", _set("conv_state", t(16, 4608, 8, dtype=F32)), "conv state"),
    ("rstate", _set("rstate", t(16, 12, 128, 128, dtype=BF)), "recurrent state"),
    ("st_idx", _set("st_idx", t(2, 6, dtype=torch.int64)), "st_idx"),
]  # fmt: skip


@pytest.mark.parametrize("label,edit,want", DECODE_BRANCHES, ids=[b[0] for b in DECODE_BRANCHES])
def test_decode_reasons(gfx950, label, edit, want):
    a = decode_args()
    edit(a)
    why = E.kda_decode_fb_gate_spec_blocker(**a)
    assert why is not None and why.startswith("kda_decode_fb_gate_spec: ") and want in why, why


@pytest.mark.parametrize("label,edit,want", PREFILL_BRANCHES, ids=[b[0] for b in PREFILL_BRANCHES])
def test_prefill_reasons(gfx950, label, edit, want):
    a = prefill_args()
    edit(a)
    why = E.kda_prefill_blocker(**a)
    assert why is not None and why.startswith("kda_prefill: ") and want in why, why


@pytest.mark.parametrize("label,edit,want", MONO_BRANCHES, ids=[b[0] for b in MONO_BRANCHES])
def test_mono_reasons(gfx950, label, edit, want):
    a = mono_args()
    edit(a)
    why = call(E.kda_mono_decode_blocker, a)
    assert why is not None and why.startswith("kda_mono_decode: ") and want in why, why


def test_mono_workspace_reasons(gfx950):
    r = E.kda_mono_workspace_blocker
    assert r(META, hidden=7168, heads=12, head_dim=128) is None
    assert "17 rows" in r(META, hidden=7168, heads=12, head_dim=128, max_tokens=17)
    assert "9 blocks" in r(META, hidden=7168, heads=12, head_dim=128, max_blocks=9)


def test_chunk_table_routes_by_state_cache():
    a = prefill_args(heads=16)
    with pytest.raises(ValueError, match="kda_prefill_chunk_table: no kernel for heads=16"):
        E.kda_prefill_chunk_table(a["cu_seqlens"], a["chunk_indices"], a["state_indices"], a["has_initial_state"],
                                  state_cache=a["state_cache"])


def test_import_loads_no_shape_package():
    code = ("import sys; import amd_kernels.attention.kda_common; print(sorted(m for m in sys.modules "
            "if m.startswith('amd_kernels.attention.kda_') and '_common' not in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout


@pytest.mark.parametrize("old,new", [
    ("kda_decode_fb_gate_spec_unsupported_reason", "kda_decode_fb_gate_spec_blocker"),
    ("kda_prefill_unsupported_reason", "kda_prefill_blocker"),
    ("kda_mono_decode_unsupported_reason", "kda_mono_decode_blocker"),
    ("kda_mono_workspace_unsupported_reason", "kda_mono_workspace_blocker"),
])  # fmt: skip
def test_renamed_aliases(gfx950, old, new):
    alias = getattr(E, old)
    assert alias.renamed_to == new and old not in E.__all__
    a = {"kda_decode_fb_gate_spec_blocker": decode_args, "kda_prefill_blocker": prefill_args,
         "kda_mono_decode_blocker": mono_args}.get(new)
    if a is None:
        assert alias(META, hidden=7168, heads=12, head_dim=128) is None
    else:
        assert call(alias, a()) == call(getattr(E, new), a())


def test_prepare_mono_workspace(gfx950):
    ws = E.prepare_kda_mono_workspace(META, hidden=7168, heads=12, head_dim=128)
    assert mono._ROUTES.owner(type(ws)) == (7168, 12, 128)
    assert E.prepare_kda_mono_workspace(META, hidden=6144, heads=12, head_dim=128) == (
        E.kda_mono_workspace_blocker(META, hidden=6144, heads=12, head_dim=128))
