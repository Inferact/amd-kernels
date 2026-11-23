# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``qkv_a_norm_kv_insert`` and ``indexer_prep`` through the entry: at the smallest and the largest served row count,
every written tensor (outputs, KV and indexer caches, head weights, top-k rows) equals the shape package's direct call
byte for byte, and the reason is None; the reason's layout branches on crafted GPU operands."""

import pytest
import torch
from amd_kernels.attention import mla_decode_common as E
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.indexer_prep import indexer_prep as direct_prep
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.qkv_a_norm_kv_insert import (
    IndexSide,
    qkv_a_norm_kv_insert as direct_insert,
)

from tests.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.cases import EPS, WEIGHT_SCALE, owned_slots

from .cases import assert_same, clone, require_gfx950

ROWS = (1, 192)
PAGES = 8
PAGE_STRIDE = 4352 + 256


def bf16(gen, *shape, scale=2.0, shift=0.0):
    return (torch.randn(shape, generator=gen) * scale + shift).bfloat16().cuda()


def index_cache(gen):
    base = torch.randint(0, 256, (PAGES * PAGE_STRIDE,), generator=gen, dtype=torch.uint8).cuda()
    return base.as_strided((PAGES, 64, 68), (PAGE_STRIDE, 68, 1))


def insert_args(n, form, gen):
    """``form``: "norm" (no cache), "insert" (KV insert), "index" (KV insert + the indexer side)."""
    cols = 2272 if form == "index" else 2112
    args = [bf16(gen, n, cols), bf16(gen, 1536, scale=0.3, shift=1), bf16(gen, 512, scale=0.3, shift=1), EPS,
            torch.zeros((n, 1536), dtype=torch.bfloat16, device="cuda"),
            torch.zeros((n, 512), dtype=torch.bfloat16, device="cuda"), None, None, None]
    if form != "norm":
        args[6] = torch.randint(0, 256, (PAGES, 64, 656), generator=gen, dtype=torch.uint8).cuda()
        args[7] = owned_slots(n, n, PAGES, gen, owned=3)
    if form == "index":
        args[8] = dict(ln_weight=bf16(gen, 128, scale=0.3, shift=1), ln_bias=bf16(gen, 128, scale=0.1),
                       cache=index_cache(gen), slot_mapping=owned_slots(n, n, PAGES, gen, owned=3),
                       weights=torch.zeros((n, 32), dtype=torch.float32, device="cuda"), weight_scale=WEIGHT_SCALE)
    return args


def prep_args(n, keys, gen):
    args = [bf16(gen, n, 32, 128), None, None, None, None, EPS, None, index_cache(gen),
            torch.zeros((n + 2, 2048), dtype=torch.int32, device="cuda"), WEIGHT_SCALE,
            torch.zeros((n, 32, 64), dtype=torch.uint8, device="cuda"),
            torch.zeros((n, 1, 4, 16, 4), dtype=torch.uint8, device="cuda"), None]
    if keys:
        args[1:5] = [bf16(gen, n, 128), bf16(gen, n, 32), bf16(gen, 128, scale=0.3, shift=1), bf16(gen, 128, scale=0.1)]
        args[6] = owned_slots(n, n, PAGES, gen, owned=3)
        args[12] = torch.zeros((n, 32), dtype=torch.float32, device="cuda")
    return args


@pytest.mark.parametrize("form", ["norm", "insert", "index"])
@pytest.mark.parametrize("n", ROWS)
@torch.inference_mode()
def test_insert_equals_the_package(n, form):
    require_gfx950()
    args = insert_args(n, form, torch.Generator().manual_seed(n))
    assert E.qkv_a_norm_kv_insert_blocker(*args) is None
    want = clone(args)
    if want[8] is not None:
        want[8] = IndexSide(**want[8])
    direct_insert(*want)
    E.qkv_a_norm_kv_insert(*args)
    torch.cuda.synchronize()
    side = want[8]
    want_t = want[:8] + ([side.ln_weight, side.ln_bias, side.cache, side.slot_mapping, side.weights] if side else [])
    got_t = args[:8] + ([args[8][k] for k in ("ln_weight", "ln_bias", "cache", "slot_mapping", "weights")]
                        if side else [])
    assert_same(got_t, want_t, ("insert", n, form))


@pytest.mark.parametrize("keys", [False, True])
@pytest.mark.parametrize("n", ROWS)
@torch.inference_mode()
def test_prep_equals_the_package(n, keys):
    require_gfx950()
    args = prep_args(n, keys, torch.Generator().manual_seed(100 + n))
    assert E.indexer_prep_blocker(*args) is None
    want = clone(args)
    direct_prep(*want)
    E.indexer_prep(*args)
    torch.cuda.synchronize()
    assert_same(args, want, ("prep", n, keys))


def with_(args, i, value):
    args = list(args)
    if isinstance(i, tuple):
        args[i[0]] = {**args[i[0]], i[1]: value}
        if value is None:
            del args[i[0]][i[1]]
    else:
        args[i] = value
    return args


INSERT_BRANCHES = [
    (0, lambda a: a[0].half(), "operand layout (supports_qkv_a_norm_kv_insert)"),
    (0, lambda a: a[0][:, :2112].contiguous(), "qkv_a has 2112 columns, the row needs 2272"),
    (1, lambda a: a[1][:1024], "norm weights"),
    (4, lambda a: a[4].float(), "q_out, kv_out bf16 contiguous"),
    (5, lambda a: torch.zeros((3, 512), dtype=torch.bfloat16, device="cuda"), "q_out, kv_out bf16 contiguous"),
    (6, lambda a: a[6][:0], "non-empty kv_cache"),
    (7, lambda a: a[7].int(), "int64 contiguous slot_mapping"),
    ((8, "weights"), lambda a: None, "index keys"),
    ((8, "cache"), lambda a: a[8]["cache"].as_strided((PAGES, 64, 68), (4360, 68, 1)), "indexer cache uint8"),
    ((8, "cache"), lambda a: a[8]["cache"].as_strided((PAGES, 64, 68), (4096, 64, 1)), "indexer cache uint8"),
    ((8, "slot_mapping"), lambda a: a[8]["slot_mapping"].int(), "index slot_mapping"),
    ((8, "weights"), lambda a: a[8]["weights"][:, :16], "index weights"),
    ((8, "ln_bias"), lambda a: a[8]["ln_bias"].float(), "index ln_weight / ln_bias"),
]


@pytest.mark.parametrize("i,make,text", INSERT_BRANCHES, ids=[b[2][:30] for b in INSERT_BRANCHES])
def test_insert_reasons(i, make, text):
    require_gfx950()
    args = insert_args(4, "index", torch.Generator().manual_seed(0))
    why = E.qkv_a_norm_kv_insert_blocker(*with_(args, i, make(args)))
    assert why is not None and why.startswith("qkv_a_norm_kv_insert: ") and text in why, why


def strided_q(n):
    return torch.zeros((n, 32 * 128 + 4), dtype=torch.bfloat16, device="cuda")[:, :4096].view(n, 32, 128)


PREP_BRANCHES = [
    (0, lambda a: a[0].float(), "operand layout (supports_indexer_prep)"),
    (0, lambda a: strided_q(4), "q row stride"),
    (10, lambda a: a[10][:, :, :32], "q_payload"),
    (11, lambda a: a[11].view(4, 4, 16, 4), "q_scales"),
    (2, lambda a: None, "k given: w, ln_weight"),
    (1, lambda a: a[1].float(), "k bf16"),
    (2, lambda a: a[2][:, :16], "w bf16"),
    (3, lambda a: a[3].float(), "ln_weight / ln_bias"),
    (6, lambda a: a[6].int(), "slot_mapping int64"),
    (12, lambda a: a[12].double(), "weights fp32"),
]


@pytest.mark.parametrize("i,make,text", PREP_BRANCHES, ids=[b[2][:30] for b in PREP_BRANCHES])
def test_prep_reasons(i, make, text):
    require_gfx950()
    args = prep_args(4, True, torch.Generator().manual_seed(0))
    why = E.indexer_prep_blocker(*with_(args, i, make(args)))
    assert why is not None and why.startswith("indexer_prep: ") and text in why, why
