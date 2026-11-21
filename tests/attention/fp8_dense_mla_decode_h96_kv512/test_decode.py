# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``fp8_dense_mla_decode`` against the contract in ``fp8_dense_mla_decode_h96_kv512/reference.py``.

Served shapes (README, "What it does"): 4 query rows (k = 4: c16, c24) and 3 (k = 3: c32, c48) per request, 96 heads,
this rank's shard of an 8-way round-robin context. Every case: each (row, head) against the fp64 reference (cosine
and relative error of the e4m3-P kernel), rows with no visible key (output 0, LSE -inf), two launches bit for bit
equal, and a CUDA graph replayed into NaN-filled outputs. The pool carries e4m3 NaN bytes in every row no request
lists (the kernel must not read them); the ragged cases have requests of 0 keys, as CUDA-graph padding does.
"""

import pytest
import torch
from amd_kernels.attention.fp8_dense_mla_decode_h96_kv512 import fp8_dense_mla_decode, supports_fp8_dense_mla_decode
from amd_kernels.attention.fp8_dense_mla_decode_h96_kv512.reference import HEADS, QK_DIM, fp8_dense_mla_decode_ref

Q_SCALE, KV_SCALE = 0.5, 0.25
SM_SCALE = 192**-0.5
FP8 = torch.float8_e4m3fn
COS_MIN = 0.998
REL_MAX = 0.07

# (id, query rows, local key counts, cp rank)
CASES = [
    ("c16_r8", 4, [3100, 2600, 4100, 2900, 3700, 2500, 3300, 4000], 0),
    ("c24_r13", 4, [2100, 3300, 1500, 2900, 2200, 4200, 3100, 2700, 2600, 3900, 1800, 2400, 3000], 3),
    ("c32_r20", 3, [2000 + 117 * i for i in range(20)], 7),
    ("c48_r48", 3, [700 + 53 * i for i in range(48)], 5),
    ("one_long", 4, [20000], 2),
    ("tiny", 4, [1, 2, 3, 64, 65, 129], 0),
    ("tiny_q3", 3, [1, 2, 63, 64, 66], 6),
    ("padding_tail", 4, [1500, 900, 0, 0, 0], 1),
    ("padding_mid", 3, [0, 500, 0, 70, 3000, 0, 9000], 4),
    ("all_empty", 3, [0, 0], 0),
    ("two_rows", 2, [1000, 1500], 0),
    ("one_row", 1, [1000, 1500], 0),
]
# (id, query rows, local key counts, cp rank, cp world)
WORLDS = [
    ("w1", 4, [1100, 700, 2000], 0, 1),
    ("w2", 3, [900, 1300], 1, 2),
    ("w4", 4, [64, 1700, 400], 3, 4),
]
NSETS = 85


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("needs gfx950")


def make_batch(qlen, lens, rank, seed=0, cp_world=8, block=64):
    """(q, pool, kv_indptr, kv_indices, g_kv_indptr): key tokens at random ``block``-token pages of a pool whose other
    rows are e4m3 NaN."""
    gen = torch.Generator().manual_seed(seed)
    n = len(lens)
    total = sum(lens)
    n_blocks = max(2, -(-total // block) + 3)
    pool = torch.full((n_blocks * block, QK_DIM), float("nan"), dtype=torch.float32)
    order = torch.randperm(n_blocks, generator=gen)
    idx = torch.empty(max(total, 1), dtype=torch.int32)
    pos = 0
    for b, ln in enumerate(lens):
        for j in range(ln):
            slot = int(order[(pos + j) // block]) * block + (pos + j) % block
            idx[pos + j] = slot
        pos += ln
    live = idx[:total].long()
    pool[live] = torch.randn(total, QK_DIM, generator=gen)
    q = torch.randn(n * qlen, HEADS, QK_DIM, generator=gen)
    ip = torch.zeros(n + 1, dtype=torch.int32)
    ip[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    glen = [(ln - 1) * cp_world + rank + 1 + int(torch.randint(0, cp_world, (1,), generator=gen)) if ln else 0 for ln in lens]
    gip = torch.zeros(n + 1, dtype=torch.int32)
    gip[1:] = torch.tensor(glen, dtype=torch.int32).cumsum(0)
    dev = "cuda"
    pool_e = pool.to(FP8)
    return q.to(FP8).to(dev), pool_e.to(dev), ip.to(dev), idx.to(dev), gip.to(dev)


_SCALES = {}


def scales():
    if "s" not in _SCALES:
        _SCALES["s"] = (torch.tensor([Q_SCALE], device="cuda"), torch.tensor([KV_SCALE], device="cuda"))
    return _SCALES["s"]


def launch(batch, qlen, rank, out=None, lse=None, world=8):
    q, pool, ip, idx, gip = batch
    qs, ks = scales()
    return fp8_dense_mla_decode(q, pool, ip, idx, gip, qs, ks, SM_SCALE, qlen, rank, world, out, lse)


def check(out, lse, batch, qlen, rank, world=8):
    q, pool, ip, idx, gip = batch
    ref_o, ref_l = fp8_dense_mla_decode_ref(q, pool, ip, idx, gip, Q_SCALE, KV_SCALE, SM_SCALE, qlen, world, rank)
    empty = ~torch.isfinite(ref_l)
    assert torch.equal(torch.isfinite(lse), ~empty)
    assert not torch.isnan(out.float()).any()
    assert (out.float()[empty] == 0).all()
    live = ~empty
    if live.any():
        o, r = out.float()[live], ref_o.float()[live]
        cos = torch.nn.functional.cosine_similarity(o, r, dim=-1)
        rel = (o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
        assert cos.min().item() >= COS_MIN, cos.min().item()
        assert rel.max().item() <= REL_MAX, rel.max().item()
        assert (lse[live] - ref_l[live]).abs().max().item() <= 2e-5


@pytest.mark.parametrize("name,qlen,lens,rank", CASES, ids=[c[0] for c in CASES])
@torch.inference_mode()
def test_decode(name, qlen, lens, rank):
    require_gfx950()
    batch = make_batch(qlen, lens, rank, seed=len(lens))
    out, lse = launch(batch, qlen, rank)
    out2, lse2 = launch(batch, qlen, rank)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), out2.view(torch.int16)) and torch.equal(lse, lse2)
    check(out, lse, batch, qlen, rank)


@pytest.mark.parametrize("name,qlen,lens,rank,world", WORLDS, ids=[c[0] for c in WORLDS])
@torch.inference_mode()
def test_cp_world(name, qlen, lens, rank, world):
    require_gfx950()
    batch = make_batch(qlen, lens, rank, seed=11, cp_world=world)
    out, lse = launch(batch, qlen, rank, world=world)
    check(out, lse, batch, qlen, rank, world)


@torch.inference_mode()
def test_cache_above_4_gib():
    """Key rows past the 32-bit byte offset of a 4 GiB cache."""
    require_gfx950()
    rows = (5 << 30) // QK_DIM
    free = torch.cuda.mem_get_info()[0]
    if free < (6 << 30):
        pytest.skip("needs 6 GiB free")
    batch = list(make_batch(4, [700, 900], 2, seed=5))
    small = batch[1]
    big = torch.zeros((rows, QK_DIM), dtype=FP8, device="cuda")
    n = small.shape[0]
    big[rows - n :] = small
    batch[1] = big
    batch[3] = batch[3] + (rows - n)
    out, lse = launch(batch, 4, 2)
    ref_batch = (batch[0], small, batch[2], batch[3] - (rows - n), batch[4])
    check(out, lse, ref_batch, 4, 2)


@pytest.mark.parametrize("qlen,lens,rank", [(4, [1200, 800, 0, 0], 3), (3, [3000, 40], 0)], ids=["q4", "q3"])
@torch.inference_mode()
def test_graph_replay(qlen, lens, rank):
    require_gfx950()
    batch = make_batch(qlen, lens, rank, seed=7)
    rows = len(lens) * qlen
    out = torch.full((rows, HEADS, 512), float("nan"), dtype=torch.bfloat16, device="cuda")
    lse = torch.full((rows, HEADS), float("nan"), device="cuda")
    launch(batch, qlen, rank, out, lse)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(batch, qlen, rank, out, lse)
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check(out, lse, batch, qlen, rank)


@torch.inference_mode()
def test_supports():
    require_gfx950()
    batch = make_batch(4, [100, 200], 0)
    q, pool, ip, idx, gip = batch
    qs, ks = scales()
    assert supports_fp8_dense_mla_decode(q, pool, ip, idx, gip, qs, ks, 4)
    assert not supports_fp8_dense_mla_decode(q, pool, ip, idx, gip, qs, ks, 3)
    assert not supports_fp8_dense_mla_decode(q.view(torch.uint8), pool, ip, idx, gip, qs, ks, 4)
    assert not supports_fp8_dense_mla_decode(q[:, :64], pool, ip, idx, gip, qs, ks, 4)
    assert not supports_fp8_dense_mla_decode(q, pool, ip.long(), idx, gip, qs, ks, 4)


@torch.inference_mode()
def test_zero_dim_scales():
    """vLLM passes the layer's 0-d ``_q_scale`` / ``_k_scale`` buffers: same result as the 1-element tensors."""
    require_gfx950()
    q, pool, ip, idx, gip = make_batch(4, [1500, 900], 0, seed=11)
    qs, ks = scales()
    o1, l1 = fp8_dense_mla_decode(q, pool, ip, idx, gip, qs, ks, SM_SCALE, 4, 0, 8)
    o0, l0 = fp8_dense_mla_decode(q, pool, ip, idx, gip, qs.reshape(()), ks.reshape(()), SM_SCALE, 4, 0, 8)
    torch.cuda.synchronize()
    assert torch.equal(o0, o1) and torch.equal(l0, l1)


@pytest.mark.parametrize("qlen,n", [(3, 64), (3, 16), (4, 48)])
@torch.inference_mode()
def test_all_empty_capture_dummies(qlen, n):
    """CUDA-graph capture dummies: every request has 0 keys (kv_indptr all 0) and kv_indices holds whatever the buffer
    held. The kernel must read neither kv_indices[-1] nor a KV row through a stale index (the served drafter faulted at
    c48 capture, q3), and every row is empty (output 0, LSE -inf)."""
    require_gfx950()
    n = min(n, 63)
    pool = torch.zeros(256, QK_DIM, dtype=FP8, device="cuda")
    big = torch.full((4096,), 0x7FFFFFF0, dtype=torch.int32, device="cuda")
    idx = big[1:]  # idx[-1] and every entry: an index far outside the pool
    ip = torch.zeros(n + 1, dtype=torch.int32, device="cuda")
    gip = torch.zeros(n + 1, dtype=torch.int32, device="cuda")
    q = torch.randn(n * qlen, HEADS, QK_DIM, device="cuda").to(FP8)
    qs, ks = scales()
    out, lse = fp8_dense_mla_decode(q, pool, ip, idx, gip, qs, ks, SM_SCALE, qlen, 0, 8)
    torch.cuda.synchronize()
    assert not torch.isfinite(lse).any()
    assert (out.float() == 0).all()
