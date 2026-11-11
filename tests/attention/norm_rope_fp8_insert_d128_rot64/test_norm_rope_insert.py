# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Both kernels against the contract in ``norm_rope_fp8_insert_d128_rot64/reference.py``.

Served shapes (README): an M3.1 draft layer at TP4 (16 query heads, 1 KV head, 5
layers, 128-token pages). The qkv kernel sees the draft's query block (R requests x 4
rows); the multi-layer kernel the context rows of a step (the accepted tokens in decode,
a prefill chunk). Every case: q and every cache byte against the reference,
the bytes the kernel must not write unchanged (skipped slots, other tokens), three
launches equal bit for bit, and a CUDA graph replayed on refilled outputs.
"""

import functools

import pytest
import torch
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import (
    kv_norm_rope_insert_multilayer,
)
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import host as ml_host
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.qkv_norm_rope_insert import (
    qkv_norm_rope_insert,
    supports_qkv_norm_rope_insert,
)
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.reference import (
    D,
    check_bf16,
    check_cache,
    kv_norm_rope_insert_multilayer_reference,
    qkv_norm_rope_insert_reference,
)

EPS = 1e-6  # the draft's rms_norm_eps
CS_ROWS = 4096
LAYERS = 5  # draft layers


def require_gfx950():
    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        pytest.skip("requires gfx950")


def rope_cache(gen, rows=CS_ROWS):
    ang = torch.rand((rows, 32), generator=gen, dtype=torch.float64) * 6.3
    return torch.cat([ang.cos(), ang.sin()], dim=1).bfloat16().cuda()


@functools.cache
def scale_pair(scale):
    """Device scalars, made once (a CUDA graph cannot copy them in)."""
    return torch.tensor(scale, device="cuda"), torch.tensor(scale * 1.3, device="cuda")


def graph_replay(fn, refill):
    """Capture ``fn``, refill its outputs, replay twice (yields after each replay)."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(2):
        refill()
        graph.replay()
        torch.cuda.synchronize()
        yield


# (id, tokens, query heads, KV heads, page, scale, slots given (None: all), insert)
QKV_CASES = [
    ("tp4_t1_p128", 1, 16, 1, 128, 1.0, None, True),
    ("tp4_t4_p128", 4, 16, 1, 128, 1.0, None, True),
    ("tp4_t1000_p128", 1000, 16, 1, 128, 0.37, None, True),
    ("tp4_t384_p128_fewer_slots", 384, 16, 1, 128, 1.0, 300, True),
    ("tp4_t384_noinsert", 384, 16, 1, 128, 1.0, None, False),
]


@pytest.mark.parametrize("name,n,nq,nkv,page,scale,n_slots,insert", QKV_CASES, ids=[c[0] for c in QKV_CASES])
@torch.inference_mode()
def test_qkv_norm_rope_insert(name, n, nq, nkv, page, scale, n_slots, insert):
    require_gfx950()
    gen = torch.Generator().manual_seed(n * 10 + nq)
    n_pages = (2 * n + page - 1) // page + 2
    qkv = (torch.randn((n, (nq + 2 * nkv) * D), generator=gen) * 2).bfloat16().cuda()
    pos = torch.randint(0, CS_ROWS, (n,), generator=gen).cuda()
    pos[::5] -= CS_ROWS  # negative positions wrap
    slots = torch.randperm(n_pages * page, generator=gen)[: n_slots or n].cuda()
    slots[::7] = -1
    qw = (torch.randn(D, generator=gen) * 0.5).bfloat16().cuda()
    kw = (torch.randn(D, generator=gen) * 0.5).bfloat16().cuda()
    cs = rope_cache(gen)
    ks, vs = scale_pair(scale)
    cache0 = torch.randint(0, 256, (n_pages, nkv, page, 2 * D), generator=gen, dtype=torch.uint8).cuda()
    if insert:
        assert supports_qkv_norm_rope_insert(qkv, cs, cache0)
    args = (qkv, pos, qw, kw, cs, EPS, nq, nkv)
    q_ref, ins = qkv_norm_rope_insert_reference(
        *args, cache0 if insert else None, slots, ks, vs
    )

    caches = [cache0.clone() for _ in range(3)]
    qs = [torch.full((n, nq * D), float("nan"), dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    for q, cache in zip(qs, caches):
        if insert:
            qkv_norm_rope_insert(*args, q, cache, slots, ks, vs)
        else:
            qkv_norm_rope_insert(*args, q)
    torch.cuda.synchronize()
    frac_q = check_bf16(qs[0], q_ref, name)
    frac_c = check_cache(caches[0], ins[0], cache0, ins[1], name) if insert else 1.0
    print(f"{name}: q bit-equal {frac_q:.6f}, cache bit-equal {frac_c:.6f}")
    for q, cache in zip(qs[1:], caches[1:]):
        assert torch.equal(q.view(torch.int16), qs[0].view(torch.int16)), name
        assert torch.equal(cache, caches[0]), name

    q, cache = qs[1], caches[1]

    def run():
        if insert:
            qkv_norm_rope_insert(*args, q, cache, slots, ks, vs)
        else:
            qkv_norm_rope_insert(*args, q)

    def refill():
        q.fill_(float("nan"))
        cache.copy_(cache0)

    for _ in graph_replay(run, refill):
        assert torch.equal(q.view(torch.int16), qs[0].view(torch.int16)), name
        assert torch.equal(cache, caches[0]), name


# (id, context rows, KV heads, page, per-layer slot mappings, rows inserted (None: all))
ML_CASES = [
    ("tp4_n7_p128", 7, 1, 128, False, None),
    ("tp4_n384_p128", 384, 1, 128, False, None),
    ("tp4_n4096_p128", 4096, 1, 128, True, 4000),
    ("tp4_n100_p128_dummy", 100, 1, 128, False, 0),
]


def make_multilayer(n, nkv, page, layer_slots, gen):
    n_pages = (2 * n + page - 1) // page + 2
    kv = (torch.randn((n, LAYERS * 2 * nkv * D), generator=gen) * 2).bfloat16().cuda()
    pos = torch.randint(0, CS_ROWS, (n,), generator=gen).cuda()
    weights = (torch.randn((LAYERS, D), generator=gen) * 0.5).bfloat16().cuda()
    cs = rope_cache(gen)
    shared = torch.randperm(n_pages * page, generator=gen)[:n].cuda()
    shared[::7] = -1
    if layer_slots:
        slots = [torch.randperm(n_pages * page, generator=gen)[:n].cuda() for _ in range(LAYERS)]
        for s in slots:
            s[::5] = -1
    else:
        slots = [shared] * LAYERS
    caches0 = [
        torch.randint(0, 256, (n_pages, nkv, page, 2 * D), generator=gen, dtype=torch.uint8).cuda()
        for _ in range(LAYERS)
    ]
    ks = [torch.tensor(0.75 + 0.1 * i, device="cuda") for i in range(LAYERS)]
    vs = [torch.tensor(1.25 - 0.1 * i, device="cuda") for i in range(LAYERS)]
    return kv, pos, weights, cs, caches0, slots, ks, vs


@pytest.mark.parametrize("name,n,nkv,page,layer_slots,n_insert", ML_CASES, ids=[c[0] for c in ML_CASES])
@torch.inference_mode()
def test_kv_norm_rope_insert_multilayer(name, n, nkv, page, layer_slots, n_insert):
    require_gfx950()
    gen = torch.Generator().manual_seed(n + nkv)
    kv, pos, weights, cs, caches0, slots, ks, vs = make_multilayer(n, nkv, page, layer_slots, gen)
    refs = kv_norm_rope_insert_multilayer_reference(
        kv, pos, weights, cs, caches0, slots, ks, vs, EPS, n_insert
    )
    runs = [[c.clone() for c in caches0] for _ in range(3)]
    for caches in runs:
        kv_norm_rope_insert_multilayer(kv, pos, weights, cs, caches, slots, ks, vs, EPS, n_insert=n_insert)
    torch.cuda.synchronize()
    fracs = [
        check_cache(got, ref, before, mask, f"{name} layer {i}")
        for i, (got, (ref, mask), before) in enumerate(zip(runs[0], refs, caches0))
    ]
    print(f"{name}: cache bit-equal (per layer) {', '.join(f'{f:.6f}' for f in fracs)}")
    for caches in runs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(caches, runs[0])), name

    caches = runs[1]

    def run():
        kv_norm_rope_insert_multilayer(kv, pos, weights, cs, caches, slots, ks, vs, EPS, n_insert=n_insert)

    def refill():
        for c, c0 in zip(caches, caches0):
            c.copy_(c0)

    for _ in graph_replay(run, refill):
        assert all(torch.equal(a, b) for a, b in zip(caches, runs[0])), name


@torch.inference_mode()
def test_multilayer_pointer_table_cache():
    """The per-layer cache / scale pointer table is cached by
    the persistent pointers only; new eager slot views at other offsets reuse it."""
    require_gfx950()
    gen = torch.Generator().manual_seed(3)
    kv, pos, weights, cs, caches, slots, ks, vs = make_multilayer(68, 2, 128, False, gen)
    kv_norm_rope_insert_multilayer(kv, pos, weights, cs, caches, slots, ks, vs, EPS)
    misses = ml_host._pointer_table.cache_info().misses
    for offset in (1, 3):
        views = [
            torch.cat([torch.zeros(offset, device="cuda", dtype=s.dtype), s])[offset:]
            for s in slots
        ]
        kv_norm_rope_insert_multilayer(kv, pos, weights, cs, caches, views, ks, vs, EPS)
        assert ml_host._pointer_table.cache_info().misses == misses


def test_supports_multilayer():
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import (
        supports_multilayer,
    )

    def caches(n=LAYERS, shape=(4, 1, 128, 256), dtype=torch.uint8):
        return [torch.zeros(shape, dtype=dtype) for _ in range(n)]

    scales = [torch.ones(1) for _ in range(LAYERS)]
    assert supports_multilayer(caches(), scales, scales)
    assert supports_multilayer(caches(dtype=torch.float8_e4m3fn), scales, scales)
    strided = torch.zeros(4, 1, 128, 512, dtype=torch.uint8)[..., :256]
    half = [torch.ones(1, dtype=torch.float16)] * LAYERS
    for bad, ks in (
        ([], []),
        (caches(dtype=torch.bfloat16), scales),
        (caches(shape=(4, 1, 128, 128)), scales),
        (caches(shape=(0, 1, 128, 256)), scales),
        (caches(LAYERS - 1) + caches(1, (4, 2, 128, 256)), scales),
        (caches(LAYERS - 1) + [strided], scales),
        (caches(), half),
        (caches(), scales[:-1]),
    ):
        assert not supports_multilayer(bad, ks, scales[: len(bad)])


def test_rope_blocker():
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64 import rope_blocker

    cs = torch.empty(4096, 64, dtype=torch.bfloat16)
    assert rope_blocker(64, True, cs) is None
    for rot, neox, c in ((32, True, cs), (64, False, cs), (64, True, cs.float()),
                         (64, True, torch.empty(4096, 128, dtype=torch.bfloat16)),
                         (64, True, torch.empty(64, 4096, dtype=torch.bfloat16).T)):
        assert rope_blocker(rot, neox, c) is not None, (rot, neox, tuple(c.shape))
