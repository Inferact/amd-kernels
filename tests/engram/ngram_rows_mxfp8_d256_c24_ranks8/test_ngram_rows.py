# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``NgramRows`` against ``reference.py``, bit for bit (codes and shuffled scales of both layers), on one GPU with the
eight shards as eight allocations the kernel reads by pointer: decode batches of the served rows 16-768 (1-6 rows per
request, idle request slots, CUDA-graph padding rows, image tokens, sequence starts under the lookback), int32 / int64
ids and positions, graph replay with new inputs, repeat runs, nothing compiling after the warm-up. The eight-GPU
mapping over HIP IPC is in ``test_ngram_rows_ranks8.py``."""

import pytest
import torch
from amd_kernels.engram import ngram_rows_mxfp8_d256_c24_ranks8 as pkg
from amd_kernels.engram.ngram_rows_mxfp8_d256_c24_ranks8 import reference as ref

from . import workload as wl

BUCKET = 4099
RANKS = 8


def require_gfx950():
    if not pkg.available():
        pytest.skip("requires gfx950 with FlyDSL")


@pytest.fixture(scope="module")
def setup():
    require_gfx950()
    dev = torch.device("cuda")
    primes, offsets, mult = wl.layout(BUCKET)
    tmap = wl.token_map(3)
    totals = [int(primes[layer].sum()) for layer in range(2)]
    tables = [wl.table(n, 20 + layer) for layer, n in enumerate(totals)]
    peer_rows = -(-max(totals) // RANKS)
    shards = []
    for layer, (w, s) in enumerate(tables):
        pad_rows = peer_rows * RANKS
        wp = torch.zeros(pad_rows, 256, dtype=torch.uint8)
        sp = torch.zeros(pad_rows, 8, dtype=torch.uint8)
        wp[: w.shape[0]], sp[: s.shape[0]] = w, s
        shards.append(([wp[q * peer_rows:(q + 1) * peer_rows].to(dev) for q in range(RANKS)],
                       [sp[q * peer_rows:(q + 1) * peer_rows].to(dev) for q in range(RANKS)]))
    rows = pkg.NgramRows(
        tmap.to(dev), mult.to(dev), primes.to(dev), offsets.to(dev), int(tmap[wl.PAD_ID_TOKEN]), peer_rows,
        [[t.data_ptr() for t in sh[0]] for sh in shards], [[t.data_ptr() for t in sh[1]] for sh in shards],
        image_id=wl.IMAGE_ID, keepalive=shards)
    rows.warm_up()
    return rows, tables, (tmap, mult, primes, offsets), dev


def expected(setup, ids, pos, qsl, lb, tile_rows=0):
    rows, tables, (tmap, mult, primes, offsets), dev = setup
    return ref.ngram_rows_reference(
        ids.to(dev), pos.to(dev), qsl.to(dev), lb.to(dev), tmap.to(dev), mult.to(dev), primes.to(dev),
        offsets.to(dev), rows.pad_id, [t[0].to(dev) for t in tables], [t[1].to(dev) for t in tables], 0,
        image_id=wl.IMAGE_ID, tile_rows=tile_rows)


def check(setup, ids, pos, qsl, lb, m, tile_rows=0):
    rows = setup[0]
    _, codes, shuf = expected(setup, ids, pos, qsl, lb, tile_rows)
    rows.codes.fill_(0xAA)  # the padding rows must be written, not left over
    rows.scales.fill_(0xAA)
    rows(ids.to(rows.device), pos.to(rows.device), qsl.to(rows.device), lb.to(rows.device), tile_rows)
    torch.cuda.synchronize()
    for layer in range(2):
        c, s = rows.activation(layer, m, tile_rows)
        assert torch.equal(c.view(torch.uint8).flatten(), codes[layer].flatten()), ("codes", layer, m, tile_rows)
        assert torch.equal(s.flatten(), shuf[layer]), ("scales", layer, m, tile_rows)


CASES = [(16, 1, 16, 0, 16), (10, [1, 6, 3], 32, 0, 12), (21, 6, 128, 0, 24), (32, 6, 192, 7, 40), (64, 6, 384, 0, 70),
         (128, 6, 768, 11, 140), (3, 6, 20, 0, 5), (100, 6, 600, 0, 128)]


@pytest.mark.parametrize("requests,per,rows,image_every,slots", CASES)
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("tile_rows", [0, 16, 48, 96, 128])
def test_matches_reference(setup, requests, per, rows, image_every, slots, wide, tile_rows):
    ids, pos, qsl, lb, _ = wl.decode_batch(requests, per, rows, 100 + rows, image_every=image_every, slots=slots)
    if wide:
        ids, pos = ids.to(torch.int64), pos.to(torch.int64)
    else:
        pos = pos.to(torch.int32)
    check(setup, ids, pos, qsl, lb, rows, tile_rows)


def test_prefill_chunk(setup):
    ids, pos, qsl, lb, _ = wl.decode_batch(3, [200, 77, 300], 640, 5, slots=8)
    check(setup, ids, pos, qsl, lb, 640)


def test_graph_replay_and_repeat(setup):
    rows = setup[0]
    ids, pos, qsl, lb, _ = wl.decode_batch(64, 6, 384, 1, slots=80)
    ins = [t.to(rows.device) for t in (ids, pos, qsl, lb)]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rows(*ins)
    for seed in (2, 3):
        new = wl.decode_batch(64, 6, 384, seed, slots=80)[:4]
        for dst, src in zip(ins, new):
            dst.copy_(src)
        rows.codes.fill_(7)
        rows.scales.fill_(9)
        g.replay()
        torch.cuda.synchronize()
        _, codes, shuf = expected(setup, *new)
        for layer in range(2):
            c, s = rows.activation(layer, 384)
            assert torch.equal(c.view(torch.uint8), codes[layer])
            assert torch.equal(s.flatten(), shuf[layer])
    first = rows.codes.clone()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(first, rows.codes)


def test_no_compile_after_warm_up(setup):
    from flydsl.compiler.jit_function import MlirCompiler

    rows = setup[0]
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        for requests, per, pad, _, slots in CASES[:4]:
            ids, pos, qsl, lb, _ = wl.decode_batch(requests, per, pad, 9, slots=slots)
            rows(ids.to(rows.device), pos.to(rows.device).to(torch.int32), qsl.to(rows.device), lb.to(rows.device))
            rows(ids.to(torch.int64).to(rows.device), pos.to(rows.device), qsl.to(rows.device), lb.to(rows.device))
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig
