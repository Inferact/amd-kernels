# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``NgramRows.map_tables`` on 8 GPUs: every rank keeps its shard of both layers' tables, peers read each other's
shards over HIP IPC, and each rank's rows (its own batch) equal ``reference.py`` bit for bit, eagerly and in a CUDA
graph replayed with new inputs, row-major and tiled codes; nothing compiles after the warm-up. Skipped with fewer
than 8 GPUs (``AKR_OVERSUBSCRIBE=1`` runs the 8 ranks on the GPUs there are: a plumbing check only)."""

import torch
import torch.distributed as dist
from amd_kernels.engram.ngram_rows_mxfp8_d256_c24_ranks8 import NgramRows
from amd_kernels.engram.ngram_rows_mxfp8_d256_c24_ranks8 import reference as ref
from tests.multi_gpu8 import run_on_8_gpus

from . import workload as wl

BUCKET = 4099
RANKS = 8


def worker(rank, errors):
    from flydsl.compiler.jit_function import MlirCompiler

    dev = torch.device("cuda", torch.cuda.current_device())
    primes, offsets, mult = wl.layout(BUCKET)
    tmap = wl.token_map(3)
    totals = [int(primes[layer].sum()) for layer in range(2)]
    tables = [wl.table(n, 20 + layer) for layer, n in enumerate(totals)]
    peer_rows = -(-max(totals) // RANKS)
    weights, scales = [], []
    for w, s in tables:
        wp = torch.zeros(peer_rows * RANKS, 256, dtype=torch.uint8)
        sp = torch.zeros(peer_rows * RANKS, 8, dtype=torch.uint8)
        wp[: w.shape[0]], sp[: s.shape[0]] = w, s
        weights.append(wp[rank * peer_rows:(rank + 1) * peer_rows].to(dev))
        scales.append(sp[rank * peer_rows:(rank + 1) * peer_rows].to(dev))
    rows = NgramRows.map_tables(
        weights, scales, rank, dist.group.WORLD, tmap.to(dev), mult.to(dev), primes.to(dev), offsets.to(dev),
        int(tmap[wl.PAD_ID_TOKEN]), peer_rows, image_id=wl.IMAGE_ID)
    rows.warm_up()
    full_w = [t[0].to(dev) for t in tables]
    full_s = [t[1].to(dev) for t in tables]

    def expect(batch, tile):
        ids, pos, qsl, lb = (t.to(dev) for t in batch[:4])
        return ref.ngram_rows_reference(
            ids, pos, qsl, lb, tmap.to(dev), mult.to(dev), primes.to(dev), offsets.to(dev), rows.pad_id, full_w, full_s,
            0, image_id=wl.IMAGE_ID, tile_rows=tile)

    def same(batch, m, tile):
        _, codes, shuf = expect(batch, tile)
        for layer in range(2):
            c, s = rows.activation(layer, m, tile)
            if not torch.equal(c.view(torch.uint8).flatten(), codes[layer].flatten()):
                errors.append(f"rank {rank}: codes layer {layer} m {m} tile {tile}")
            if not torch.equal(s.flatten(), shuf[layer]):
                errors.append(f"rank {rank}: scales layer {layer} m {m} tile {tile}")

    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        for m, requests, per in ((16, 2, 6), (130, 21, 6), (384, 64, 6), (768, 128, 6)):
            batch = wl.decode_batch(requests, per, m, 1000 * rank + m, slots=160)
            ins = [t.to(dev) for t in batch[:4]]
            for tile in (0, 96):
                rows(*ins, tile)
                torch.cuda.synchronize()
                same(batch, m, tile)
        batch = wl.decode_batch(64, 6, 384, 5 + rank, slots=160)
        ins = [t.to(dev) for t in batch[:4]]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            rows(*ins, 48)
        for seed in (6, 7):
            batch = wl.decode_batch(64, 6, 384, seed * 100 + rank, slots=160)
            for dst, src in zip(ins, batch[:4]):
                dst.copy_(src)
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            same(batch, 384, 48)
    finally:
        MlirCompiler.compile = orig
    dist.barrier()
    rows.close()


def test_peer_tables_over_ipc():
    run_on_8_gpus(worker, timeout=600)
