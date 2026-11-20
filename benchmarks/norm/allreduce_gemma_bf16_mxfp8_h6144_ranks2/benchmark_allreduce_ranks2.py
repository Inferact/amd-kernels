# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One row count per two-rank process set; distinct inputs beyond the 256 MB MALL, graphs of at most 256 calls,
alternated MAX-rank rounds. Baseline: AITER's custom all-reduce, the one-GPU bf16 norm, then (MXFP8) the FlyDSL
quantizer of its output (row-major codes). Arms: the all-row form and the owned-residual form (owner from
``host.OWNER_MIN_ROWS``); ``--sweep``: the (rows per batch, stagers, reducers, receivers) candidates of both forms.

    python3 benchmarks/norm/allreduce_gemma_bf16_mxfp8_h6144_ranks2/benchmark_allreduce_ranks2.py [--rows ...]
        [--tile 16] [--rounds 7] [--sweep]
"""
import argparse
import json
import statistics

import torch
import torch.distributed as dist

H = 6144
LLC_BYTES = 400 << 20
MAX_GRAPH_CALLS = 256


def _candidates(rows, owner):
    out = []
    q = (rows + 1) // 2 if owner else rows
    for rpb in (1, 2, 4, 8):
        nb = -(-q // rpb) * (2 if owner else 1)
        for nstage in (32, 64, 128, 256):
            for nred in (32, 64, 128, 256):
                c = (rpb, min(nb, nstage), min(nb, nred), min(nb, nred) if owner else 0)
                if c not in out and sum(c[1:]) <= 768:
                    out.append(c)
    return out


def worker(rank, errors, m, tile, rounds, sweep, owner_min=None):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_e4m3_quantize_flydsl
    from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144_ranks2 import Bf16ArNormComm, host
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an

    if owner_min is not None:
        host.OWNER_MIN_ROWS = owner_min
    dev = torch.device('cuda', rank)
    ca = CustomAllreduce(group=dist.group.WORLD, device=dev, max_size=64 * 1024 * 1024)
    comm = Bf16ArNormComm(rank, dist.group.WORLD, dev)
    w = (torch.randn(H, generator=torch.Generator().manual_seed(5)) * 0.2).bfloat16().to(dev)
    comm.warm_up(w)
    n = max(8, min(2048, -(-LLC_BYTES // (2 * m * H * 2))))
    if sweep:
        n = min(n, 24)
    g = torch.Generator(device=dev).manual_seed(12000 + rank)
    xs = [torch.randn(m, H, generator=g, device=dev).bfloat16() for _ in range(n)]
    rs = [torch.randn(m, H, generator=g, device=dev).bfloat16() for _ in range(n)]

    def old(i):
        a = ca.custom_all_reduce(xs[i])
        y, res = an.gemma_add_rmsnorm(a, rs[i], w, 1e-6)
        if tile < 0:
            return y, res
        return (*mxfp8_e4m3_quantize_flydsl(y), res)

    def new(owner):
        def fn(i):
            if tile < 0:
                return comm.ar_add_rmsnorm(xs[i], rs[i], w, owned_residual=owner)
            return comm.ar_add_rmsnorm_mxfp8(xs[i], rs[i], w, 1e-6, tile, owned_residual=owner)
        return fn

    arms = {'base': old, 'all_row': new(False)}
    if m >= host.OWNER_MIN_ROWS:
        arms['owner'] = new(True)
    if sweep:
        arms = {}
        for owner in (False, True):
            for cfg in _candidates(m, owner):
                def fn(i, cfg=cfg, owner=owner):
                    comm.config = lambda *a, cfg=cfg: cfg
                    saved = host.OWNER_MIN_ROWS
                    host.OWNER_MIN_ROWS = 1
                    try:
                        return new(owner)(i)
                    finally:
                        comm.config = None
                        host.OWNER_MIN_ROWS = saved
                arms[('owner' if owner else 'all_row',) + cfg] = fn
    for fn in arms.values():
        fn(0)
    torch.cuda.synchronize()
    graphs = {}
    for k, fn in arms.items():
        gs = []
        for lo in range(0, n, MAX_GRAPH_CALLS):
            gr = torch.cuda.CUDAGraph()
            with ca.capture(), comm.capture(), torch.cuda.graph(gr):
                for i in range(lo, min(n, lo + MAX_GRAPH_CALLS)):
                    fn(i)
            gs.append(gr)
        graphs[k] = gs
    for gs in graphs.values():
        for gr in gs:
            gr.replay()
    torch.cuda.synchronize()
    ts = {k: [] for k in graphs}
    for r in range(rounds):
        for k in (list(graphs) if r % 2 == 0 else list(reversed(list(graphs)))):
            dist.barrier()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(2):
                for gr in graphs[k]:
                    gr.replay()
            b.record()
            b.synchronize()
            ts[k].append(a.elapsed_time(b) * 1e3 / (2 * n))
    every = [None, None]
    dist.all_gather_object(every, ts)
    if rank == 0:
        vals = {k: [max(v[k][j] for v in every) for j in range(rounds)] for k in ts}
        med = {str(k): statistics.median(v) for k, v in vals.items()}
        if sweep:
            best = sorted(med.items(), key=lambda kv: kv[1])
            print(json.dumps(dict(rows=m, tile=tile, best=best[:4] + [kv for kv in best if 'all_row' in kv[0]][:3])),
                  flush=True)
        else:
            print(json.dumps(dict(rows=m, tile=tile, sets=n, medians=med)), flush=True)
    comm.close()
    ca.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rows', nargs='+', type=int, default=[7, 56, 70, 112, 224, 448, 896, 1792])
    p.add_argument('--tile', type=int, default=16, help='code tile rows; 0 row-major; -1 bf16 out')
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--owner-min', type=int, help='override host.OWNER_MIN_ROWS (the owner arm from this row count)')
    a = p.parse_args()
    from tests.multi_gpu import run_on_2_gpus
    for m in a.rows:
        run_on_2_gpus(worker, m, a.tile, a.rounds, a.sweep, a.owner_min, timeout=3600)
