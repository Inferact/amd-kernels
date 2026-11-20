# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks, the eager fc as the draft calls it: ``ConcatLinearComm`` (N-shard GEMM + slab reduce + shard exchange
into [rows, 6144]) on 3 rotated weight shards (566 MB) against what vLLM runs at TP2 (torch.cat + F.linear on the
replicated [6144, 30720] weight, 2 rotated copies, 755 MB). 20 distinct tap sets, 5 rounds of 20 eager calls
after a barrier, max over ranks, median us per call. ``--sweep`` times exchange forms (rows per group, push) instead.

    python3 benchmarks/gemm/bf16_concat_linear_n3072_k30720/benchmark_comm.py [--rows ...] [--sweep]
"""
import argparse
import statistics
import torch
import torch.distributed as dist
from amd_kernels.gemm.bf16_concat_linear_n3072_k30720 import pack_weight
from amd_kernels.gemm.bf16_concat_linear_n3072_k30720.comm import ConcatLinearComm, exchange_for
from amd_kernels.gemm.bf16_concat_linear_n3072_k30720.host import config_for
from tests.multi_gpu import run_on_2_gpus

CALLS = 20
XGMI = 64e9


def _timed(fn, rank):
    for i in range(3):
        fn(i)
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        dist.barrier()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for i in range(CALLS):
            fn(i)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / CALLS)
    every = [None] * 2
    dist.all_gather_object(every, times)
    return statistics.median(max(v[i] for v in every) for i in range(5))


def worker(rank, errors, rows, sweep):
    torch.set_num_threads(8)
    dev = torch.device('cuda', rank)
    g = torch.Generator(device=dev).manual_seed(0)
    full = [(torch.randn((6144, 30720), device=dev, generator=g) * 0.02).bfloat16() for _ in range(1 if sweep else 2)]
    comms = [ConcatLinearComm(pack_weight(full[0][rank * 3072:(rank + 1) * 3072].clone()), rank, dist.group.WORLD)
             for _ in range(3)]
    for c in comms:
        c.warm_up()
    for m in rows:
        xs = [[torch.randn((m, 6144), device=dev, generator=g).bfloat16() for _ in range(5)] for _ in range(CALLS)]
        outs = [torch.empty((m, 6144), device=dev, dtype=torch.bfloat16) for _ in range(CALLS)]
        if sweep:
            res = {}
            for push in (True, False):
                for rpw in (1, 2, 4, 8):
                    for cpg in (1, 2, 3, 6, 12):
                        if (m + rpw - 1) // rpw * cpg > 1024 or (rpw > 1 and (m + rpw - 1) // rpw * cpg < 64):
                            continue
                        res[(rpw, cpg, push)] = _timed(
                            lambda i, ex=(rpw, cpg, push): comms[i % 3](xs[i], outs[i], exchange_config=ex), rank)
            if rank == 0:
                best = sorted(res.items(), key=lambda kv: kv[1])[:10]
                print(f'rows {m}: ' + ' '.join(f"r{k[0]}p{k[1]}{'push' if k[2] else 'pull'}:{v:.1f}" for k, v in best), flush=True)
            continue
        ours = _timed(lambda i: comms[i % 3](xs[i], outs[i]), rank)
        base = _timed(lambda i: torch.nn.functional.linear(torch.cat(xs[i], 1), full[i % 2]), rank)
        hbm = (3072 * 30720 * 2 + m * 61440) / 6.5e12
        floor = (max(hbm, 2 * m * 3072 * 30720 / 2.5e15) + m * 3072 * 2 / XGMI) * 1e6
        if rank == 0:
            print(f'rows {m}: replicated cat + F.linear {base:6.1f} us | ConcatLinearComm {ours:6.1f} us | '
                  f'{base / ours:.2f}x | serial roofline {floor:5.1f} us ({floor / ours:.0%}) | tier {config_for(m)} '
                  f'exchange {exchange_for(m)}', flush=True)
    for c in comms:
        c.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, nargs='+', default=[8, 56, 80, 128, 256, 448, 512, 1024, 2048])
    p.add_argument('--sweep', action='store_true')
    a = p.parse_args()
    run_on_2_gpus(worker, a.rows, a.sweep, timeout=3000)
