# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the ``vocab_gather_comm`` entry adds on 8 GPUs (hidden 7168, 20480 vocabulary rows a shard): the entry
returns the package's ``VocabGather``, so a lookup runs the package's code. Graph replay us (20 lookups per graph,
each on its own ids; entry-built and directly built gathers alternate) and host us per eager call (the enqueue loop
of 100 calls, no synchronization inside), min of 20 rounds, the maximum over ranks; and the host us of
``vocab_gather_unsupported_reason`` (the caller runs it once per call).
Run: python3 benchmarks/embedding/vocab_gather_common/benchmark_routing.py
"""

import importlib
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H, VR, RANKS = 7168, 20480, 8
NIN, CALLS, ROUNDS = 20, 100, 20
MS = (1, 128, 2048)


def worker(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=RANKS)
    from amd_kernels.embedding.vocab_gather_common import vocab_gather_comm, vocab_gather_unsupported_reason
    from amd_kernels.embedding.vocab_gather_common.host import _SHAPE_HOSTS

    group = dist.group.WORLD
    dev = torch.device("cuda", rank)
    weight = torch.randn(VR, H, device=dev, dtype=torch.bfloat16)
    pkg = importlib.import_module(_SHAPE_HOSTS[(H, RANKS)])
    gathers = [pkg.VocabGather(weight, rank, group), vocab_gather_comm(weight, rank, group)]
    for g in gathers:
        g.warm_up()

    def rank_max(x):
        t = torch.tensor([x])
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
        return t.item()

    def replay_us(inputs):
        graphs = []
        for gather in gathers:
            gather(inputs[0])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for ids in inputs:
                    gather(ids)
            graphs.append(g)
        best = [float("inf")] * 2
        for _ in range(ROUNDS):
            for i, g in enumerate(graphs):
                dist.barrier(group)
                a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                e.record()
                e.synchronize()
                best[i] = min(best[i], a.elapsed_time(e) * 1000 / len(inputs))
        return [rank_max(b) for b in best]

    def host_us(fn, *args):
        best = float("inf")
        for _ in range(ROUNDS):
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(CALLS):
                fn(*args)
            best = min(best, (time.perf_counter() - t) * 1e6 / CALLS)
        torch.cuda.synchronize()
        return rank_max(best)

    if rank == 0:
        print("| tokens | replay direct us | replay entry us | host direct us | host entry us | added us | reason us |")
        print("|---|---|---|---|---|---|---|")
    gen = torch.Generator(device=dev).manual_seed(5)
    for m in MS:
        inputs = [torch.randint(0, RANKS * VR, (m,), device=dev, generator=gen) for _ in range(NIN)]
        assert vocab_gather_unsupported_reason(gathers[1], inputs[0]) is None
        rd, re = replay_us(inputs)
        hd, he = (host_us(g, inputs[0]) for g in gathers)
        hr = host_us(vocab_gather_unsupported_reason, gathers[1], inputs[0])
        if rank == 0:
            print(f"| {m} | {rd:.2f} | {re:.2f} | {hd:.2f} | {he:.2f} | {he - hd:+.2f} | {hr:.2f} |")
    for g in gathers:
        g.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(29600 + os.getpid() % 300,), nprocs=RANKS)
