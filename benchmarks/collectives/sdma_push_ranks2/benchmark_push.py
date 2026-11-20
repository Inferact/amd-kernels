# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``SdmaPush`` on two ranks vs RCCL all-gather at world 2, the same bytes per rank (an SP layer's hand-offs at T
tokens: R = T / 2 shard rows of the a2a 4224 + compact 7328 + reduce-scatter 12288 + codes 6336 B).

Both ranks push their block into the peer's region and wait for the peer's (an all-gather's link traffic); the pushes
alternate between two regions with a wait in between. Eager (stream memory operations, no graph capture): stream
events around 10 back-to-back push + wait pairs, 5 rounds, the max over ranks, median over rounds; us per push. Also a
bf16 GEMM loop as long as the pushes: its time alone and with the pushes running on another stream.

  python3 benchmarks/collectives/sdma_push_ranks2/benchmark_push.py [out.json] [--tokens T ...]
"""
import json
import statistics
import sys

import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_2_gpus

ROW_BYTES = 4224 + 7328 + 12288 + 6336
ROUNDS = 5
N = 10


def _round_time(fn):
    dist.barrier()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    fn()
    b.record()
    torch.cuda.synchronize()
    every = [None, None]
    dist.all_gather_object(every, a.elapsed_time(b) * 1000 / N)
    return max(every)


def _worker(rank, errors, tokens, out):
    from amd_kernels.collectives.sdma_push_ranks2 import SdmaPush

    dev = torch.device("cuda", rank)
    nccl = dist.new_group(backend="nccl")
    peer = 1 - rank
    max_n = max(tokens) // 2 * ROW_BYTES
    t = SdmaPush(rank, dist.group.WORLD, dev, [2 * max_n, 2 * max_n])
    src = torch.randint(0, 255, (max_n,), dtype=torch.uint8, device=dev)
    gathered = torch.empty((2 * max_n,), dtype=torch.uint8, device=dev)
    a = torch.randn((8192, 6144), device=dev).bfloat16()
    bmat = torch.randn((6144, 6144), device=dev).bfloat16()
    side = torch.cuda.Stream()
    results = {}
    for T in tokens:
        n = T // 2 * ROW_BYTES

        def sdma():
            for i in range(N):
                seq = t.push(i % 2, [(peer, src.data_ptr(), rank * n, n)])
                t.wait(i % 2, seq)

        def rccl():
            for _ in range(N):
                dist.all_gather_into_tensor(gathered[:2 * n], src[:n], group=nccl)

        arms = {"sdma": sdma, "rccl_ag": rccl}
        for fn in arms.values():
            fn()
        torch.cuda.synchronize()
        times = {k: [] for k in arms}
        for r in range(ROUNDS):
            for k in (list(arms) if r % 2 == 0 else list(reversed(arms))):
                times[k].append(_round_time(arms[k]))
        got = {k: statistics.median(v) for k, v in times.items()}
        # a GEMM loop as long as the pushes, alone and with the pushes on another stream (its own events)
        g0, g1 = torch.cuda.Event(True), torch.cuda.Event(True)
        a @ bmat
        g0.record()
        a @ bmat
        g1.record()
        torch.cuda.synchronize()
        iters = max(1, int(got["sdma"] * N / (g0.elapsed_time(g1) * 1000)))
        gemm_t = {"alone": [], "next_to_sdma": []}
        for r in range(ROUNDS):
            for kind in gemm_t:
                dist.barrier()
                torch.cuda.synchronize()
                if kind == "next_to_sdma":
                    with torch.cuda.stream(side):
                        sdma()
                g0.record()
                for _ in range(iters):
                    a @ bmat
                g1.record()
                torch.cuda.synchronize()
                gemm_t[kind].append(g0.elapsed_time(g1) * 1000)
        got.update({f"gemm_{k}": statistics.median(v) for k, v in gemm_t.items()})
        got["GBs"] = n / got["sdma"] / 1e3
        got["MB"] = n / 1e6
        results[T] = got
        if rank == 0:
            print(f"T={T:5d} {n / 1e6:7.1f} MB/rank: sdma {got['sdma']:8.1f} us ({got['GBs']:.1f} GB/s)  "
                  f"rccl all-gather {got['rccl_ag']:8.1f} us  gemm loop alone {got['gemm_alone']:8.1f} us, "
                  f"next to the pushes {got['gemm_next_to_sdma']:8.1f} us", flush=True)
    t.drain()
    if rank == 0 and out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)


def main():
    args = sys.argv[1:]
    out = args.pop(0) if args and args[0].endswith(".json") else None
    tokens = [4096, 8192, 16384, 32768]
    if "--tokens" in args:
        tokens = [int(x) for x in args[args.index("--tokens") + 1:]]
    run_on_2_gpus(_worker, tokens, out, timeout=3600)


if __name__ == "__main__":
    main()
