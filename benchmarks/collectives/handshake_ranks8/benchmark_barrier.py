# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of chained eight-rank hand-offs: steady-state us per call (512 vs 16 calls per graph: the difference
removes the replay's fixed start skew, which a short graph divides over its calls) and a table against the floor.
Variants: empty | lane | wave | pull | pullw [:ncta] (barriers), ex_ll | ex_flag | ex_pull :N[xC]
(exchange of N words per peer on C workgroups)."""
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
CALLS = (16, 512)
ROUNDS = 21
EMPTY_US = 1.55
FLAG_US = 0.8
EGRESS = 343e9
# measured phases: acknowledgement wait, landed-data read, remote-flag penalty
PHASES_US = {"ex_ll": 0.0, "ex_flag": 0.64 + 0.85, "ex_pull": 0.28 + 0.15 + 0.85}


def _worker(rank, port, variants):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    from amd_kernels import warm_up
    from amd_kernels.collectives.handshake_ranks8.host import barrier, exchange, make_comm

    warm_up.set_auto(False)

    dev = torch.device("cuda", rank)
    comm = make_comm(rank, dist.group.WORLD, dev)
    out = []
    for name in variants:
        v, _, n = name.partition(":")
        n, _, nc = n.partition("x")
        n, nc = int(n or 1), int(nc or 1)
        if v.startswith("ex_"):
            path = v[3:]
            xs = [torch.randint(0, 1 << 30, (8, n), dtype=torch.int32, device=dev) for _ in range(8)]
            ys = [torch.empty_like(xs[0]) for _ in range(8)]

            def call(i):
                exchange(comm, xs[i % 8], ys[i % 8], path, nc)
        else:
            def call(i):
                barrier(comm, v, n)
        call(0)
        torch.cuda.synchronize()
        dist.barrier()
        res = []
        for calls in CALLS:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(calls):
                    call(i)
            ts = []
            for rd in range(ROUNDS):
                g.replay()
                torch.cuda.synchronize()
                dist.barrier()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                g.replay()
                e.record()
                torch.cuda.synchronize()
                t = torch.tensor([s.elapsed_time(e) * 1000])
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                ts.append(t.item())
                dist.barrier()
            res.append(min(ts))
            del g
        per = (res[1] - res[0]) / (CALLS[1] - CALLS[0])
        out.append((name, per, res[0] / CALLS[0], res[0] - per * CALLS[0], 7 * n * 4 if v.startswith("ex_") else 0,
                    PHASES_US.get(v, 0.0)))
    if rank == 0:
        empty = next((o[1] for o in out if o[0].startswith("empty")), EMPTY_US)
        print(f"us per call, MAX over ranks, min of {ROUNDS} replays; floor = empty kernel + {FLAG_US} us flag one-way "
              f"+ the path's ack / read phases + egress bytes / {EGRESS / 1e9:.0f} GB/s")
        print(f"{'variant':16s} {'steady':>7s} {'+empty':>7s} {'egress KB':>10s} {'floor':>6s} {'floor/steady':>12s} "
              f"{'16-call avg':>11s} {'start skew':>10s}")
        for name, per, avg16, skew, nbytes, phases in out:
            floor = empty + FLAG_US + phases + nbytes / EGRESS * 1e6
            print(f"{name:16s} {per:7.2f} {per - empty:7.2f} {nbytes / 1024:10.1f} {floor:6.2f} {floor / per:12.0%} "
                  f"{avg16:11.2f} {skew:9.1f}us", flush=True)
    dist.barrier()


def main():
    variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ["empty", "lane", "wave", "pull"]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_worker, args=(port, variants), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
