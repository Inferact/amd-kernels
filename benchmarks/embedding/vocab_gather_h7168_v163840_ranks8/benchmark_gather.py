# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank embedding latency vs vLLM's VocabParallelEmbedding (mask / gather / fill + AITER all-reduce).

Graph replay of 20 distinct token batches per M, nine rounds (minimum), MAX over ranks, us. Ceilings
(measured): the balanced floor is one launch gap (1.6 us), one remote read round trip (1.0 us) and 7/8 of the
rows (M x 14336 B) over the measured 365 GB/s per-GPU ingress of an 8-GPU all-read; the critical-path model adds what the
data fixes: the local part (the kernel on rows of its own shard), one remote latency of 1.5 us and the busiest owner's rows
(every rank reads the same ids, so the link of the owner holding the most rows of the batch is the longest) at the 52 GB/s
per link of an 8-GPU all-read. `--ids balanced` draws the owners round robin (no imbalance), `--ids random` is uniform.
`% ceiling` = balanced floor / this entry, `% model` = model / this entry.
"""
import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

H, VR, RANKS = 7168, 20480, 8
INGRESS_GBS = 365.0


LINK_GBS = 52.0
REMOTE_US = 1.5


def make_ids(kind, m, dev, gen):
    if kind == "random":
        return torch.randint(0, RANKS * VR, (m,), device=dev, generator=gen)
    owners = torch.arange(m, device=dev) % RANKS
    return owners * VR + torch.randint(0, VR, (m,), device=dev, generator=gen)


def worker(rank, ms, port, kind):
    torch.cuda.set_device(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tp_group, graph_capture
    from vllm.model_executor.layers.vocab_parallel_embedding import get_masked_input_and_mask
    from amd_kernels.embedding.vocab_gather_h7168_v163840_ranks8 import VocabGather

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(RANKS, rank, f"tcp://127.0.0.1:{port}", rank, "nccl")
        initialize_model_parallel(tensor_model_parallel_size=RANKS)
        group = get_tp_group().cpu_group
        dev = torch.device("cuda", rank)
        torch.manual_seed(rank)
        weight = torch.randn(VR, H, device=dev, dtype=torch.bfloat16)
        gather = VocabGather(weight, rank, group)
        gather.warm_up()

        def stock(ids):
            masked, mask = get_masked_input_and_mask(ids, rank * VR, (rank + 1) * VR, 0, 0, 0)
            out = F.embedding(masked, weight)
            out.masked_fill_(mask.unsqueeze(-1), 0)
            return tensor_model_parallel_all_reduce(out)

        def timed(fn, inputs):
            g = torch.cuda.CUDAGraph()
            fn(inputs[0])
            torch.cuda.synchronize()
            with graph_capture(device=dev) as ctx:
                with torch.cuda.graph(g, stream=ctx.stream):
                    for ids in inputs:
                        fn(ids)
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(9):
                dist.barrier(group)
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for _ in range(10):
                    g.replay()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000 / (10 * len(inputs)))
            t = torch.tensor([min(ts)], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return t.item()

        if rank == 0:
            print(f"{'M':>5} {'stock us':>9} {'this us':>8} {'speedup':>8} {'GB/s':>7} {'ceiling us':>10} {'% ceiling':>9} "
                  f"{'local us':>8} {'max rows':>8} {'model us':>9} {'% model':>8}")
        for m in ms:
            gen = torch.Generator(device=dev).manual_seed(5 + m)
            inputs = [make_ids(kind, m, dev, gen) for _ in range(20)]
            a, b = timed(stock, inputs), timed(gather, inputs)
            local = timed(gather, [rank * VR + torch.randint(0, VR, (m,), device=dev, generator=gen) for _ in range(20)])
            busiest = sum(int(torch.bincount(ids // VR, minlength=RANKS).max()) for ids in inputs) / len(inputs)
            model = local + REMOTE_US + busiest * H * 2 / (LINK_GBS * 1e3)
            by = m * H * 2 * 2
            floor = 1.6 + 1.0 + m * H * 2 * (RANKS - 1) / RANKS / (INGRESS_GBS * 1e3)
            if rank == 0:
                print(f"{m:>5} {a:>9.1f} {b:>8.1f} {a / b:>7.2f}x {by / b / 1e3:>7.1f} {floor:>10.1f} {100 * floor / b:>8.0f}% "
                      f"{local:>8.1f} {busiest:>8.1f} {model:>9.1f} {100 * model / b:>8.0f}%")
        gather.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ms", type=int, nargs="*", default=[1, 12, 24, 40, 65, 80, 128, 192, 256])
    p.add_argument("--ids", choices=("random", "balanced"), default="random")
    a = p.parse_args()
    mp.spawn(worker, args=(a.ms, 29600 + os.getpid() % 300, a.ids), nprocs=RANKS)
