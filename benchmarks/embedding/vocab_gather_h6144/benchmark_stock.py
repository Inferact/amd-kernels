# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two- or four-rank (``--world``) draft embedding against what vLLM runs without it: ``VocabParallelEmbedding``'s
mask / local gather / zero fill and AITER's custom all-reduce, then (``--norm``) the one-GPU Gemma norm + MXFP8 quant
against the fused ``vocab_gather_gemma_mxfp8``. Real shard size (200064 / world rows), warm weights, 100 distinct
batches of 7 rows per request (one random anchor + six mask ids) per graph, alternated rounds after a barrier, max
over ranks, median us per call.

    python3 benchmarks/embedding/vocab_gather_h6144/benchmark_stock.py --world 2 [--norm]
"""
import argparse
import statistics
import torch
import torch.distributed as dist
from amd_kernels.embedding.vocab_gather_h6144 import VocabGather
from tests.multi_gpu import run_on_gpus


def worker(rank, errors, requests, norm):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from amd_kernels.embedding.vocab_gather_gemma_mxfp8_h6144 import vocab_gather_gemma_mxfp8
    from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144 import gemma_rmsnorm_mxfp8
    torch.set_num_threads(8)
    dev = torch.device('cuda', rank)
    world = dist.get_world_size()
    v = 200064 // world
    gen = torch.Generator(device=dev).manual_seed(31 + rank)
    weight = torch.randn((v, 6144), device=dev, dtype=torch.bfloat16, generator=gen)
    norm_w = torch.randn(6144, device=dev, dtype=torch.bfloat16, generator=gen)
    lookup = VocabGather(weight, rank, dist.group.WORLD)
    ca = CustomAllreduce(group=dist.group.WORLD, device=dev, max_size=64 * 1024 * 1024)
    start = rank * v

    def stock_embed(ids):
        local = ids - start
        valid = (local >= 0) & (local < v)
        x = torch.nn.functional.embedding(torch.where(valid, local, 0), weight)
        x.masked_fill_(~valid[:, None], 0)
        return ca.custom_all_reduce(x)

    for nr in requests:
        inputs = []
        for i in range(100):
            ids = torch.full((nr, 7), 200058, device=dev, dtype=torch.int64)
            g = torch.Generator(device=dev).manual_seed(1000 * nr + i)
            ids[:, 0] = torch.randint(0, 200064, (nr,), device=dev, generator=g)
            inputs.append(ids.flatten())
        if norm:
            fns = (lambda i: gemma_rmsnorm_mxfp8(stock_embed(inputs[i]), norm_w, 1e-6, 16),
                   lambda i: vocab_gather_gemma_mxfp8(inputs[i], lookup, norm_w, 1e-6, 16))
        else:
            fns = (lambda i: stock_embed(inputs[i]), lambda i: lookup(inputs[i]))
        for fn in fns:
            fn(0)
        torch.cuda.synchronize()
        graphs = []
        for fn in fns:
            gr = torch.cuda.CUDAGraph()
            with ca.capture(), torch.cuda.graph(gr):
                for i in range(100):
                    fn(i)
            graphs.append(gr)
        for gr in graphs:
            gr.replay()
        torch.cuda.synchronize()
        ts = [[], []]
        for r in range(6):
            for k in ((0, 1) if r % 2 == 0 else (1, 0)):
                dist.barrier()
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(10):
                    graphs[k].replay()
                b.record()
                b.synchronize()
                ts[k].append(a.elapsed_time(b))  # ms per 1000 calls = us per call
        every = [None] * world
        dist.all_gather_object(every, ts)
        if rank == 0:
            med = [statistics.median(max(e[k][r] for e in every) for r in range(6)) for k in range(2)]
            print(f"requests {nr} (rows {7 * nr}): stock {med[0]:.2f} us | ours {med[1]:.2f} us | "
                  f"{med[0] / med[1]:.2f}x", flush=True)
        del graphs
    ca.close()
    lookup.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reqs', type=int, nargs='+', default=[1, 10, 16, 32, 48, 64, 128, 256])
    p.add_argument('--world', type=int, default=2, choices=(2, 4))
    p.add_argument('--norm', action='store_true')
    a = p.parse_args()
    run_on_gpus(worker, a.reqs, a.norm, world=a.world, timeout=900)
