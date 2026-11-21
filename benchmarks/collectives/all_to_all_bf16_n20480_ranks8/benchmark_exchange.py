# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank lm_head logits: vLLM's GEMM + all-gather / + all_to_all_single vs this GEMM pushing into the receivers' buffers.

Rows: M hidden rows replicated on every rank (as the model's), owner-sorted by request owner (rank i % 8, a request
has --tokens rows). Four timings, each GEMM included, graph replay of 6 distinct inputs over three weight copies,
minimum of nine rounds, MAX over ranks, us per call:
  vLLM AG       rocm_unquantized_gemm_impl + tensor_model_parallel_all_gather -> [M, 163840]   (vLLM's default)
  vLLM A2A      GEMM + all_to_all_single + permute -> [owned rows, 163840]  (--enable-batch-sharded-sampling)
  this AG       push GEMM (all ranks) + wait -> [M, 163840] in place in the receive buffer
  this owner    push GEMM (owner-sorted row map) + wait -> [owned rows, 163840] in place
Ceilings (measured): the GEMM's HBM stream (293.6 MB at 6.5 TB/s) or its MFMA time (1.5 PFLOP/s),
whichever is larger, plus the exchange: one launch gap and one cross-GPU flag hop (1.6 + 0.8 us) plus the bytes of the
busiest link at 49 GB/s (the measured 8-GPU all-to-all: 343 GB/s out over seven links): AG moves M x 40 KB per
link, owner rows (largest owner count) x 40 KB. `% ceiling` = ceiling / this.
"""
import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N, K, WORLD = 20480, 7168, 8
LINK_GBS = 49.0
HBM_TBS, PEAK_TFLOPS = 6.5, 1500.0
LATENCY_US = 2.4


def worker(rank, ms, tokens, port):
    torch.cuda.set_device(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                  tensor_model_parallel_all_gather)
    from vllm.distributed.parallel_state import get_tp_group, graph_capture
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
    from amd_kernels.collectives.all_to_all_bf16_n20480_ranks8 import RowAllToAll
    
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(WORLD, rank, f"tcp://127.0.0.1:{port}", rank, "nccl")
        initialize_model_parallel(tensor_model_parallel_size=WORLD)
        tp = get_tp_group()
        dev = torch.device("cuda", rank)
        g = torch.Generator(device=dev).manual_seed(rank)
        ws = [(torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16) for _ in range(3)]
        comm = RowAllToAll(rank, tp.cpu_group, dev)
        comm.warmup()
        stock = dispatch_unquantized_gemm()

        def timed(fn, ni=6):
            for i in range(2):
                fn(i)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with graph_capture(device=dev) as ctx:
                with torch.cuda.graph(graph, stream=ctx.stream):
                    for i in range(ni):
                        fn(i)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(9):
                dist.barrier(tp.cpu_group)
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(10):
                    graph.replay()
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b) * 100 / ni)
            t = torch.tensor([min(ts)], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return t.item()

        if rank == 0:
            print(f"{'M':>4} {'vLLM AG':>8} {'vLLM A2A':>9} {'this AG':>8} {'this owner':>10} {'AG x':>6} {'A2A x':>6} "
                  f"{'AG ceil':>8} {'AG %':>5} {'own ceil':>9} {'own %':>6}")
        for m in ms:
            reqs = max(1, m // tokens)
            owner = torch.arange(m) // tokens % WORLD  # request r -> rank r % 8
            order = torch.argsort(owner, stable=True)
            counts = torch.bincount(owner, minlength=WORLD).tolist()
            cs = torch.tensor(counts, dtype=torch.int32, device=dev)
            idx = order.to(dev)
            ident = torch.arange(m, dtype=torch.int64, device=dev)
            xs = [torch.randn(m, K, device=dev, generator=torch.Generator(device=dev).manual_seed(7 + i)).to(torch.bfloat16)
                  for i in range(6)]

            def vllm_ag(i):
                return tensor_model_parallel_all_gather(stock(None, xs[i], ws[i % 3], None))

            def vllm_a2a(i):
                y = stock(None, xs[i][idx], ws[i % 3], None)
                recv = y.new_empty(WORLD * counts[rank], N)
                dist.all_to_all_single(recv, y, output_split_sizes=[counts[rank]] * WORLD, input_split_sizes=counts,
                                       group=tp.device_group)
                return recv.view(WORLD, counts[rank], N).permute(1, 0, 2).reshape(counts[rank], WORLD * N)

            def this_ag(i):
                return comm.gemm_exchange(xs[i], ws[i % 3], ident, None, i % 2).recv(m, i % 2)

            def this_owner(i):
                return comm.gemm_exchange(xs[i], ws[i % 3], idx, cs, i % 2).recv(max(counts[rank], 1), i % 2)

            res = [timed(f) for f in (vllm_ag, vllm_a2a, this_ag, this_owner)]
            if rank == 0:
                gemm = max((N * K * 2 + m * K * 2) / (HBM_TBS * 1e6), 2 * m * N * K / (PEAK_TFLOPS * 1e6))
                ag_ceil = gemm + LATENCY_US + m * N * 2 / (LINK_GBS * 1e3)
                own_ceil = gemm + LATENCY_US + max(counts) * N * 2 / (LINK_GBS * 1e3)
                print(f"{m:>4} {res[0]:>8.1f} {res[1]:>9.1f} {res[2]:>8.1f} {res[3]:>10.1f} {res[0] / res[2]:>5.2f}x "
                      f"{res[1] / res[3]:>5.2f}x {ag_ceil:>8.1f} {100 * ag_ceil / res[2]:>4.0f}% {own_ceil:>9.1f} "
                      f"{100 * own_ceil / res[3]:>5.0f}%", flush=True)
        comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ms", type=int, nargs="*", default=[12, 24, 40, 65, 80, 96, 128, 160, 192])
    p.add_argument("--tokens", type=int, default=4, help="rows per request")
    a = p.parse_args()
    mp.spawn(worker, args=(a.ms, a.tokens, 29700 + os.getpid() % 200), nprocs=WORLD)
