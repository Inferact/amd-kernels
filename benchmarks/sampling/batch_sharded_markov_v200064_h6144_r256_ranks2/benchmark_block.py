# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks: graph replay of the verify logits of R requests (8 rows each, dense owners) and of the seven-step
Markov draft, MAX rank, five alternated rounds, median us.

Target logits:
- vllm: vLLM's TP2 path: its lm_head GEMM (``rocm_unquantized_gemm_impl``, VLLM_ROCM_USE_AITER=1) over
  all rows on the row-major shard, then the RCCL all-gather of the shards into [M, 200064] on every rank;
- vllm a2a: upstream batch-sharded sampling: the same GEMM, then RCCL all_to_all_single + permute (this rank's rows);
- exchange / local: ``BatchShardedSampling.target_logits`` through the logits exchange / with the peer shard copy.
Draft: ``markov_draft`` of R requests (draft lm_head of 7 R rows, seven Markov steps over the owned requests, the id
gather) through the exchange / locally; vllm = only the draft logits of vLLM's replicated draft (its GEMM over 7 R
rows + the all-gather), before its Markov loop over all R requests.
Run: python3 benchmarks/sampling/batch_sharded_markov_v200064_h6144_r256_ranks2/benchmark_block.py [R ...]
"""
import os
import socket
import statistics
import sys

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, V, H, N, RANK, STEPS = 2, 200064, 6144, 100032, 256, 7
SETS = 12


def capture(calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls[:2]:
            c()
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    g.replay()
    torch.cuda.synchronize()
    return g


def time_us(g, n):
    dist.barrier()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record()
    b.synchronize()
    t = torch.tensor([a.elapsed_time(b) * 1000 / (5 * n)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def run(rank, port, reqs):
    torch.set_num_threads(8)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=WORLD, rank=rank, device_id=dev)
    gloo = dist.new_group(backend="gloo")
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
    from amd_kernels.gemm.bf16_n100032_k6144 import shuffle_weight
    from amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks2 import BatchShardedSampling, host
    g = torch.Generator(device=dev).manual_seed(500 + rank)
    w = (torch.randn(N, H, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    lm = shuffle_weight(w)
    max_reqs = max(reqs)
    s_x = BatchShardedSampling(rank, gloo, dev, max_reqs=max_reqs, target_rows=max_reqs * 8)
    s_l = BatchShardedSampling(rank, gloo, dev, max_reqs=max_reqs, target_rows=max_reqs * 8, lm_weight=lm)
    dg = torch.Generator(device=dev).manual_seed(77)
    q = STEPS + 1
    t = max_reqs * q
    slots = max_reqs + 8
    req = torch.arange(max_reqs, device=dev).repeat_interleave(STEPS)
    step = torch.arange(STEPS, device=dev).repeat(max_reqs)
    sample_indices = (req * q + step + 1).to(torch.int64)
    sample_idx_mapping = torch.randperm(slots, device=dev, generator=dg)[:max_reqs][req].to(torch.int32)
    sample_pos = torch.randint(1, 1 << 20, (max_reqs,), device=dev, generator=dg)[req] + step
    input_ids = torch.randint(0, V, (t,), device=dev, generator=dg).to(torch.int32)
    anchor = torch.arange(max_reqs, device=dev) * q
    temperature = torch.ones(slots, device=dev)
    seeds = torch.randint(0, 1 << 62, (slots,), device=dev, generator=dg)
    w1 = (torch.randn(V, RANK, device=dev, generator=dg) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(V, RANK, device=dev, generator=dg) * 0.05).to(torch.bfloat16)
    cache = torch.zeros(slots, STEPS, V, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(max_reqs, STEPS, dtype=torch.int64, device=dev)
    for s in (s_x, s_l):
        s.setup_markov(STEPS, max_reqs, anchor, w1, w2)
    hg = torch.Generator(device=dev).manual_seed(9)
    hidden = [torch.randn(max_reqs * 8, H, device=dev, generator=hg).to(torch.bfloat16) for _ in range(SETS)]
    if rank == 0:
        print("target: R | M | vllm us | vllm a2a us | exchange us | local us", flush=True)
    for R in reqs:
        m = 8 * R
        owners = torch.arange(R) % WORLD
        counts = [int((owners == r).sum()) * 8 for r in range(WORLD)]
        req_sorted = torch.cat([torch.nonzero(owners == r).flatten() for r in range(WORLD)])
        rows = (req_sorted[:, None] * 8 + torch.arange(8)).flatten().to(dev)
        n = counts[rank]
        gathered = torch.empty(WORLD * m, N, dtype=torch.bfloat16, device=dev)
        recv = torch.empty(WORLD * n, N, dtype=torch.bfloat16, device=dev)

        def vllm_ag(x):
            y = rocm_unquantized_gemm_impl(x[:m], w)
            dist.all_gather_into_tensor(gathered, y)
            return gathered.view(WORLD, m, N).permute(1, 0, 2).reshape(m, V)

        def vllm_a2a(x):
            y = rocm_unquantized_gemm_impl(x[rows], w)
            dist.all_to_all_single(recv, y, [n] * WORLD, counts)
            return recv.view(WORLD, n, N).permute(1, 0, 2).reshape(n, V)

        variants = {
            "vllm": [lambda x=x: vllm_ag(x) for x in hidden],
            "vllm_a2a": [lambda x=x: vllm_a2a(x) for x in hidden],
            "exchange": [lambda x=x: s_x.target_logits(x, lm, rows, counts, n) for x in hidden],
            "local": [lambda x=x: s_l.target_logits(x, lm, rows, counts, n) for x in hidden],
        }
        host.LOCAL_MIN_ROWS = 0  # the local variant at every M
        a = s_x.target_logits(hidden[0], lm, rows, counts, n)
        b = s_l.target_logits(hidden[0], lm, rows, counts, n)
        c = vllm_a2a(hidden[0])
        torch.cuda.synchronize()
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (rank, R)
        if n:
            cos = torch.nn.functional.cosine_similarity(a.float().flatten(), c.float().flatten(), dim=0)
            assert cos > 0.9999, float(cos)
        graphs = {k: capture(v) for k, v in variants.items()}
        ts = {k: [] for k in graphs}
        for _ in range(5):
            for k, gr in graphs.items():
                ts[k].append(time_us(gr, SETS))
        if rank == 0:
            print(f"{R} | {m} | " + " | ".join(f"{statistics.median(ts[k]):.1f}" for k in variants), flush=True)
        del graphs
        torch.cuda.empty_cache()
    heads = [torch.randn(t, H, device=dev, generator=dg).to(torch.bfloat16) for _ in range(SETS)]
    if rank == 0:
        print("draft: R | rows | vllm logits us | exchange us | local us", flush=True)
    for R in reqs:
        m = R * STEPS
        gathered = torch.empty(WORLD * m, N, dtype=torch.bfloat16, device=dev)

        def vllm_draft(x):
            y = rocm_unquantized_gemm_impl(x[:m], w)
            dist.all_gather_into_tensor(gathered, y)
            return gathered.view(WORLD, m, N).permute(1, 0, 2).reshape(m, V)

        def block(s, x):
            s.markov_draft(R, x, lm, sample_indices, sample_idx_mapping, sample_pos, input_ids, temperature, seeds,
                           cache, ids)

        variants = {"vllm": [lambda x=x: vllm_draft(x) for x in heads],
                    "exchange": [lambda x=x: block(s_x, x) for x in heads],
                    "local": [lambda x=x: block(s_l, x) for x in heads]}
        graphs = {k: capture(v) for k, v in variants.items()}
        ts = {k: [] for k in graphs}
        for _ in range(5):
            for k, gr in graphs.items():
                ts[k].append(time_us(gr, SETS))
        if rank == 0:
            print(f"{R} | {m} | " + " | ".join(f"{statistics.median(ts[k]):.1f}" for k in variants), flush=True)
        del graphs
        torch.cuda.empty_cache()
    s_x.close()
    s_l.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    reqs = [int(v) for v in sys.argv[1:]] or [1, 4, 8, 10, 12, 16, 20, 24, 32, 40, 48, 52, 64, 96, 128, 256]
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(run, args=(port, reqs), nprocs=WORLD, join=True)
