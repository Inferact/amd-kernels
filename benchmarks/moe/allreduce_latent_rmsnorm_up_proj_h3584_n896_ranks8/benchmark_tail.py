# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight ranks, CUDA graph replay of 24 distinct weight / activation sets per graph (the 6.4 MB up-projection weights
alone span 154 MB; the activations rotate too), median of five alternated rounds, slowest rank. Baseline = vLLM's path at
the same place: AITER's custom all-reduce of the latent, then the Triton fused RMSNorm + up-projection
(two launches); the final all-reduce of the [rows, 7168] output is timed for reference and is not part of the entry.
Prints the roofline table: the latency + xGMI + HBM floor of the fused call and the achieved fraction of it."""
import argparse
import json
import statistics

import torch
import torch.distributed as dist
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 import LatentTailComm, allreduce_latent_rmsnorm_up_proj

K, N, H = 3584, 896, 7168
NI = 24
LAUNCH_US = 1.57  # an empty kernel's graph period, measured on MI355X
FLAG_US = 0.7  # one-way cross-GPU flag store + poll
XGMI_GBS = 430.0  # one GPU reading seven peers
HBM_GBS = 6900.0  # non-temporal reads
CU_GBS = 154.0  # one CU's L2 -> L1 rate (64 B per clock)


def floor_us(m: int) -> float:
    """Critical path of the fused call: launch, the x read, two pushes (flag + bytes over seven links), the up
    projection's x read through one CU, the MFMA / LDS / epilogue and the store drain. The weights (6.4 MB, ~1 us of
    HBM) load while the all-reduce runs."""
    hop = FLAG_US + (m * 7 * 1024.0 * 7 / 8) / (XGMI_GBS * 1e3)
    tiles = -(-(-(-m // 16)) // 6)
    x_cu = tiles * 16 * 7168 / (CU_GBS * 1e3)
    return LAUNCH_US + 0.25 + 2 * hop + 0.1 + max(0.5, x_cu) + 0.9


def worker(rank, errors, rows, rounds):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.models.kimi_k3.amd.ops.latent_norm_up_proj import latent_norm_up_proj_triton as triton_fn

    dev = torch.device("cuda", rank)
    ca = CustomAllreduce(group=dist.group.WORLD, device=dev, max_size=64 * 1024 * 1024)
    comm = LatentTailComm(rank, dist.group.WORLD, dev)
    ws = [(torch.randn(N, K, device=dev) * 0.03).bfloat16() for _ in range(NI)]
    nw = (1 + 0.1 * torch.randn(K, device=dev)).bfloat16()
    out = {}
    for m in rows:
        lat = [torch.randn(m, K, device=dev).bfloat16() for _ in range(NI)]
        full = [torch.randn(m, H, device=dev).bfloat16() for _ in range(NI)]
        views = [f[:, rank * N:(rank + 1) * N] for f in full]
        fns = {
            "stock_ar": lambda i: ca.custom_all_reduce(lat[i]),
            "stock_up": lambda i: triton_fn(lat[i], nw, 1e-6, ws[i], views[i]),
            "stock_final_ar": lambda i: ca.custom_all_reduce(full[i]),
            "fused": lambda i: allreduce_latent_rmsnorm_up_proj(comm, lat[i], nw, ws[i], views[i]),
        }
        graphs = {}
        for name, fn in fns.items():
            fn(0)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with ca.capture(), torch.cuda.graph(g):
                for i in range(NI):
                    fn(i)
            g.replay()
            graphs[name] = g
        torch.cuda.synchronize()
        times = {k: [] for k in fns}
        for r in range(rounds):
            for name in list(fns) if r % 2 == 0 else list(fns)[::-1]:
                dist.barrier()
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(10):
                    graphs[name].replay()
                b.record()
                b.synchronize()
                times[name].append(a.elapsed_time(b) * 1e3 / (10 * NI))
        every = [None] * 8
        dist.all_gather_object(every, times)
        out[m] = {k: statistics.median(max(v[k][j] for v in every) for j in range(rounds)) for k in fns}
    if rank == 0:
        print(json.dumps(out), flush=True)
        report(out)
    comm.close()
    ca.close()


def report(res):
    print("\nrows | stock AR | stock Triton | stock total | fused | speedup | floor | fused / floor | final AR (not in the entry)")
    for m, r in res.items():
        stock = r["stock_ar"] + r["stock_up"]
        fl = floor_us(m)
        print(f"{m:4d} | {r['stock_ar']:.1f} | {r['stock_up']:.1f} | {stock:.1f} | {r['fused']:.1f} | "
              f"{stock / r['fused']:.2f}x | {fl:.1f} | {100 * fl / r['fused']:.0f}% | {r['stock_final_ar']:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", nargs="+", type=int, default=[12, 24, 40, 65, 80, 128, 192])
    p.add_argument("--rounds", type=int, default=5)
    a = p.parse_args()
    from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.multi_gpu8 import run_on_8_gpus

    run_on_8_gpus(worker, a.rows, a.rounds, timeout=900)
