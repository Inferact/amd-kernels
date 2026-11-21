# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One row count per eight-rank process set; graphs over input sets larger than the LLC, alternated rounds, MAX over
ranks, us per call (min and median of the rounds), and the roofline columns.

Baseline: AITER's custom all-reduce, then vLLM's Triton AttnRes kernel (regtile, the served default) on the reduced
tensor; the all-reduce and the AttnRes kernel alone are timed too. The call is the pre-MLP one of a layer (prefix +
delta, a bank of ``--blocks`` rows, output norm); ``--write`` adds the bank write of a block-write layer. Ceilings
(measured on MI355X; xGMI by ``ceilings.py``): HBM 6.4 TB/s, all-to-all xGMI 355 GB/s per GPU; the floor is the
all-reduce floor 5 us + 2 * 7/8 * bytes / 400 GB/s plus 1.2 us of tail (prefix add, norm, store).
"""
import argparse
import contextlib
import json
import statistics

import torch
import torch.distributed as dist

H = 7168
HBM_TBPS = 6.4  # measured read bandwidth of cold buffers, not the spec sheet
XGMI_GBPS = 355.0  # measured all-to-all push bandwidth per GPU (ceilings.py --mode bw)
ROW = H * 2


def traffic(m, blocks, write):
    """(HBM bytes of one rank, xGMI bytes it sends, FLOPs of one rank) of m rows of the pre-MLP call (prefix, ``blocks``
    bank rows, output norm). HBM: partial, prefix and the bank rows read; prefix out, output (and the bank row of a
    write) written; the 2 x 7/8 row of mailbox data received is written once and read once. xGMI: 2 phases x 7/8 row
    sent (and the same received). FLOPs: ss, dot per source (2 x 2 per element of blocks + 1 sources), the mix (2 per
    element and source), the prefix add, the output norm (about 4), the eight-way reduction of a 1/8 slice."""
    hbm = m * ROW * (2 + blocks + 2 + (1 if write else 0) + 2 * 7 / 8 * 2)
    flops = m * H * (4 * (blocks + 1) + 2 * (blocks + 1) + 1 + 4 + 7 / 8)
    return hbm, m * 2 * 7 * ROW // 8, flops


def floor_us(m):
    """Model floor: the measured all-reduce floor (5 us + 2 * 7/8 * bytes / 400 GB/s) plus the tail after the
    exchange."""
    return 5.0 + 1.75 * m * ROW / 400e3 + 1.2


def chain_floor_us(m):
    """Primitive-chain floor from measured pieces: the empty-kernel graph period 1.57 us,
    the partial load 0.5, a flight 0.77 + a mailbox read 0.23 twice, the reduction 0.1, the tail (prefix, norm, store)
    0.6, plus the two phases' bytes at the measured all-to-all push ceiling (they cannot overlap the flights)."""
    return 1.57 + 0.5 + 2 * (0.77 + 0.23) + 0.1 + 0.6 + m * 2 * 7 * ROW / 8 / (XGMI_GBPS * 1e3)


def worker(rank, errors, m, blocks, write, rounds):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8 import AttnResComm
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    torch.manual_seed(7000 + rank)
    dev = torch.device("cuda", rank)
    ca = CustomAllreduce(group=dist.group.WORLD, device=dev, max_size=64 * 1024 * 1024)
    comm = AttnResComm(rank, dist.group.WORLD, dev)
    set_bytes = m * H * 2 * (4 + blocks)
    n = max(8, min(100, int(600e6 // set_bytes)))
    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch.bfloat16)  # noqa: E731
    xs = [mk(m, H) * 0.5 for _ in range(n)]
    prefixes = [mk(m, H) for _ in range(n)]
    banks = [mk(m, 8, H) for _ in range(n)]
    nw, qk, ow = (1 + 0.1 * mk(H)).bfloat16(), 0.3 * mk(H), (1 + 0.1 * mk(H)).bfloat16()
    widx = blocks if write else -1

    def old(i):
        d = ca.custom_all_reduce(xs[i])
        return attn_res(prefixes[i], d, banks[i], nw, qk, ow, blocks, widx, 1e-6, 1e-6)

    def new(i):
        return comm.allreduce_attnres_rmsnorm(xs[i], prefixes[i], banks[i], nw, qk, blocks, widx, ow)

    deltas = [mk(m, H) for _ in range(n)]

    def ar_only(i):
        return ca.custom_all_reduce(xs[i])

    def attn_only(i):
        return attn_res(prefixes[i], deltas[i], banks[i], nw, qk, ow, blocks, widx, 1e-6, 1e-6)

    for fn in (old, new, ar_only, attn_only):
        fn(0)
    torch.cuda.synchronize()
    graphs = []
    for cm, fn in ((ca, old), (None, new), (ca, ar_only), (None, attn_only)):
        g = torch.cuda.CUDAGraph()
        with (cm.capture() if cm else contextlib.nullcontext()), torch.cuda.graph(g):
            for i in range(n):
                fn(i)
        graphs.append(g)
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()
    ts = [[] for _ in graphs]
    for r in range(rounds):
        for k in range(len(graphs)) if r % 2 == 0 else range(len(graphs) - 1, -1, -1):
            dist.barrier()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(10):
                graphs[k].replay()
            b.record()
            b.synchronize()
            ts[k].append(a.elapsed_time(b) * 1e3 / (10 * n))
    every = [None] * 8
    dist.all_gather_object(every, ts)
    if rank == 0:
        vals = [[max(v[k][r] for v in every) for r in range(rounds)] for k in range(len(graphs))]
        med = [statistics.median(v) for v in vals]
        hbm, link, flops = traffic(m, blocks, write)
        best = min(vals[1])
        floor, chain = floor_us(m), chain_floor_us(m)
        print(json.dumps(dict(
            rows=m, blocks=blocks, write=write, sets=n, stock_us=round(med[0], 2), ar_us=round(med[2], 2),
            attn_res_us=round(med[3], 2), fused_us=round(best, 2), fused_median_us=round(med[1], 2),
            speedup=round(med[0] / best, 3), hbm_mb=round(hbm / 1e6, 2),
            hbm_pct_ceiling=round(100 * hbm / (HBM_TBPS * 1e6) / best, 1), xgmi_mb=round(link / 1e6, 2),
            xgmi_pct_ceiling=round(100 * link / (XGMI_GBPS * 1e3) / best, 1), hbm_tbps=round(hbm / best / 1e6, 3),
            xgmi_gbps=round(link / best / 1e3, 1), gflops=round(flops / 1e9, 3), tflops_per_s=round(flops / best / 1e6, 3),
            floor_us=round(floor, 2), pct_of_floor=round(100 * floor / best, 1), chain_floor_us=round(chain, 2),
            pct_of_chain_floor=round(100 * chain / best, 1))), flush=True)
    comm.close()
    ca.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", nargs="+", type=int, default=[12, 24, 40, 65, 80, 128, 192])
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--write", action="store_true")
    p.add_argument("--rounds", type=int, default=5)
    a = p.parse_args()
    from tests.norm.allreduce_attnres_rmsnorm_h7168_ranks8.multi_gpu8 import run_on_8_gpus

    for m in a.rows:
        run_on_8_gpus(worker, m, a.blocks, a.write, a.rounds)
