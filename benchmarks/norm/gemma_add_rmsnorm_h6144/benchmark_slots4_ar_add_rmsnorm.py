# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of the MoE combine fused into the 4-rank all-reduce + add +
Gemma RMSNorm + MXFP8 quant (``SlotsArNormComm.slots4_ar_add_rmsnorm_mxfp8``) on
4 gfx950 GPUs, against the baseline sequence after the MoE at that row count:
the MoE entry's reduced-mode sum of the slots and the shared output in one launch
(``common.reduce_slots_shared``), then AITER's custom all-reduce, bf16, Quick Reduce off,
up to its 64 MiB cap (5461 rows), RCCL above it, as vLLM's ``all_reduce`` falls through
to pynccl; then the one-GPU ``gemma_add_rmsnorm_mxfp8``, ``--tile=0``:
``gemma_add_rmsnorm_mxfp8_row_major``.
``add+ar+norm`` leaves out the slot sum (the MoE entry's reduced mode already did
it); ``ar+norm`` is the last step alone.

Per row count one graph per side holds ``CALLS`` calls (16 from 4096 rows), each on
its own slots, shared output and residual; the time per call is the replay time / CALLS. The
sides alternate, ROUNDS rounds of 10 replays each. Each sample is the maximum
over ranks; printed: median [min - max] of those samples. All ranks replay
together (a barrier before every timed replay set).

    python3 benchmarks/norm/gemma_add_rmsnorm_h6144/benchmark_slots4_ar_add_rmsnorm.py [out.json] [--tile=0] [--eager] [rows ...]
"""

import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

from tests.multi_gpu import WORLD, run_on_4_gpus

EPS = 1e-6
H = 6144
ROWS = [8, 16, 32, 64, 128, 256, 512, 1024]
CALLS = 100
LARGE_ROWS, LARGE_CALLS = 4096, 16
AITER_AR_MAX_BYTES = 64 << 20  # AITER_CUSTOM_AR_MAX_SIZE's default
TILE = 16
ROUNDS = 5
REPLAYS = 10

# --tile=0: row-major codes.
# --eager also times direct calls, including host dispatch, after graph arms.


def _calls(rows):
    return LARGE_CALLS if rows >= LARGE_ROWS else CALLS


def _time(graph, calls):
    dist.barrier()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (REPLAYS * calls) * 1e3


def _worker(rank, errors, rows_list, out_json, tile=TILE, eager=False):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.reduce_slots_shared import reduce_slots_shared
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import reduce_slots_bf16_ref
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.host import (
        SlotsArNormComm,
    )

    dev = torch.device("cuda", rank)
    group = dist.group.WORLD
    ca = CustomAllreduce(group=group, device=dev, max_size=8192 * 1024 * 8 * 2)
    assert not ca.disabled
    # vLLM's fallback above AITER's cap: RCCL (pynccl there, a torch NCCL group here)
    nccl = None
    if max(rows_list) * H * 2 > AITER_AR_MAX_BYTES:
        nccl = dist.new_group(list(range(WORLD)), backend="nccl")
        probe = torch.ones(8, device=dev)
        dist.all_reduce(probe, group=nccl)
        torch.cuda.synchronize()

    def all_reduce(x):
        y = ca.custom_all_reduce(x)
        if y is None:
            dist.all_reduce(x, group=nccl)  # in place; pynccl writes a new tensor
            y = x
        return y

    comm = SlotsArNormComm(rank, group, dev)
    w = (torch.randn(H, device=dev) * 0.5).bfloat16()
    if rank == 0:
        props = torch.cuda.get_device_properties(dev)
        print(f"rank 0 pci_bus {props.pci_bus_id:#x}", flush=True)
    results = {}
    for rows in rows_list:
        calls = _calls(rows)
        torch.manual_seed(rows * 10 + rank)
        slots = [torch.randn(rows, 4, H, device=dev).bfloat16() for _ in range(calls)]
        shared = [torch.randn(rows, H, device=dev).bfloat16() for _ in range(calls)]
        res = [torch.randn(rows, H, device=dev).bfloat16() for _ in range(calls)]
        reds = [reduce_slots_bf16_ref(s) for s in slots]
        xs = [r + h for r, h in zip(reds, shared)]

        def norm(x, r):
            if tile:
                return an.gemma_add_rmsnorm_mxfp8(x, r, w, EPS, tile)
            return an.gemma_add_rmsnorm_mxfp8_row_major(x, r, w, EPS)

        def ar_norm(x, r):
            return norm(all_reduce(x), r)

        # compile everything outside the captures
        comm.slots4_ar_add_rmsnorm_mxfp8(slots[0], shared[0], res[0], w, EPS, tile)
        norm(xs[0], res[0])
        torch.cuda.synchronize()
        graphs = {}
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for s, h, r in zip(slots, shared, res):
                comm.slots4_ar_add_rmsnorm_mxfp8(s, h, r, w, EPS, tile)
        graphs["fused"] = g
        for key in ("today", "add+ar+norm", "ar+norm"):
            g = torch.cuda.CUDAGraph()
            with ca.capture(), torch.cuda.graph(g):
                for s, h, r, x, red in zip(slots, shared, res, xs, reds):
                    if key == "today":
                        x = reduce_slots_shared(s, h)
                    elif key == "add+ar+norm":
                        x = torch.add(red, h)
                    ar_norm(x, r)
            graphs[key] = g
        for g in graphs.values():
            for _ in range(2):
                g.replay()
        torch.cuda.synchronize()
        times = {k: [] for k in graphs}
        def eager_fused():
            for s, h, r in zip(slots, shared, res):
                comm.slots4_ar_add_rmsnorm_mxfp8(s, h, r, w, EPS, tile)
        def eager_today(combine=True):
            for s, h, r, red in zip(slots, shared, res, reds):
                x = reduce_slots_shared(s, h) if combine else torch.add(red, h)
                norm(all_reduce(x), r)
        eager_fns = {"eager_fused": eager_fused, "eager_today": eager_today,
                     "eager_add_ar_norm": lambda: eager_today(False)} if eager else {}
        times.update({k: [] for k in eager_fns})
        for round_id in range(ROUNDS):
            keys = list(graphs)
            if round_id % 2:
                keys.reverse()
            for k in keys:
                times[k].append(_time(graphs[k], calls))
            keys = list(eager_fns)
            if round_id % 2:
                keys.reverse()
            for k in keys:
                fn = eager_fns[k]
                fn()  # warm the eager allocation pool after graph execution
                torch.cuda.synchronize()
                dist.barrier()
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(REPLAYS):
                    fn()
                end.record()
                torch.cuda.synchronize()
                times[k].append(start.elapsed_time(end) * 1e3 / (REPLAYS * calls))
        every = [None] * WORLD
        dist.all_gather_object(every, times)
        if rank == 0:
            samples = {k: [max(e[k][i] for e in every) for i in range(ROUNDS)]
                       for k in times}
            line = [f"rows={rows}"]
            for k in times:
                v = samples[k]
                vmax = max(statistics.median(e[k]) for e in every)
                line.append(
                    f"{k} {statistics.median(v):.2f} [{min(v):.2f}-{max(v):.2f}] (max rank {vmax:.2f})"
                )
            t = samples["today"]
            f = samples["fused"]
            wins = sum(a < b for a, b in zip(f, t))
            line.append(f"fused saves {statistics.median(t) - statistics.median(f):+.2f} us ({wins}/{ROUNDS} rounds)")
            print(" | ".join(line), flush=True)
            results[rows] = {"rank0": times, "every": every,
                             "calls": calls, "aiter_ar": rows * H * 2 <= AITER_AR_MAX_BYTES,
                             "tile": tile, "max_rank_samples": samples}
        del graphs, slots, shared, res, xs, reds
        torch.cuda.empty_cache()
    if rank == 0 and out_json:
        with open(out_json, "w") as fh:
            json.dump(results, fh, indent=1)
    ca.close()
    if nccl is not None:
        # torch's NCCL group never finishes destroy_process_group after graph-captured
        # RCCL calls: every rank leaves after a gloo barrier
        torch.cuda.synchronize()
        dist.barrier()
        os._exit(0)


def main():
    if torch.cuda.device_count() < WORLD:
        sys.exit("needs 4 GPUs")
    args = sys.argv[1:]
    out_json = None
    if args and args[0].endswith(".json"):
        out_json, args = args[0], args[1:]
    tile = TILE
    eager = False
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag.startswith("--tile="):
            tile = int(flag[7:])
        elif flag == "--eager":
            eager = True
        else:
            raise ValueError(flag)
    rows = [int(a) for a in args] or ROWS
    run_on_4_gpus(_worker, rows, out_json, tile, eager, timeout=3000)


if __name__ == "__main__":
    main()
