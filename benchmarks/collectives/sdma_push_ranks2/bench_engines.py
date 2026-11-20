# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SDMA bandwidth over the one xGMI link of a GPU pair by engine assignment: rank 0 (one direction) or both ranks
(both directions) copy ``mb`` MB into the peer's IPC pool, split evenly over the listed engines. Prints the engines
``hsa_amd_memory_copy_engine_status`` reports free and GB/s per direction (host-timed from enqueue to the last
completion signal, best of 5, max over ranks).

  python3 benchmarks/collectives/sdma_push_ranks2/bench_engines.py [mb ...] [out.json]
"""
import ctypes
import json
import sys
import time

import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_2_gpus

CONFIGS = ((2,), (3,), (8,), (2, 3), (2, 8), (2, 4), (2, 3, 8, 9), (2, 3, 4, 5, 6, 7, 8, 9))


def _worker(rank, errors, sizes, out):
    from amd_kernels.collectives.sdma_push_ranks2 import hsa_runtime as rt
    from amd_kernels.collectives.sdma_push_ranks2.ipc import alloc_pool, exchange, open_pool

    dev = torch.device("cuda", rank)
    peer = 1 - rank
    n_max = max(sizes) << 20
    ptr, handle, pool = alloc_pool(n_max, dev)
    hs = exchange((handle, rt.pci_key(dev)), rank, 2, dist.group.WORLD)
    peer_ptr = open_pool(hs[peer][0], dev)
    agents = rt.gpu_agents()
    me, other = agents[hs[rank][1]], agents[hs[peer][1]]
    lib = rt.hsa()
    src = torch.randint(0, 255, (n_max,), dtype=torch.uint8, device=dev)
    mask = ctypes.c_uint32()
    lib.hsa_amd_memory_copy_engine_status(other, me, ctypes.byref(mask))
    free = [e for e in range(32) if mask.value >> e & 1]
    print(f"rank {rank}: engines free to the peer {free}", flush=True)
    torch.cuda.synchronize()

    def run(n, engines):
        sigs = []
        part = n // len(engines) // 256 * 256
        t0 = time.perf_counter()
        for j, e in enumerate(engines):
            size = part if j < len(engines) - 1 else n - j * part
            s = rt.signal(1)
            sigs.append(s)
            rc = lib.hsa_amd_memory_async_copy_on_engine(peer_ptr + j * part, other, src.data_ptr() + j * part, me,
                                                         size, 0, None, s, 1 << e, True)
            assert rc == 0, rc
        for s in sigs:
            while lib.hsa_signal_wait_scacquire(s, rt.HSA_SIGNAL_CONDITION_LT, 1, 1 << 40,
                                                rt.HSA_WAIT_STATE_ACTIVE) > 0:
                pass
        return time.perf_counter() - t0

    results = {}
    for mb in sizes:
        n = mb << 20
        for both in (False, True):
            for engines in CONFIGS:
                active = both or rank == 0
                best = 1e9
                for _ in range(5):
                    dist.barrier()
                    if active:
                        best = min(best, run(n, engines))
                t = [None, None]
                dist.all_gather_object(t, best if active else 0.0)
                gbs = n / max(t) / 1e9
                key = f"{mb}MB {'bidir' if both else 'uni'} engines{list(engines)}"
                results[key] = gbs
                if rank == 0:
                    print(f"{key:44s} {gbs:6.1f} GB/s per direction ({max(t) * 1e6:.0f} us)", flush=True)
    if rank == 0 and out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)


def main():
    args = sys.argv[1:]
    out = args.pop() if args and args[-1].endswith(".json") else None
    sizes = [int(a) for a in args] or [8, 64, 256]
    run_on_2_gpus(_worker, sizes, out, timeout=1800)


if __name__ == "__main__":
    main()
