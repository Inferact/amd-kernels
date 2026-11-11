# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SDMA push bandwidth on 4 ranks by engine assignment: every rank pushes `mb` MB to each of its 3 peers at once,
split over `k` engines per peer; prints the engines each rank sees free (hsa_amd_memory_copy_engine_status) and GB/s
out per GPU (host-timed from enqueue to the last completion signal, best of 5).
  torchrun --nproc-per-node 4 bench_engines.py [mb]"""
import ctypes, os, sys, time

import torch
import torch.distributed as dist

from amd_kernels.collectives.sdma_push_ranks4 import hsa_runtime as rt

rank = int(os.environ["RANK"]); W = 4
torch.cuda.set_device(rank); dist.init_process_group("gloo")
MB = int(sys.argv[1]) if len(sys.argv) > 1 else 64
n = MB << 20
buf = torch.zeros(W * n, dtype=torch.uint8, device="cuda")
src = torch.randint(0, 255, (n,), dtype=torch.uint8, device="cuda")
from torch.multiprocessing.reductions import reduce_tensor, rebuild_cuda_tensor
hs = [None] * W; dist.all_gather_object(hs, reduce_tensor(buf)[1])
peers_buf = [buf if r == rank else rebuild_cuda_tensor(*hs[r]) for r in range(W)]
keys = [None] * W; dist.all_gather_object(keys, rt.pci_key(torch.device("cuda", rank)))
ag = rt.gpu_agents(); agents = [ag[k] for k in keys]; lib = rt.hsa()
PEERS = [p for p in range(W) if p != rank]
free = {}
for p in PEERS:
    m = ctypes.c_uint32(); lib.hsa_amd_memory_copy_engine_status(agents[p], agents[rank], ctypes.byref(m))
    free[p] = [e for e in range(32) if m.value >> e & 1]
print(f"rank {rank}: free engines per peer {free}", flush=True)
def run(assign):
    """assign: peer -> list of engine ids; each peer's n bytes split evenly over its engines."""
    sigs = []
    t0 = time.perf_counter()
    for p in PEERS:
        es = assign[p]; part = n // len(es)
        for j, e in enumerate(es):
            s = rt.signal(1); sigs.append(s)
            rc = lib.hsa_amd_memory_async_copy_on_engine(peers_buf[p].data_ptr() + rank * n + j * part, agents[p],
                src.data_ptr() + j * part, agents[rank], part, 0, None, s, 1 << e, True)
            assert rc == 0, rc
    for s in sigs:
        while lib.hsa_signal_wait_scacquire(s, rt.HSA_SIGNAL_CONDITION_LT, 1, 1 << 40, rt.HSA_WAIT_STATE_ACTIVE) > 0:
            pass
    return time.perf_counter() - t0
configs = {
    "1 engine/peer (2,4,6)": {p: [e] for p, e in zip(PEERS, (2, 4, 6))},
    "2 engines/peer (2,3 4,5 6,7)": {p: es for p, es in zip(PEERS, ((2, 3), (4, 5), (6, 7)))},
    "2 engines/peer (2,8 4,10 6,12)": {p: es for p, es in zip(PEERS, ((2, 8), (4, 10), (6, 12)))},
    "4 engines/peer": {p: es for p, es in zip(PEERS, ((2, 3, 8, 9), (4, 5, 10, 11), (6, 7, 12, 13)))},
}
for name, assign in configs.items():
    ok = all(all((not free[p]) or e in free[p] for e in assign[p]) for p in PEERS)
    best = 1e9
    for _ in range(5):
        dist.barrier(); torch.cuda.synchronize()
        best = min(best, run(assign))
    t = torch.tensor([best]); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"{name:34s}: {3 * n / t.item() / 1e9:6.1f} GB/s out per GPU ({t.item() * 1e3:.2f} ms for 3 x {MB} MB)"
              f"{'' if ok else '  (some engines not in the free mask)'}", flush=True)
dist.destroy_process_group()
