# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight ranks: the communicator built through the entry gives every rank the same slots and indptr as one built
from the shape package directly (and as ``reference.py``) at 1, 7, 64 and 256 rows, eager and in a graph captured
with imports disabled; the reasons are None there and name the mismatch for another top-k or a foreign object."""

from unittest.mock import patch

import torch
import torch.distributed as dist

from amd_kernels.collectives.dcp8_topk_gather_merge_top2048.reference import reference
from tests.collectives.dcp8_topk_gather_merge_top2048.workload import K, PAGE, STRIDE, make_block_table, make_plane
from tests.multi_gpu8 import WORLD, run_on_8_gpus

ROWS = (1, 7, 64, 256)


def worker(rank, errors):
    from amd_kernels.collectives.dcp8_topk_gather_merge_top2048 import TopkGatherMerge
    from amd_kernels.collectives.topk_gather_merge_common import (
        topk_gather_merge_blocker,
        topk_gather_merge_comm,
        topk_gather_merge_comm_blocker,
    )

    dev = torch.device("cuda", rank)
    group = dist.group.WORLD
    why = topk_gather_merge_comm_blocker(rank, group, dev, topk=K)
    if why is not None:
        errors.append(f"rank {rank}: factory reason {why}")
    if "no kernel for topk=1024, world=8" not in (topk_gather_merge_comm_blocker(rank, group, dev, topk=1024) or ""):
        errors.append(f"rank {rank}: no reason for topk 1024")
    comm = topk_gather_merge_comm(rank, group, dev, topk=K)
    direct = TopkGatherMerge(rank, group, dev)
    comm.warmup()
    direct.warmup()
    for n, rows in enumerate(ROWS):
        planes = [make_plane(n, r, rows, ties=n % 2) for r in range(WORLD)]
        bt = make_block_table(n, rows)
        want_slots, want_indptr = reference(torch.stack([p[0] for p in planes]), planes[rank][1], bt, rank, PAGE,
                                            STRIDE)
        args = (planes[rank][0].to(dev), planes[rank][1].to(dev), bt.to(dev))
        outs = {}
        for name, c in (("entry", comm), ("direct", direct)):
            slots = torch.full((rows * K,), -7, dtype=torch.int32, device=dev)
            indptr = torch.full((rows + 1,), -7, dtype=torch.int32, device=dev)
            why = topk_gather_merge_blocker(c, *args, slots, indptr, PAGE, STRIDE)
            if why is not None:
                errors.append(f"rank {rank} rows {rows}: {why}")
            c(*args, slots, indptr, PAGE, STRIDE)
            torch.cuda.synchronize()
            outs[name] = (slots[: want_slots.numel()].cpu(), indptr.cpu())
        if not (torch.equal(outs["entry"][1], outs["direct"][1]) and torch.equal(outs["entry"][0], outs["direct"][0])):
            errors.append(f"rank {rank} rows {rows}: entry != direct")
        if not (torch.equal(outs["entry"][1], want_indptr) and torch.equal(outs["entry"][0], want_slots)):
            errors.append(f"rank {rank} rows {rows}: entry != reference")
        if rows == 64:
            slots = torch.full((rows * K,), -7, dtype=torch.int32, device=dev)
            indptr = torch.full((rows + 1,), -7, dtype=torch.int32, device=dev)
            stream = torch.cuda.Stream(dev)
            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream):
                comm(*args, slots, indptr, PAGE, STRIDE)
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with patch("importlib.import_module", side_effect=AssertionError("import inside capture")):
                with torch.cuda.graph(g, stream=stream):
                    comm(*args, slots, indptr, PAGE, STRIDE)
            slots.fill_(-7)
            indptr.fill_(-7)
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            if not (torch.equal(indptr.cpu(), want_indptr) and torch.equal(slots[: want_slots.numel()].cpu(),
                                                                            want_slots)):
                errors.append(f"rank {rank}: graph replay != reference")
            del g
    a = (args[0][:, :1024].contiguous(), args[1][:, :1024].contiguous(), args[2], slots, indptr, PAGE, STRIDE)
    why = topk_gather_merge_blocker(comm, *a) or ""
    if "communicator built for topk=2048, world=8, call is topk=1024" not in why:
        errors.append(f"rank {rank}: reason for topk 1024: {why!r}")
    if "is not a communicator of this entry" not in (topk_gather_merge_blocker(object(), *a) or ""):
        errors.append(f"rank {rank}: no reason for a foreign object")
    torch.cuda.synchronize()
    dist.barrier()
    comm.close()
    direct.close()
    if topk_gather_merge_blocker(comm, *args, slots, indptr, PAGE, STRIDE) != (
            "topk_gather_merge: communicator closed"):
        errors.append(f"rank {rank}: no reason for a closed communicator")


def test_entry_equals_package_on_eight_ranks():
    run_on_8_gpus(worker, timeout=110)
