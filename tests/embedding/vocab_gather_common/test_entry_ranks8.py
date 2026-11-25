# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``vocab_gather_comm`` on 8 GPUs, for every key: the 8-rank group (hidden 7168) and 4- and 2-rank subgroups
(hidden 6144). On every rank the entry builds the package's ``VocabGather``; its lookups (int32 / int64 ids, 1-2048
tokens) equal a directly built gather's bit for bit, with no reason; nothing compiles after ``warm_up()``; a graph of
the call captured with ``importlib.import_module`` failing replays as the eager call; a wrong rank and an uncovered
group size get a reason before any collective."""

import importlib

import torch
import torch.distributed as dist

from tests.multi_gpu8 import run_on_8_gpus


def worker(rank, errors):
    from amd_kernels import compile_watch
    from amd_kernels.embedding.vocab_gather_common import (
        vocab_gather_blocker,
        vocab_gather_comm,
        vocab_gather_comm_blocker,
    )
    from amd_kernels.embedding.vocab_gather_common.host import _ROUTES

    dev = torch.device("cuda", rank)
    pairs = [dist.new_group([2 * i, 2 * i + 1]) for i in range(4)]
    quads = [dist.new_group(list(range(4 * i, 4 * i + 4))) for i in range(2)]
    late = []
    compile_watch.arm(late.append)
    cases = ((7168, dist.group.WORLD, rank), (6144, quads[rank // 4], rank % 4), (6144, pairs[rank // 2], rank % 2))
    for hidden, group, grank in cases:
        size = dist.get_world_size(group)
        tag = (hidden, size, rank)
        gen = torch.Generator().manual_seed(1000 * size + rank)
        weight = torch.randn(128, hidden, generator=gen).bfloat16().to(dev)
        why = vocab_gather_comm_blocker(weight, grank, group)
        if why is not None:
            errors.append(f"{tag}: {why}")
            continue
        wrong = vocab_gather_comm_blocker(weight, (grank + 1) % size, group)
        if wrong is None or "this process is rank" not in wrong:
            errors.append(f"{tag}: wrong rank: {wrong}")
        if size == 8 and "no kernel" not in (vocab_gather_comm_blocker(weight[:, :6144].contiguous(), grank, group)
                                             or ""):
            errors.append(f"{tag}: hidden 6144 on 8 ranks has a kernel")
        pkg = importlib.import_module(_ROUTES.hosts[(hidden, size)])
        entry = vocab_gather_comm(weight, grank, group)
        direct = pkg.VocabGather(weight, grank, group)
        if type(entry) is not pkg.VocabGather:
            errors.append(f"{tag}: entry built {type(entry)}")
        entry.warm_up()
        direct.warm_up()
        torch.cuda.synchronize()
        before = len(late)
        ids_gen = torch.Generator().manual_seed(7)
        for m in (1, 80, 2048):
            for dtype in (torch.int32, torch.int64):
                ids = torch.randint(-1, size * 128 + 1, (m,), generator=ids_gen, dtype=dtype).to(dev)
                if vocab_gather_blocker(entry, ids) is not None:
                    errors.append(f"{tag}: call reason {vocab_gather_blocker(entry, ids)}")
                a, b = entry(ids), direct(ids)
                torch.cuda.synchronize()
                if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                    errors.append(f"{tag}: m {m} {dtype}: entry != direct")
        ids = torch.randint(0, size * 128, (64,), generator=ids_gen).to(dev)
        eager = entry(ids)
        g = torch.cuda.CUDAGraph()
        real_import = importlib.import_module

        def no_import(name, *a, **k):
            raise AssertionError(f"import of {name} inside capture")

        importlib.import_module = no_import
        try:
            with torch.cuda.graph(g):
                out = entry(ids)
        finally:
            importlib.import_module = real_import
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        if not torch.equal(out.view(torch.int16), eager.view(torch.int16)):
            errors.append(f"{tag}: graph replay != eager")
        if len(late) != before:
            errors.append(f"{tag}: {len(late) - before} compiles after warm_up(): {late[before:]}")
        entry.close()
        direct.close()


def test_entry_ranks8():
    run_on_8_gpus(worker, timeout=110)
