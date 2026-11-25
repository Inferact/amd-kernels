# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""On 8 ranks: the communicator from the entry; both latent tail calls through the entry are bit for bit the direct
package calls on every rank, at the smallest and the largest served rows; a call the communicator was not built for
gets a reason; after the warm-up the served rows compile nothing, and a graph of the entry call, captured with
``importlib.import_module`` unavailable, replays the eager result."""
import importlib
from unittest import mock

import torch
import torch.distributed as dist

from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.workload import make
from tests.multi_gpu8 import run_on_8_gpus

KEY = (896, 3584, 8)


def bits(t):
    return t.contiguous().view(torch.uint8)


def body(rank, errors):
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.moe import latent_moe_common as E
    from amd_kernels.moe.latent_moe_common import latent_tail

    warm_up.set_auto(True)
    dev = torch.device("cuda", rank)
    group = dist.group.WORLD
    host = importlib.import_module(latent_tail._ROUTES.hosts[KEY])
    if E.latent_tail_comm_blocker(rank, group, dev, up_rows=896, latent=3584) is not None:
        errors.append(f"rank {rank}: factory reason")
    comm = E.latent_tail_comm(rank, group, dev, up_rows=896, latent=3584)
    if type(comm) is not host.LatentTailComm:
        errors.append(f"rank {rank}: factory built a {type(comm)}")

    def slots_of(m, seed):
        g = torch.Generator(device=dev).manual_seed(seed + rank)
        cap = 16 * m + 896 * 15
        slots = (torch.randn(cap, 3584, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        rev = torch.randperm(cap, device=dev, generator=g)[: 16 * m].to(torch.int32)
        return slots, rev

    for m in (1, 256):
        parts, nw, w, shared0 = make(m, 50 + m, dev, rank)
        a, b = shared0.clone(), shared0.clone()
        if E.allreduce_latent_rmsnorm_up_proj_blocker(comm, parts[rank], nw, w, a) is not None:
            errors.append(f"rank {rank}: reason on served rows {m}")
        host.allreduce_latent_rmsnorm_up_proj(comm, parts[rank], nw, w, a)
        E.allreduce_latent_rmsnorm_up_proj(comm, parts[rank], nw, w, b)
        torch.cuda.synchronize()
        if not torch.equal(bits(a), bits(b)):
            errors.append(f"rank {rank}: latent tail rows {m} differ from the package call")
    for m in (1, 64):
        _, nw, w, shared0 = make(m, 70 + m, dev, rank)
        slots, rev = slots_of(m, 90 + m)
        a, b = shared0.clone(), shared0.clone()
        if E.allreduce_slots_rmsnorm_up_proj_blocker(comm, slots, rev, nw, w, a) is not None:
            errors.append(f"rank {rank}: slots reason on served rows {m}")
        host.allreduce_slots_rmsnorm_up_proj(comm, slots, rev, nw, w, a)
        E.allreduce_slots_rmsnorm_up_proj(comm, slots, rev, nw, w, b)
        torch.cuda.synchronize()
        if not torch.equal(bits(a), bits(b)):
            errors.append(f"rank {rank}: slots rows {m} differ from the package call")

    parts, nw, w, shared0 = make(12, 7, dev, rank)
    wide = torch.zeros(1024, 3584, dtype=torch.bfloat16, device=dev)
    reason = E.allreduce_latent_rmsnorm_up_proj_blocker(comm, parts[rank], nw, wide, shared0)
    if "communicator built for up_rows=896, latent=3584, world=8, call is up_rows=1024" not in (reason or ""):
        errors.append(f"rank {rank}: mismatch reason {reason!r}")

    msgs = []
    compile_watch.arm(msgs.append)
    late0 = compile_watch.late_compiles()
    for m in (1, 12, 40, 64, 65, 80, 128, 192, 256):
        parts, nw, w, shared0 = make(m, m, dev, rank)
        E.allreduce_latent_rmsnorm_up_proj(comm, parts[rank], nw, w, shared0)
    for m in (1, 16, 33, 64):
        _, nw, w, shared0 = make(m, m, dev, rank)
        slots, rev = slots_of(m, m)
        E.allreduce_slots_rmsnorm_up_proj(comm, slots, rev, nw, w, shared0)
    torch.cuda.synchronize()
    if compile_watch.late_compiles() != late0:
        errors.append(f"rank {rank}: compiles after the warm-up: {msgs[:2]}")

    parts, nw, w, shared0 = make(12, 3, dev, rank)
    eager = shared0.clone()
    E.allreduce_latent_rmsnorm_up_proj(comm, parts[rank], nw, w, eager)
    view = shared0.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), mock.patch("importlib.import_module", side_effect=RuntimeError("import")):
        E.allreduce_latent_rmsnorm_up_proj(comm, parts[rank], nw, w, view)
    view.copy_(shared0)
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(bits(view), bits(eager)):
        errors.append(f"rank {rank}: graph replay differs from eager")
    del graph
    comm.close()
    compile_watch.disarm()
    warm_up.set_auto(False)


def test_latent_tail_entry_on_8_ranks():
    run_on_8_gpus(body, timeout=600)
