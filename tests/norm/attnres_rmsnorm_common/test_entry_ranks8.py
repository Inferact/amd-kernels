# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""On 8 ranks: the communicator from the entry; its method is bit for bit the package's function on every rank at the
smallest and the largest served rows (with and without prefix, output norm, block write, no reduce) and the reason is
None there; a call the communicator was not built for gets a reason; after the warm-up the served rows compile
nothing, and a graph of the method, captured with ``importlib.import_module`` unavailable, replays the eager result."""
import importlib
from unittest import mock

import torch
import torch.distributed as dist

from tests.multi_gpu8 import run_on_8_gpus
from tests.norm.allreduce_attnres_rmsnorm_h7168_ranks8.workload import CASES, make

KEY = (7168, 8)


def bits(t):
    return t.contiguous().view(torch.uint8)


def body(rank, errors):
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.norm import attnres_rmsnorm_common as N
    from amd_kernels.norm.attnres_rmsnorm_common import host as entry

    warm_up.set_auto(True)
    dev = torch.device("cuda", rank)
    group = dist.group.WORLD
    host = importlib.import_module(entry._ROUTES.hosts[KEY])
    if N.attnres_comm_blocker(rank, group, dev, hidden=7168) is not None:
        errors.append(f"rank {rank}: factory reason")
    comm = N.attnres_comm(rank, group, dev, hidden=7168)
    if type(comm) is not host.AttnResComm:
        errors.append(f"rank {rank}: factory built a {type(comm)}")

    for m in (1, 256):
        for ci, (hp, nb, widx, on) in enumerate(CASES):
            for reduce in (True, False):
                parts, prefix0, bank0, nw, qk, ow = make(m, m * 10 + ci, dev)
                outs = []
                package = lambda *a, **k: host.allreduce_attnres_rmsnorm(comm, *a, **k)
                for call in (comm.allreduce_attnres_rmsnorm, package):
                    prefix, bank = prefix0.clone(), bank0.clone()
                    a = (parts[rank], prefix if hp else None, bank, nw, qk, nb, widx, ow if on else None)
                    if N.allreduce_attnres_rmsnorm_blocker(comm, *a, reduce=reduce) is not None:
                        errors.append(f"rank {rank}: reason on served rows {m}")
                    out, pout = call(*a, reduce=reduce)
                    outs.append((out, pout, bank))
                torch.cuda.synchronize()
                if not all(torch.equal(bits(x), bits(y)) for x, y in zip(*outs)):
                    errors.append(f"rank {rank}: rows {m} case {ci} reduce {reduce} differ from the package call")

    parts, prefix0, bank0, nw, qk, ow = make(12, 5, dev)
    reason = N.allreduce_attnres_rmsnorm_blocker(comm, parts[rank][:, :6144], None, None, None, None, 0)
    if reason != "allreduce_attnres_rmsnorm: communicator built for hidden=7168, world=8, call is hidden=6144":
        errors.append(f"rank {rank}: mismatch reason {reason!r}")

    msgs = []
    compile_watch.arm(msgs.append)
    late0 = compile_watch.late_compiles()
    for m in (1, 12, 24, 40, 65, 80, 128, 192, 256):
        parts, prefix0, bank0, nw, qk, ow = make(m, m, dev)
        comm.allreduce_attnres_rmsnorm(parts[rank], prefix0, bank0, nw, qk, 3, -1, ow)
    torch.cuda.synchronize()
    if compile_watch.late_compiles() != late0:
        errors.append(f"rank {rank}: compiles after the warm-up: {msgs[:2]}")

    parts, prefix0, bank0, nw, qk, ow = make(12, 9, dev)
    prefix = prefix0.clone()
    eager, eager_prefix = (t.clone() for t in comm.allreduce_attnres_rmsnorm(parts[rank], prefix, bank0, nw, qk, 3, -1,
                                                                             ow))
    prefix.copy_(prefix0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), mock.patch("importlib.import_module", side_effect=RuntimeError("import")):
        out, _ = comm.allreduce_attnres_rmsnorm(parts[rank], prefix, bank0, nw, qk, 3, -1, ow)
    prefix.copy_(prefix0)
    graph.replay()
    torch.cuda.synchronize()
    if not (torch.equal(bits(out), bits(eager)) and torch.equal(bits(prefix), bits(eager_prefix))):
        errors.append(f"rank {rank}: graph replay differs from eager")
    del graph
    comm.close()
    compile_watch.disarm()
    warm_up.set_auto(False)


def test_attnres_entry_on_8_ranks():
    run_on_8_gpus(body, timeout=600)
