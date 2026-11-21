# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 on eight ranks: the first call warms every row tier up (no launch inside a warm-up); then the served row
counts, eager and in a graph, compile nothing."""
import torch
import torch.distributed as dist
from tests.gemm.bf16_linear_gather_n896_k35840_ranks8.multi_gpu8 import run_on_8_gpus

SERVED = (1, 8, 12, 16, 17, 32, 33, 40, 48, 49, 65, 80, 81, 96, 128, 129, 144, 192)


def worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8 import ProjComm, context_proj_gather_norm, host
    from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8.reference import K, N_FULL, N_SHARD
    from flydsl.compiler.jit_function import CompiledFunction, MlirCompiler

    warm_up.set_auto(True)
    warm_up.set_workers(4)
    dev = torch.device("cuda", rank)
    launches = []
    launch = CompiledFunction.__call__

    def spy(self, *args):
        if warm_up._in_warm_up.get():
            launches.append(1)
        return launch(self, *args)

    CompiledFunction.__call__ = spy
    comm = ProjComm(rank, dist.group.WORLD, dev)
    w = host.pack_weight(torch.zeros(N_SHARD, K, dtype=torch.bfloat16, device=dev))
    nw = torch.ones(N_FULL, dtype=torch.bfloat16, device=dev)
    real = MlirCompiler.compile
    compiles = []

    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)

    try:
        aux = [torch.zeros(1, K // 5, dtype=torch.bfloat16, device=dev) for _ in range(5)]
        xp = host.packed_rows_buffer(1, dev)
        host.pack_aux(aux, xp, 1)
        context_proj_gather_norm(comm, xp, w, nw, 1)
        torch.cuda.synchronize()
        if launches:
            errors.append(f"rank {rank}: {len(launches)} launches inside a warm-up")
        late0 = warm_up.late_compiles()
        MlirCompiler.compile = counting
        for m in SERVED:
            aux = [torch.randn(m, K // 5, dtype=torch.bfloat16, device=dev) for _ in range(5)]
            xp = host.packed_rows_buffer(m, dev)
            host.pack_aux(aux, xp, m)
            context_proj_gather_norm(comm, xp, w, nw, m)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = context_proj_gather_norm(comm, xp, w, nw, m)
            g.replay()
            torch.cuda.synchronize()
            if compiles:
                errors.append(f"rank {rank}: {len(compiles)} compiles at {m} rows")
                compiles.clear()
            if not bool(torch.isfinite(out).all()):
                errors.append(f"rank {rank}: non-finite output at {m} rows")
        if warm_up.late_compiles() != late0:
            errors.append(f"rank {rank}: {warm_up.late_compiles() - late0} late compiles under an entry")
    finally:
        MlirCompiler.compile = real
        CompiledFunction.__call__ = launch
    comm.close()


def test_warm_up_then_no_compile():
    run_on_8_gpus(worker, timeout=110)
