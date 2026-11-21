# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 on eight ranks: the first call of each mode warms every variant up (no launch inside a warm-up); then the
served row counts, eager and in a graph, compile nothing."""
import torch
import torch.distributed as dist

from tests.norm.allreduce_attnres_rmsnorm_h7168_ranks8.multi_gpu8 import run_on_8_gpus
from tests.norm.allreduce_attnres_rmsnorm_h7168_ranks8.workload import SERVED_ROWS, make

MODES = (  # (prefix, bank rows used, write idx, output norm, reduce, bank, packed output)
    (True, 3, -1, True, True, True, False),
    (True, 8, -1, False, True, True, False),
    (False, 0, 0, True, True, True, False),
    (True, 0, -1, True, True, False, False),
    (True, 4, -1, True, False, True, False),
    (True, 4, -1, True, True, True, True),
)


def worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8 import AttnResComm
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
    comm = AttnResComm(rank, dist.group.WORLD, dev)
    real = MlirCompiler.compile
    compiles = []

    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)

    try:
        for mode in MODES:
            hp, nb, widx, on, rd, hb, pk = mode
            x = torch.zeros((1, 7168), dtype=torch.bfloat16, device=dev)
            nw = torch.ones(7168, dtype=torch.bfloat16, device=dev)
            bank = torch.zeros((1, 8, 7168), dtype=torch.bfloat16, device=dev)
            comm.allreduce_attnres_rmsnorm(x, x.clone() if hp else None, bank if hb else None, nw, nw, nb, widx,
                                           nw if on else None, reduce=rd, out_packed=pk)
        torch.cuda.synchronize()
        if launches:
            errors.append(f"rank {rank}: {len(launches)} launches inside a warm-up")
        late0 = warm_up.late_compiles()
        MlirCompiler.compile = counting
        for m in SERVED_ROWS:
            parts, prefix, bank, nw, qk, ow = make(m, m, dev)
            for hp, nb, widx, on, rd, hb, pk in MODES:
                args = (parts[rank], prefix.clone() if hp else None, bank if hb else None, nw, qk, nb, widx,
                        ow if on else None)
                comm.allreduce_attnres_rmsnorm(*args, reduce=rd, out_packed=pk)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out, _ = comm.allreduce_attnres_rmsnorm(*args, reduce=rd, out_packed=pk)
                g.replay()
                torch.cuda.synchronize()
                if compiles:
                    errors.append(f"rank {rank}: {len(compiles)} compiles at {m} rows, mode {(hp, nb, widx, on, rd, hb, pk)}")
                    compiles.clear()
                if not pk and not bool(torch.isfinite(out).all()):
                    errors.append(f"rank {rank}: non-finite output at {m} rows")
        if warm_up.late_compiles() != late0:
            errors.append(f"rank {rank}: {warm_up.late_compiles() - late0} late compiles under an entry")
    finally:
        MlirCompiler.compile = real
        CompiledFunction.__call__ = launch
    comm.close()


def test_warm_up_then_no_compile():
    run_on_8_gpus(worker, timeout=900)
