# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4: the first call warms every row tier up (no launch inside a warm-up); then the served row counts, eager and in
a graph, compile nothing."""
import torch
from amd_kernels import warm_up
from amd_kernels.decoder.bf16_dense_mlp_h7168_i1792 import MlpWorkspace, dense_mlp, host, pack_down, pack_gate_up
from amd_kernels.decoder.bf16_dense_mlp_h7168_i1792.reference import HIDDEN, INTER
from flydsl.compiler.jit_function import CompiledFunction, MlirCompiler

SERVED = (1, 8, 12, 16, 17, 32, 33, 40, 48, 52, 60, 64, 65, 80, 81, 96, 97, 128, 129, 144, 145, 192)


def test_warm_up_then_no_compile():
    warm_up.set_auto(True)
    launches = []
    launch = CompiledFunction.__call__

    def spy(self, *args):
        if warm_up._in_warm_up.get():
            launches.append(1)
        return launch(self, *args)

    CompiledFunction.__call__ = spy
    real = MlirCompiler.compile
    compiles = []

    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)

    try:
        dev = torch.device("cuda")
        gu = pack_gate_up(torch.zeros(2 * INTER, HIDDEN, dtype=torch.bfloat16, device=dev))
        dn = pack_down(torch.zeros(HIDDEN, INTER, dtype=torch.bfloat16, device=dev))
        ws = MlpWorkspace(dev)
        dense_mlp(ws, host.packed_rows_buffer(1, dev), gu, dn, 1)
        torch.cuda.synchronize()
        assert not launches, f"{len(launches)} launches inside a warm-up"
        late0 = warm_up.late_compiles()
        MlirCompiler.compile = counting
        for rows in SERVED:
            xp = host.packed_rows_buffer(rows, dev)
            xp.zero_()
            dense_mlp(ws, xp, gu, dn, rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = dense_mlp(ws, xp, gu, dn, rows)
            graph.replay()
            torch.cuda.synchronize()
            assert not compiles, f"{len(compiles)} compiles at {rows} rows"
            assert bool(torch.isfinite(out).all())
        assert warm_up.late_compiles() == late0
    finally:
        MlirCompiler.compile = real
        CompiledFunction.__call__ = launch
        warm_up.set_auto(False)
