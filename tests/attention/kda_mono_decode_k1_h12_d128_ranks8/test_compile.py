# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4): the first call of a mode (qlen, blocks, delta, write block,
state length) compiles every served width, no kernel runs in the warm-up, and a later call at each width, eager and in a
CUDA graph, compiles nothing."""

import pytest
import torch
from amd_kernels import warm_up
from flydsl.compiler.jit_function import CompiledFunction, MlirCompiler

from amd_kernels.attention.kda_mono_decode_k1_h12_d128_ranks8 import Workspace, kda_mono_decode_k1

from .workload import clone_state, make, require_gfx950

MODES = [(6, 4, False), (6, 1, True), (4, 8, False), (3, 4, False)]  # (qlen, blocks, block write)


def call(ws, layer, s, L, nb, widx):
    return kda_mono_decode_k1(
        ws, layer, s["prefix"], s["delta"], s["blocks"], s["ares_nw"], s["ares_qk"], s["in_nw"], s["w_in"], s["w_fb"],
        s["conv_w"], s["conv_state"], s["a_log"], s["dt_bias"], s["on_w"], s["rstate"], s["st_idx"], s["num_acc"],
        qlen=L, nblocks=nb, write_idx=nb if widx else -1,
    )  # fmt: skip


@pytest.mark.parametrize("L,nb,widx", MODES)
def test_warm_up_covers_every_served_width(monkeypatch, L, nb, widx):
    require_gfx950()
    warm_up.set_auto(True)
    warm_up.set_workers(2)
    try:
        launches, compiles = [], []
        launch = CompiledFunction.__call__
        real = MlirCompiler.compile

        def spy(self, *args):
            if warm_up._in_warm_up.get():
                launches.append(1)
            return launch(self, *args)

        monkeypatch.setattr(CompiledFunction, "__call__", spy)
        ws = Workspace("cuda")
        s = make(L, L, nb, seed=1)
        ws.begin_step()
        call(ws, 0, clone_state(s), L, nb, widx)
        torch.cuda.synchronize()
        assert not launches, "a warm-up launched a kernel"
        monkeypatch.setattr(MlirCompiler, "compile", lambda *a, **k: (compiles.append(1), real(*a, **k))[1])
        for S in range(L, 17, L):
            x = make(S, L, nb, seed=S)
            ws.begin_step()
            call(ws, 0, clone_state(x), L, nb, widx)
            g = torch.cuda.CUDAGraph()
            y = clone_state(x)
            with torch.cuda.graph(g):
                call(ws, 1, y, L, nb, widx)
            g.replay()
            torch.cuda.synchronize()
            assert not compiles, f"{len(compiles)} compiles at {S} rows"
    finally:
        warm_up.set_auto(False)
