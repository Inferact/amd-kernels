# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 on 4 ranks: after ``MegaMoeTP.prepare`` of every local token count (with the fused-tail variants), a walk
over those counts, eager and under CUDA graph capture, compiles, loads and arms nothing: no FlyDSL compile, no launcher
run for the first time, no launch config armed."""

import pytest
import torch
import torch.distributed as dist
from tests.multi_gpu import WORLD, run_on_4_gpus

from amd_kernels.moe.mega_moe_tp import mega_moe_tp_supported

from .workload import SHAPES, build_weights, make_case, new_layer

MAX_LOCAL = 256
WALK = (1, 2, 3, 7, 8, 16, 31, 32, 63, 64, 100, 128, 129, 200, 255, 256)


def _worker(rank, errors, mode):
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels.moe.mega_moe_tp import fly_utils
    from amd_kernels.moe.mega_moe_tp import mega_moe_tp as host

    shape = SHAPES["m3"]
    wt = build_weights(shape, rank, WORLD, torch.device("cuda", rank))
    layer = new_layer(shape, wt, rank, WORLD, mode, MAX_LOCAL)
    tail = mode == "ag_rs"
    layer.prepare(range(1, MAX_LOCAL + 1), tail=tail, tail_bf16=tail)
    events = []
    real = (MlirCompiler.compile, fly_utils._compile_and_run, host._preload_compiled)

    def spy(what, fn):
        def wrapped(*a, **kw):
            events.append(what)
            return fn(*a, **kw)

        return wrapped

    MlirCompiler.compile = spy("compile", real[0])
    fly_utils._compile_and_run = spy("first run", real[1])
    host._preload_compiled = spy("arm", real[2])
    try:
        for m in WALK:
            c = make_case(shape, wt, rank, WORLD, mode, m * WORLD, "random", m)
            layer(c.x, c.w, c.ids)
            if tail and layer.tail_ok(m):
                res = torch.zeros_like(c.x)
                for bf16 in (False, True):
                    layer(c.x, c.w, c.ids, tail=(res, res, c.x[0]), bf16=bf16)
            torch.cuda.synchronize()
            dist.barrier()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                layer(c.x, c.w, c.ids)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                layer(c.x, c.w, c.ids)
            g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            if events:
                errors.append(f"rank {rank} {mode} m={m}: {sorted(set(events))} after prepare")
                events.clear()
            if layer.poll_errors():
                errors.append(f"rank {rank} {mode} m={m}: watchdog error")
                layer.clear_errors()
    finally:
        MlirCompiler.compile, fly_utils._compile_and_run, host._preload_compiled = real
    layer.close()


@pytest.mark.parametrize("mode", ["ag_rs", "ar"])
def test_mega_moe_tp_warm_up(mode):
    if not torch.cuda.is_available() or not mega_moe_tp_supported():
        pytest.skip("needs gfx950")
    run_on_4_gpus(_worker, mode, timeout=3600)
