# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 at two ranks: after ``DecoderArNorm``'s warm-ups (both communicators on one buffer pair), no row count of
1-4095 reaches a kernel variant the warm-up did not compile, and probes at every tier edge compile nothing, eager and
in graphs, both code layouts and every output."""

import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_2_gpus

H = 6144
EPS = 1e-6


def _worker(rank, errors):
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm_ranks2 as arn
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm_ranks2 import host

    dev = torch.device("cuda", rank)
    w = torch.ones(H, dtype=torch.bfloat16, device=dev)
    real_kernel, warm = host._kernel, set()

    def spy(*key):
        warm.add(key)
        return real_kernel(*key)

    host._kernel = spy
    try:
        comms = arn.DecoderArNorm(rank, dist.group.WORLD, dev, w, EPS)
    finally:
        host._kernel = real_kernel
    for rows in range(1, host.MAX_ROWS + 1):
        for out, slots in (("bf16", False), ("tiled", False), ("rowmajor", False), ("tiled", True),
                           ("rowmajor", True)):
            rpb, _, _, coded = host.config(rows, out != "bf16", slots)
            if (out, slots, rpb, coded) not in warm:
                errors.append(f"rank {rank} rows {rows} {out} slots {slots}: variant {rpb, coded} not warmed up")
                return
    probes = sorted({1, 2, 8, 48, 49, 64, 256, 257, 1024, 1025, 2048, 2049, 4095, 4096, 7700}
                    | set(range(1, host.MAX_ROWS + 1, 97)))

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    real = MlirCompiler.compile
    MlirCompiler.compile = unexpected
    try:
        for rows in probes:
            x = torch.zeros((rows, H), dtype=torch.bfloat16, device=dev)
            slots = torch.zeros((rows, 4, H), dtype=torch.bfloat16, device=dev)
            r = torch.zeros_like(x)

            def calls():
                comms.staged.ar_add_rmsnorm(x, r, w, EPS)
                for tile in (0, 16, 32, 64):
                    comms.staged.ar_add_rmsnorm_mxfp8(x, r, w, EPS, tile)
                    comms.slots.slots4_ar_add_rmsnorm_mxfp8(slots, x, r, w, EPS, tile)

            calls()
            g = torch.cuda.CUDAGraph()
            with comms.capture(), torch.cuda.graph(g):
                calls()
            g.replay()
            torch.cuda.synchronize()
    except AssertionError as e:
        errors.append(f"rank {rank}: {e}")
    finally:
        MlirCompiler.compile = real


def test_ranks2_no_compile_after_warm_up():
    run_on_2_gpus(_worker, timeout=1800)
