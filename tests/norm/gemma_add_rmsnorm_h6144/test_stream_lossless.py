# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Exact transport in the unregistered BF16 stream: contract, warm-up and reuse."""
import pytest
import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_4_gpus
from tests.norm.gemma_add_rmsnorm_h6144.test_staged_ar_add_rmsnorm import (
    _bits, _inputs, _residual, _worker,
)
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import H


# Every rows-per-workgroup edge of the two-phase coded tier (R2 / 4 / 8 / 16) and its ends.
SCATTER_EDGES = (321, 448, 449, 480, 481, 512, 513, 641, 1024, 1025, 1281, 2048, 2049, 2051, 3253, 4095)


def test_stream_lossless_reference_replay_compile_and_mfma():
    # No registration: these are the staged kernels used by mixed serving steps.
    run_on_4_gpus(_worker, SCATTER_EDGES, True, True, None, False, False)


def test_stream_lossless_strided_inputs_and_residuals():
    run_on_4_gpus(_worker, (481, 1025, 2049, 3252, 4095), True, True, None, False, False, True)


def _switch_worker(rank, errors):
    """One unregistered input and raw arena, reused across q/R and tier changes."""
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.ar_add_rmsnorm_stream import sub_rows
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import allreduce_rows_reference
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from tests.mfma_spinner import mfma_spinner

    dev = torch.device('cuda', rank)
    w = torch.randn(H, generator=torch.Generator().manual_seed(96)).bfloat16().to(dev)
    comm = arn.staged_comm(rank, dist.group.WORLD, dev)
    comm.warm_up(w, 1e-6)
    # the small calls take the registered batch codec, the large ones the staged codec, on one communicator and raw
    # arena
    comm.stream_pull_max_rows = 2048
    arena = torch.empty((4095, H), dtype=torch.bfloat16, device=dev)
    sizes = (1752, 2048, 2049, 2560, 2561, 2816, 2817, 3072, 3073, 3252, 3253, 4095, 2049, 1752)
    every = [_inputs(m, i, dev, True) for i, m in enumerate(sizes)]
    residuals = [_residual(m, dev, True) for m in sizes]
    spin = mfma_spinner(128)
    side = torch.cuda.Stream(priority=-1)
    # Compile the independent one-GPU reference before guarding the op's warm-up.
    an.gemma_add_rmsnorm(every[0][rank], residuals[0], w, 1e-6)
    real = MlirCompiler.compile
    def late(*args, **kwargs):
        raise AssertionError('compile after library warm-up')
    MlirCompiler.compile = late
    try:
        outputs = []
        graph = torch.cuda.CUDAGraph()
        with comm.capture(), torch.cuda.graph(graph):
            for m, inputs, residual in zip(sizes, every, residuals):
                arena[:m].copy_(inputs[rank])
                outputs.append(comm.ar_add_rmsnorm(arena[:m], residual, w, 1e-6))
                arena.fill_(float('nan'))
        delta = torch.zeros_like(comm.stream_counts)
        for m in sizes:
            q = (m + 3) // 4
            rpw = comm.lossless_scatter_rows_per_wg(m) if m >= 2049 else comm.rows_per_wg(m)
            assert 4 * (((q + rpw - 1) // rpw + 7) // 8 * 8) <= 256
            step = rpw if m <= 2048 else sub_rows(rpw)
            for c in range(4):
                for j in range(0, q, step):
                    if c * q + j < m:
                        delta[c * q + j] += 1
        for rep in range(3):
            if rep:
                for inputs in every:
                    for x in inputs:
                        x.neg_()
            torch.cuda.synchronize()
            dist.barrier()
            before = comm.stream_counts.clone()
            if rank == 3:
                spin(200000, side)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(comm.stream_counts - before, delta), (rank, rep)
            for m, inputs, residual, got in zip(sizes, every, residuals, outputs):
                want = an.gemma_add_rmsnorm(allreduce_rows_reference(inputs), residual, w, 1e-6)
                assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(got, want)), (rank, rep, m)
    finally:
        MlirCompiler.compile = real


def test_stream_lossless_shape_switch_and_immediate_input_reuse():
    run_on_4_gpus(_switch_worker, timeout=300)


def _streams_worker(rank, errors, m):
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.ar_add_rmsnorm_stream import sub_rows
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import allreduce_rows_reference
    from flydsl.compiler.jit_function import MlirCompiler

    dev = torch.device('cuda', rank)
    w = torch.randn(H, generator=torch.Generator().manual_seed(81)).bfloat16().to(dev)
    comms = [arn.staged_comm(rank, dist.group.WORLD, dev) for _ in range(2)]
    assert comms[0].stream_input_raw_bufs[rank] != comms[1].stream_input_raw_bufs[rank]
    for comm in comms:
        comm.warm_up(w, 1e-6)
    every = [_inputs(m, i, dev, True) for i in range(2)]
    inputs = [e[rank] for e in every]
    backing = torch.empty((m, H + 8), dtype=torch.bfloat16, device=dev)
    residual = backing[:, :H]
    residual.copy_(_residual(m, dev))
    # the contract chain: the reference all-reduce, then the one-GPU norm
    refs = [an.gemma_add_rmsnorm(allreduce_rows_reference(e), residual, w, 1e-6) for e in every]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in range(2)]
    graphs, outputs = [], []
    real = MlirCompiler.compile
    def late(*args, **kwargs):
        raise AssertionError('compile after library warm-up')
    MlirCompiler.compile = late
    try:
        for i, comm in enumerate(comms):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=streams[i]):
                outputs.append(comm.ar_add_rmsnorm(inputs[i], residual, w, 1e-6))
            graphs.append(graph)
        for comm in comms:
            comm.stream_counts.fill_(2**31 - 2)
        torch.cuda.synchronize()
        for rep in range(4):
            for i, graph in enumerate(graphs):
                with torch.cuda.stream(streams[i]):
                    graph.replay()
            torch.cuda.synchronize()
            for got, ref in zip(outputs, refs):
                assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(got, ref)), (rank, rep)
        q = (m + 3) // 4
        for comm in comms:
            step = sub_rows(comm.lossless_scatter_rows_per_wg(m))
            assert comm.stream_counts[:m].tolist() == [
                -2**31 + 2 if (r % q) % step == 0 else 2**31 - 2 for r in range(m)]
    finally:
        MlirCompiler.compile = real


@pytest.mark.parametrize('rows', (481, 1024, 2560, 4095))
def test_stream_lossless_independent_streams_and_generation_wrap(rows):
    run_on_4_gpus(_streams_worker, rows, timeout=240)


def _default_worker(rank, errors):
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm import staged_host as sh
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import allreduce_rows_reference
    from flydsl.compiler.jit_function import MlirCompiler
    dev = torch.device('cuda', rank)
    w = torch.randn(H, generator=torch.Generator().manual_seed(81)).bfloat16().to(dev)
    comm = arn.staged_comm(rank, dist.group.WORLD, dev)
    comm.warm_up(w, 1e-6)
    x = torch.zeros((1, H), dtype=torch.bfloat16, device=dev)
    an.gemma_add_rmsnorm(x, x, w, 1e-6)  # compile the reference chain's norm before the guard
    names = []
    real_run, real_compile = sh._run_compiled, MlirCompiler.compile
    def spy(exe, *args):
        names.append(exe.kernel_name)
        return real_run(exe, *args)
    def late(*args, **kwargs):
        raise AssertionError('compile after default library warm-up')
    sh._run_compiled, MlirCompiler.compile = spy, late
    try:
        for m in (320, 321, 480, 481, 512, 513, 1024, 1025, 2048, 2049, 3252, 3253, 4095):
            every = _inputs(m, 0, dev, True)
            x = every[rank]
            residual = _residual(m, dev)
            names.clear()
            a = comm.ar_add_rmsnorm(x, residual, w, 1e-6)
            assert ('lossless12' in names[-1]) == (m >= 321), (m, names)
            assert ('lossless12_scatter_stream' in names[-1]) == (m >= 321), (m, names)
            if m >= 321:
                want = 2 if m <= 512 else 4 if m <= 1024 else 8 if m <= 2048 else 16
                assert names[-1].endswith(f'_r{want}'), (m, names)
            b = an.gemma_add_rmsnorm(allreduce_rows_reference(every), residual, w, 1e-6)
            torch.cuda.synchronize()
            assert all(torch.equal(_bits(u), _bits(v)) for u, v in zip(a, b)), (rank, m)
    finally:
        sh._run_compiled, MlirCompiler.compile = real_run, real_compile


def test_stream_lossless_default_dispatch():
    run_on_4_gpus(_default_worker, timeout=240)
