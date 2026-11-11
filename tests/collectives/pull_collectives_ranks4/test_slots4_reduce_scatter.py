# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract, changed-input graph replay, double-buffer reuse and warm-up.

Every rank generates all rank operands deterministically; no reference
communication or sum-order ambiguity. Total rows include non-power-of-two
shards, the SP boundary and the served maximum. Every written byte is checked.
"""
import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_4_gpus

H = 6144


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _expect(errors, tag, got, want):
    if not torch.equal(_bits(got), _bits(want)):
        n = int((_bits(got) != _bits(want)).sum())
        errors.append(f"{tag}: {n} bytes differ")


def _inputs(rows, p, dev, step):
    g = torch.Generator(device=dev).manual_seed(rows * 19 + p + step * 1000)
    s = torch.randn((rows, 4, H), generator=g, device=dev).bfloat16()
    h = torch.randn((rows, H), generator=g, device=dev).bfloat16()
    # Signed zeros and retained BF16 denormals, including cancellation.
    s[:, :, :2] = -0.0
    s[:, p, 2:4] = 2.0**-133
    h[:, :4] = 0.0
    return s, h


def _worker(rank, errors, rows_list):
    from amd_kernels.collectives.pull_collectives_ranks4 import SlotsReduceScatterComm
    from amd_kernels.collectives.pull_collectives_ranks4.reference import slots4_reduce_scatter_reference
    from flydsl.compiler.jit_function import MlirCompiler

    dev = torch.device('cuda', rank)
    comm = SlotsReduceScatterComm(rank, dist.group.WORLD, dev)
    comm.warm_up()
    compiles = []
    real = MlirCompiler.compile
    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)
    MlirCompiler.compile = counting
    try:
        for rows in rows_list:
            xs = [_inputs(rows, p, dev, 0) for p in range(4)]
            s, h = xs[rank]
            want = slots4_reduce_scatter_reference([x[0] for x in xs], [x[1] for x in xs], rank)
            guard = torch.full((rows//4*H+16,), 3.25, dtype=torch.bfloat16, device=dev)
            out = guard[8:-8].view(rows//4,H)
            out.fill_(float('nan'))
            comm.slots4_reduce_scatter(s, h, out)
            torch.cuda.synchronize()
            _expect(errors, f'{rank} rows{rows} eager', out, want)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                comm.slots4_reduce_scatter(s, h, out)
            for step in range(3):
                if step == 1:
                    xs = [_inputs(rows, p, dev, 1) for p in range(4)]
                    s.copy_(xs[rank][0]); h.copy_(xs[rank][1])
                    want = slots4_reduce_scatter_reference([x[0] for x in xs], [x[1] for x in xs], rank)
                out.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                _expect(errors, f'{rank} rows{rows} replay{step}', out, want)
                if not bool((guard[:8] == 3.25).all() & (guard[-8:] == 3.25).all()):
                    errors.append(f'{rank} rows{rows}: output guard overwritten')
            del graph, xs, s, h, out, want
            torch.cuda.empty_cache()
        if compiles:
            errors.append(f'{rank}: {len(compiles)} compiles after warmup')
    finally:
        MlirCompiler.compile = real


def test_slots4_reduce_scatter_contract_graph_warmup():
    # Uneven chunk sizes exercise the last vector of the narrower pull stripes.
    run_on_4_gpus(_worker, [4, 20, 132, 4096, 4100, 8192, 16384, 32764, 32768])


def _norm_worker(rank, errors, row_counts, spin=False, max_rows=32768):
    from amd_kernels.collectives.pull_collectives_ranks4 import SlotsReduceScatterComm
    from amd_kernels.collectives.pull_collectives_ranks4.reference import (
        slots4_reduce_scatter_reference, reduce_scatter_reference,
    )
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
        CODE_MISMATCH, MXFP8_COS, SCALE_MISMATCH, unshuffle_scales, add_rmsnorm_reference, compare_mxfp8,
    )
    from flydsl.compiler.jit_function import MlirCompiler
    dev = torch.device('cuda', rank)
    comm = SlotsReduceScatterComm(rank, dist.group.WORLD, dev, max_rows=max_rows)
    if spin:
        from tests.mfma_spinner import mfma_spinner
        spinner = mfma_spinner(32 if spin is True else int(spin))
        side = torch.cuda.Stream(priority=-1)
    comm.warm_up()
    compiles = []
    real = MlirCompiler.compile
    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)
    MlirCompiler.compile = counting
    try:
        for rows in row_counts:
            shard = rows // 4
            xs = [_inputs(rows, p, dev, 0) for p in range(4)]
            s, h = xs[rank]
            want = slots4_reduce_scatter_reference([x[0] for x in xs], [x[1] for x in xs], rank)
            dense = comm.reduce_scatter(h)
            torch.cuda.synchronize()
            _expect(errors, f'{rank} dense{rows}', dense,
                    reduce_scatter_reference([x[1] for x in xs], rank))
            r = torch.randn((shard,H), device=dev).bfloat16()
            w = (torch.randn(H, device=dev)*0.2).bfloat16()
            high, res_ref = add_rmsnorm_reference(want, r, w, 1e-6)
            def valid(got):
                q, sc, res = got
                return q[:shard].view(torch.uint8), unshuffle_scales(sc, shard), res
            eager = comm.slots4_reduce_scatter_add_rmsnorm_mxfp8(s,h,r,w,1e-6)
            torch.cuda.synchronize()
            _expect(errors, f'{rank} norm{rows} residual', eager[2],res_ref)
            q,sc,_ = valid(eager)
            stats = compare_mxfp8(q,sc,high)
            if (stats['scale_max_diff'] > 1 or stats['scale_mismatch'] > SCALE_MISMATCH
                    or stats['code_mismatch'] > CODE_MISMATCH or stats['code_far'] != 0
                    or stats['cos'] < MXFP8_COS):
                errors.append(f'{rank} norm{rows}: {stats}')
            ref = [t.clone() for t in eager]
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                got = comm.slots4_reduce_scatter_add_rmsnorm_mxfp8(s,h,r,w,1e-6)
            for _ in range(3):
                got[0].view(torch.uint8).fill_(255); got[1].fill_(255); got[2].fill_(float('nan'))
                if spin:
                    side.wait_stream(torch.cuda.current_stream())
                    spinner(1000000,side)
                g.replay()
                torch.cuda.synchronize()
                for k,(a,b) in enumerate(zip(valid(got),valid(ref))):
                    _expect(errors,f'{rank} norm{rows} replay{k}',a,b)
            del g,got,eager,ref
            del xs,s,h,want,r,w,high,res_ref
            torch.cuda.empty_cache()
        if compiles:
            errors.append(f'{rank}: {len(compiles)} compiles after warmup')
    finally:
        MlirCompiler.compile = real


def test_slots4_norm_contract_graph_warmup():
    run_on_4_gpus(_norm_worker, [4, 20, 132, 4096, 4100, 8192, 16384, 32764, 32768])


def test_slots4_norm_next_to_mfma():
    run_on_4_gpus(_norm_worker, [4096], True)


def test_slots4_norm_next_to_mfma_at_max_rows():
    # The largest SP payload and a resident MFMA workgroup on every CU.
    # Unlike the short 4K test, this keeps the chunk handoffs busy throughout the transfer.
    run_on_4_gpus(_norm_worker, [32768], 256)


def _late_worker(rank, errors, rows_list):
    """One rank starts each call late (a 300 us kernel ahead of it on its stream), so the
    peers' pullers wait on its per-chunk flags while their own stagers have finished."""
    from amd_kernels.collectives.pull_collectives_ranks4 import SlotsReduceScatterComm
    from amd_kernels.collectives.pull_collectives_ranks4.reference import slots4_reduce_scatter_reference
    from tests.mfma_spinner import mfma_spinner
    dev = torch.device('cuda', rank)
    comm = SlotsReduceScatterComm(rank, dist.group.WORLD, dev)
    comm.warm_up()
    spinner = mfma_spinner(32)
    stream = torch.cuda.current_stream()
    for rows in rows_list:
        for step in range(8):
            xs = [_inputs(rows, p, dev, step) for p in range(4)]
            want = slots4_reduce_scatter_reference([x[0] for x in xs], [x[1] for x in xs], rank)
            dist.barrier()
            if rank == step % 4:
                spinner(30000, stream)
            out = comm.slots4_reduce_scatter(*xs[rank])
            torch.cuda.synchronize()
            _expect(errors, f'{rank} rows{rows} late rank {step % 4}', out, want)
            del xs, want, out
        torch.cuda.empty_cache()


def test_slots4_reduce_scatter_late_rank():
    run_on_4_gpus(_late_worker, [132, 4096, 32768])


def test_slots4_norm_small_pool_warmup():
    # The 32-group dense pull is unreachable in a four-row pool.
    run_on_4_gpus(_norm_worker, [4], False, 4)
