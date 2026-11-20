# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks: contract, changed-input graph replay, double-buffer reuse and warm-up.

Every rank generates all rank operands deterministically; no reference
communication or sum-order ambiguity. Total rows include non-power-of-two
shards, the SP boundary and the served maximum. Every written byte is checked.
"""
import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_2_gpus

H = 6144


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _expect(errors, tag, got, want):
    if not torch.equal(_bits(got), _bits(want)):
        n = int((_bits(got) != _bits(want)).sum())
        errors.append(f"{tag}: {n} bytes differ")


def _inputs(rows, p, dev, step, owner=False):
    """(slots, shared) of rank p; ``owner``: shared [rows / 2, H] (rank p's whole shared expert on its own rows)."""
    g = torch.Generator(device=dev).manual_seed(rows * 19 + p + step * 1000)
    s = torch.randn((rows, 4, H), generator=g, device=dev).bfloat16()
    h = torch.randn((rows // 2 if owner else rows, H), generator=g, device=dev).bfloat16()
    # Signed zeros and retained BF16 denormals, including cancellation.
    s[:, :, :2] = -0.0
    s[:, p, 2:4] = 2.0**-133
    h[:, :4] = 0.0
    return s, h


def _worker(rank, errors, rows_list, owner=False):
    from amd_kernels.collectives.pull_collectives_ranks2 import SlotsReduceScatterComm
    from amd_kernels.collectives.pull_collectives_ranks2.reference import slots4_reduce_scatter_reference
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
            xs = [_inputs(rows, p, dev, 0, owner) for p in range(2)]
            s, h = xs[rank]
            want = slots4_reduce_scatter_reference([x[0] for x in xs], [x[1] for x in xs], rank)
            guard = torch.full((rows//2*H+16,), 3.25, dtype=torch.bfloat16, device=dev)
            out = guard[8:-8].view(rows//2,H)
            out.fill_(float('nan'))
            comm.slots4_reduce_scatter(s, h, out)
            torch.cuda.synchronize()
            _expect(errors, f'{rank} rows{rows} eager', out, want)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                comm.slots4_reduce_scatter(s, h, out)
            for step in range(3):
                if step == 1:
                    xs = [_inputs(rows, p, dev, 1, owner) for p in range(2)]
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
    # Uneven chunk sizes exercise the last vector of the narrower pull stripes; 4096..32768 are the served tokens.
    run_on_2_gpus(_worker, [2, 4, 20, 66, 132, 4096, 4100, 8192, 16384, 32766, 32768])


def test_slots4_reduce_scatter_owner_form():
    run_on_2_gpus(_worker, [2, 4, 20, 66, 132, 4096, 4100, 8192, 16384, 32766, 32768], True)


def _norm_worker(rank, errors, row_counts, spin=False, max_rows=32768, coded=True, owner=False):
    from amd_kernels.collectives.pull_collectives_ranks2 import SlotsReduceScatterComm
    from amd_kernels.collectives.pull_collectives_ranks2.reference import (
        slots4_reduce_scatter_reference, reduce_scatter_reference,
    )
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
        CODE_MISMATCH, MXFP8_COS, SCALE_MISMATCH, unshuffle_scales, add_rmsnorm_reference, compare_mxfp8,
    )
    from flydsl.compiler.jit_function import MlirCompiler
    dev = torch.device('cuda', rank)
    comm = SlotsReduceScatterComm(rank, dist.group.WORLD, dev, max_rows=max_rows, coded=coded)
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
            shard = rows // 2
            xs = [_inputs(rows, p, dev, 0, owner) for p in range(2)]
            s, h = xs[rank]
            want = slots4_reduce_scatter_reference([x[0] for x in xs], [x[1] for x in xs], rank)
            if not owner:
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
            two = comm.slots4_reduce_scatter_add_rmsnorm_mxfp8(s,h,r,w,1e-6,fused=False)
            torch.cuda.synchronize()
            for k,(x,y) in enumerate(zip(valid(two),valid(ref))):
                _expect(errors,f'{rank} norm{rows} two launches vs one, field {k}',x,y)
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
    run_on_2_gpus(_norm_worker, [2, 20, 66, 132, 4096, 4100, 8192, 16384, 32766, 32768])


def test_slots4_norm_bf16_link_contract():
    run_on_2_gpus(_norm_worker, [2, 66, 4100, 16384, 32768], False, 32768, False)


def test_slots4_norm_owner_form():
    """Owner-form shared [m, H]: coded and BF16-link tails, one and two launches, graph replay."""
    run_on_2_gpus(_norm_worker, [2, 20, 66, 132, 4096, 4100, 8192, 16384, 32766, 32768], False, 32768, True, True)
    run_on_2_gpus(_norm_worker, [2, 66, 4100, 32768], False, 32768, False, True)


def test_slots4_norm_owner_form_next_to_mfma():
    run_on_2_gpus(_norm_worker, [32768], 256, 32768, True, True)


def test_slots4_norm_next_to_mfma():
    run_on_2_gpus(_norm_worker, [4096], True)


def test_slots4_norm_next_to_mfma_at_max_rows():
    # The largest SP payload and a resident MFMA workgroup on every CU.
    # Unlike the short 4K test, this keeps the chunk handoffs busy throughout the transfer.
    run_on_2_gpus(_norm_worker, [32768], 256)


def _late_worker(rank, errors, rows_list, coded=True, owner=False):
    """One rank starts each call late (a 300 us kernel ahead of it on its stream), so the
    peer's pullers wait on its per-chunk tickets while their own stagers have finished."""
    from amd_kernels.collectives.pull_collectives_ranks2 import SlotsReduceScatterComm
    from amd_kernels.collectives.pull_collectives_ranks2.reference import (
        reduce_scatter_reference, slots4_reduce_scatter_reference,
    )
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import unshuffle_scales
    from tests.mfma_spinner import mfma_spinner
    dev = torch.device('cuda', rank)
    comm = SlotsReduceScatterComm(rank, dist.group.WORLD, dev, coded=coded)
    comm.warm_up()
    spinner = mfma_spinner(32)
    stream = torch.cuda.current_stream()
    for rows in rows_list:
        for step in range(8):
            xs = [_inputs(rows, p, dev, step, owner) for p in range(2)]
            want = slots4_reduce_scatter_reference([x[0] for x in xs], [x[1] for x in xs], rank)
            dist.barrier()
            if rank == step % 2:
                spinner(30000, stream)
            out = comm.slots4_reduce_scatter(*xs[rank])
            torch.cuda.synchronize()
            _expect(errors, f'{rank} rows{rows} late rank {step % 2}', out, want)
            # the one-launch tail and the dense reduce-scatter, the other rank late
            r = torch.zeros((rows // 2, H), dtype=torch.bfloat16, device=dev)
            w = torch.ones(H, dtype=torch.bfloat16, device=dev)
            two = comm.slots4_reduce_scatter_add_rmsnorm_mxfp8(*xs[rank], r, w, 1e-6, fused=False)
            torch.cuda.synchronize()
            dist.barrier()
            if rank != step % 2:
                spinner(30000, stream)
            one = comm.slots4_reduce_scatter_add_rmsnorm_mxfp8(*xs[rank], r, w, 1e-6)
            dense = comm.reduce_scatter(xs[rank][1]) if not owner else None
            torch.cuda.synchronize()
            shard = rows // 2
            def valid(got):
                return got[0][:shard].view(torch.uint8), unshuffle_scales(got[1], shard), got[2]
            for k, (a, b) in enumerate(zip(valid(one), valid(two))):
                _expect(errors, f'{rank} rows{rows} late tail field {k}', a, b)
            if not owner:
                _expect(errors, f'{rank} rows{rows} late dense', dense,
                        reduce_scatter_reference([x[1] for x in xs], rank))
            del xs, want, out, one, two, dense
        torch.cuda.empty_cache()


def test_slots4_reduce_scatter_late_rank():
    run_on_2_gpus(_late_worker, [132, 4096, 32768])


def test_slots4_owner_form_late_rank():
    run_on_2_gpus(_late_worker, [132, 4096, 32768], True, True)


def _mixed_worker(rank, errors, rows_list):
    """Legacy and owner-form calls interleaved on one communicator (they share its tickets and pool slots); the
    owner form equals the legacy form on a shared tensor that is zero outside this rank's block."""
    from amd_kernels.collectives.pull_collectives_ranks2 import SlotsReduceScatterComm
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import unshuffle_scales
    dev = torch.device('cuda', rank)
    comm = SlotsReduceScatterComm(rank, dist.group.WORLD, dev)
    comm.warm_up()
    for rows in rows_list:
        shard = rows // 2
        s, h = _inputs(rows, rank, dev, 3, True)
        full = torch.zeros((rows, H), dtype=torch.bfloat16, device=dev)
        full[rank * shard:(rank + 1) * shard] = h
        r = torch.randn((shard, H), device=dev).bfloat16()
        w = (torch.randn(H, device=dev) * 0.2).bfloat16()
        def valid(got):
            return got[0][:shard].view(torch.uint8), unshuffle_scales(got[1], shard), got[2]
        for fused in (True, False):
            a = comm.slots4_reduce_scatter_add_rmsnorm_mxfp8(s, h, r, w, 1e-6, fused)
            b = comm.slots4_reduce_scatter_add_rmsnorm_mxfp8(s, full, r, w, 1e-6, fused)
            torch.cuda.synchronize()
            for k, (x, y) in enumerate(zip(valid(a), valid(b))):
                _expect(errors, f'{rank} rows{rows} fused{fused} owner vs zero-padded legacy field {k}', x, y)
        _expect(errors, f'{rank} rows{rows} rs owner vs legacy', comm.slots4_reduce_scatter(s, h),
                comm.slots4_reduce_scatter(s, full))
        torch.cuda.empty_cache()


def test_slots4_owner_form_equals_zero_padded_legacy():
    run_on_2_gpus(_mixed_worker, [2, 66, 4100, 16384])


def test_slots4_norm_small_pool_warmup():
    run_on_2_gpus(_norm_worker, [2], False, 2)


def _codec_rows(rows, p, dev, kind, seed=0):
    """Slot 0 carries the values (slots 1-3 and shared zero: the slot sum is slot 0 with -0 -> +0), shaped to hit
    every path of the code: signed zeros, one exception per lane (<= 6 per virtual wave), several per lane and more
    than six per wave (raw virtual waves), NaN / Inf / denormals, wide exponent spreads."""
    g = torch.Generator(device=dev).manual_seed(rows * 7 + p * 3 + kind + 100 * seed)
    s = torch.zeros((rows, 4, H), dtype=torch.bfloat16, device=dev)
    x = torch.randn((rows, H), generator=g, device=dev)
    if kind == 1:  # one tiny value in lanes 0..5 of every virtual wave: exceptions, coded
        x.view(rows, 12, 64, 8)[:, :, :6, 3] = 2.0**-30
    elif kind == 2:  # two tiny values in lane 7: a raw virtual wave; every third row also all of wave 5
        x.view(rows, 12, 64, 8)[:, :, 7, 1:3] = 2.0**-40
        x.view(rows, 12, 64, 8)[::3, 5, :, 0] *= 2.0**-60
    elif kind == 3:  # seven lanes with one exception each in wave 9: raw
        x.view(rows, 12, 64, 8)[:, 9, 10:17, 6] = 2.0**-25
    elif kind == 4:  # specials, zeros of both signs, denormals, exponent extremes
        v = x.view(rows, 12, 64, 8)
        v[:, 0, 0, 0] = float("nan")
        v[:, 1, 3, 2] = float("inf")
        v[:, 2, 5, 7] = -float("inf")
        v[:, 3, :, :] = 0.0
        v[:, 4, 1::2, :] = -0.0
        v[:, 6, 2, 4] = 2.0**-133
        v[:, 7, 4, 0] = 3.0e38
        v[:, 8, :, :] *= 2.0**100
        v[:, 10, ::5, 1] = 2.0**-14
    s[:, 0] = x.bfloat16()
    return s, torch.zeros((rows, H), dtype=torch.bfloat16, device=dev)


def _codec_worker(rank, errors, rows_list):
    """The coded tail's three outputs equal the BF16-link tail's byte for byte (and the residual the reference) on
    rows that take every branch of the code, eager and on graph replay."""
    from amd_kernels.collectives.pull_collectives_ranks2 import SlotsReduceScatterComm
    from amd_kernels.collectives.pull_collectives_ranks2.reference import slots4_reduce_scatter_reference
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import add_rmsnorm_reference, unshuffle_scales
    dev = torch.device('cuda', rank)
    coded = SlotsReduceScatterComm(rank, dist.group.WORLD, dev, coded=True)
    plain = SlotsReduceScatterComm(rank, dist.group.WORLD, dev, coded=False)
    coded.warm_up()
    plain.warm_up()
    for rows in rows_list:
        shard = rows // 2
        r = torch.randn((shard, H), device=dev).bfloat16()
        w = (torch.randn(H, device=dev) * 0.2).bfloat16()
        def valid(got):
            return got[0][:shard].view(torch.uint8), unshuffle_scales(got[1], shard), got[2]
        for kind in range(5):
            xs = [_codec_rows(rows, p, dev, kind) for p in range(2)]
            want = slots4_reduce_scatter_reference([x[0] for x in xs], [x[1] for x in xs], rank)
            a = coded.slots4_reduce_scatter_add_rmsnorm_mxfp8(*xs[rank], r, w, 1e-6)
            b = plain.slots4_reduce_scatter_add_rmsnorm_mxfp8(*xs[rank], r, w, 1e-6)
            torch.cuda.synchronize()
            _expect(errors, f'{rank} rows{rows} kind{kind} residual', a[2], add_rmsnorm_reference(want, r, w, 1e-6)[1])
            for k, (x, y) in enumerate(zip(valid(a), valid(b))):
                _expect(errors, f'{rank} rows{rows} kind{kind} coded vs bf16 field {k}', x, y)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                got = coded.slots4_reduce_scatter_add_rmsnorm_mxfp8(*xs[rank], r, w, 1e-6)
            for rep in range(2):
                got[0].view(torch.uint8).fill_(255); got[1].fill_(255); got[2].fill_(float('nan'))
                g.replay()
                torch.cuda.synchronize()
                for k, (x, y) in enumerate(zip(valid(got), valid(b))):
                    _expect(errors, f'{rank} rows{rows} kind{kind} replay{rep} field {k}', x, y)
            del g, got, a, b, xs, want
        torch.cuda.empty_cache()


def test_slots4_coded_tail_codec_paths():
    run_on_2_gpus(_codec_worker, [2, 66, 4100, 16384])
