# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Registered one-shot reads must not race with overlapping norm outputs: the entry falls back before any direct-reader
kernel is fetched. Reuse, skew, poisoned replays, FP64, FTZ and warm-up at every acked row count:
``test_staged_ar_add_rmsnorm.py``."""
import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_4_gpus
from .test_staged_ar_add_rmsnorm import H, EPS, _inputs, _written, _bits


def _overlap_worker(rank, errors):
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm import staged_host as sh
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import allreduce_rows_reference

    dev = torch.device('cuda', rank)
    w = torch.zeros(H, dtype=torch.bfloat16, device=dev)
    comm = arn.staged_comm(rank, dist.group.WORLD, dev)
    comm.warm_up(w, EPS)

    def forbidden():
        raise AssertionError('overlapping output reached a direct reader')

    names = ('_oneshot_ack_kernel', '_oneshot8_ack_kernel', '_oneshot8_ack_cached_local_kernel')
    real = {n: getattr(sh, n) for n in names}
    for n in names:  # restored below: later calls of this file run in the same rank processes
        setattr(sh, n, forbidden)
    try:
        _overlap_rows(rank, errors, comm, w, dev, an, allreduce_rows_reference)
    finally:
        for n, f in real.items():
            setattr(sh, n, f)


def _overlap_rows(rank, errors, comm, w, dev, an, allreduce_rows_reference):
    for rows in (1, 8, 11, 13):
        inputs = _inputs(rows, 9, dev)
        r = torch.randn(rows, H, device=dev).bfloat16()
        reduced = allreduce_rows_reference(inputs)
        for kind in ('out', 'res', 'codes'):
            storage = torch.empty((32, H), dtype=torch.bfloat16, device=dev)
            x = storage[:rows]
            x.copy_(inputs[rank])
            want = (an.gemma_add_rmsnorm_mxfp8(reduced, r, w, EPS, 16) if kind == 'codes'
                    else an.gemma_add_rmsnorm(reduced, r, w, EPS))
            graph = torch.cuda.CUDAGraph()
            with comm.capture(), torch.cuda.graph(graph):
                x.copy_(inputs[rank])
                if kind == 'codes':
                    q = storage.view(torch.uint8).flatten()[:16*H].view(torch.float8_e4m3fn).view(16, H)
                    got = comm.ar_add_rmsnorm_mxfp8(x, r, w, EPS, 16, q_out=q)
                else:
                    kw = {'out' if kind == 'out' else 'res_out': x}
                    got = comm.ar_add_rmsnorm(x, r, w, EPS, **kw)
            graph.replay()
            torch.cuda.synchronize()
            tile = 16 if kind == 'codes' else None
            for u, v in zip(_written(got, rows, tile), _written(want, rows, tile)):
                if not torch.equal(_bits(u), _bits(v)):
                    errors.append(f'rank {rank}, rows {rows}, overlapping {kind}')


def test_registered_oneshot_ack_overlapping_outputs():
    run_on_4_gpus(_overlap_worker)


def _generation_worker(rank, errors):
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.host import _DevicePointer
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.ar_add_rmsnorm_oneshot_flags128_owner_order import WFLAG_OFF, WFLAG_BYTES
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import allreduce_rows_reference

    def signed(value):
        return (value + 2**31) % 2**32 - 2**31

    dev = torch.device('cuda', rank)
    w = torch.zeros(H, dtype=torch.bfloat16, device=dev)
    comm = arn.staged_comm(rank, dist.group.WORLD, dev)
    comm.warm_up(w, EPS)
    storage = torch.as_tensor(_DevicePointer(comm.bufs[rank], WFLAG_OFF+WFLAG_BYTES), device=dev)
    flags = storage[WFLAG_OFF:WFLAG_OFF+WFLAG_BYTES].view(torch.int32).view(-1, 32)
    for rows, mx in ((8, False), (1, True), (4, True), (5, True), (8, True)):
        counts = comm.ack_counts8 if mx else comm.ack_counts
        staged_counts = comm.counts8 if mx else comm.counts
        column = 8 if mx else 4
        r = torch.randn(rows, H, device=dev).bfloat16()
        inputs = [_inputs(rows, c+11, dev) for c in range(3)]
        want = []
        for parts in inputs:
            reduced = allreduce_rows_reference(parts)
            want.append(an.gemma_add_rmsnorm_mxfp8(reduced, r, w, EPS, 16) if mx
                        else an.gemma_add_rmsnorm(reduced, r, w, EPS))
        for start in (2**30-2, 2**31-2, -2):
            # a legal last-completed direct-call state, on both signed boundaries
            counts.fill_(start)
            flags[:, column:column+4].fill_(signed(2*start+1))
            # a long eager interval may advance the independent staged domain
            staged_counts.fill_(2**30+9)
            x = torch.empty((rows, H), dtype=torch.bfloat16, device=dev)
            outputs = []
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with comm.capture(), torch.cuda.graph(graph):
                if rank == 3:
                    torch.cuda._sleep(100000)
                for parts in inputs:
                    x.copy_(parts[rank])
                    outputs.append(comm.ar_add_rmsnorm_mxfp8(x, r, w, EPS, 16) if mx
                                   else comm.ar_add_rmsnorm(x, r, w, EPS))
                    x.fill_(float('nan'))
            for rep in range(2):
                graph.replay()
                torch.cuda.synchronize()
                if not bool((counts[:rows] == signed(start+3*(rep+1))).all()):
                    errors.append(f'rank {rank} {rows}/{mx} counter wrap at {start}, replay {rep}')
                if not bool((staged_counts == 2**30+9).all()):
                    errors.append(f'rank {rank}: direct reader advanced staged generation')
                for got, ref in zip(outputs, want):
                    for u, v in zip(_written(got, rows, 16 if mx else None), _written(ref, rows, 16 if mx else None)):
                        if not torch.equal(_bits(u), _bits(v)):
                            errors.append(f'rank {rank} {rows}/{mx} bits at generation {start}')


def test_registered_oneshot_ack_generation_wrap_and_independent_staged_counts():
    run_on_4_gpus(_generation_worker)
