# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Lossless all-gather: exact codec, old-chain parity, warm-up, skew and stream ownership."""
import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_4_gpus
from tests.norm.gemma_add_rmsnorm_h6144.test_slots4_ar_add_rmsnorm import (
    _bits, _inputs, _residual, _worker, _written,
)
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import H


def test_slots4_lossless_reference_replay_and_compile():
    run_on_4_gpus(_worker, (159, 160, 161, 176, 192, 193, 224, 241, 255, 256, 257), 16, False, True)


def _reuse_worker(rank, errors):
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.host import SlotsArNormComm
    from flydsl.compiler.jit_function import MlirCompiler

    dev = torch.device('cuda', rank)
    w = torch.randn(H, generator=torch.Generator().manual_seed(81)).bfloat16().to(dev)
    comm = SlotsArNormComm(rank, dist.group.WORLD, dev)
    comm.warm_up(w, 1e-6, 16)
    real_compile = MlirCompiler.compile
    def late(*args, **kwargs):
        raise AssertionError('compiled after SlotsArNormComm.warm_up')
    MlirCompiler.compile = late
    try:
        # Changing q, raw overflow slots and the dispatch tier may not leave live reads.
        for m in (160, 255, 159, 193, 256, 161, 257, 160):
            padded = torch.empty((m, H + 8), dtype=torch.bfloat16, device=dev)
            r = padded[:, :H]
            r.copy_(_residual(m, dev))
            every = [_inputs(m, i, dev, True)[rank] for i in range(4)]
            # Force raw waves and isolated exceptions in local and reduced rows.
            for sl, sh in every:
                sl[:, :, :32] = 0
                sh[:, :32] = 0
                sl[:, 0, 0:16] = 2.0**-40
                sl[:, 0, 16:32] = 2.0**20
                sl[:, 0, 1024] = 2.0**-40
            sl = torch.empty_like(every[0][0])
            sh = torch.empty_like(every[0][1])
            refs = []
            comm.lossless_min_rows = 10**9
            for source, shared in every:
                sl.copy_(source); sh.copy_(shared)
                refs.append(comm.slots4_ar_add_rmsnorm_mxfp8(sl, sh, r, w, 1e-6, 16))
                sl.fill_(7); sh.fill_(-3)  # immediate producer scratch reuse
            comm.lossless_min_rows = type(comm).lossless_min_rows
            outs = []
            for source, shared in every:
                sl.copy_(source); sh.copy_(shared)
                if rank == 3:
                    torch.cuda._sleep(300000)
                outs.append(comm.slots4_ar_add_rmsnorm_mxfp8(sl, sh, r, w, 1e-6, 16))
                sl.fill_(-5); sh.fill_(9)
            torch.cuda.synchronize()
            for i, (out, ref) in enumerate(zip(outs, refs)):
                for a, b in zip(_written(out, m), _written(ref, m)):
                    assert torch.equal(_bits(a), _bits(b)), (rank, m, i)
        # The device generation can cross the signed boundary without a reset.
        m = 160
        sl, sh = _inputs(m, 8, dev, True)[rank]
        r = _residual(m, dev)
        comm.lossless_min_rows = 10**9
        ref = comm.slots4_ar_add_rmsnorm_mxfp8(sl, sh, r, w, 1e-6, 16)
        comm.lossless_min_rows = type(comm).lossless_min_rows
        comm.counts[:m].fill_(2**31 - 2)
        wrap_out = [comm.slots4_ar_add_rmsnorm_mxfp8(sl, sh, r, w, 1e-6, 16) for _ in range(4)]
        torch.cuda.synchronize()
        assert set(comm.counts[:m].tolist()) == {-2**31 + 2}
        for out in wrap_out:
            for a, b in zip(_written(out, m), _written(ref, m)):
                assert torch.equal(_bits(a), _bits(b)), (rank, 'generation wrap')
        # fresh objects do not inherit a warmed object's call counts; two simultaneous graphs use separate communicator
        # state and streams
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        fresh = [SlotsArNormComm(rank, dist.group.WORLD, dev) for _ in range(2)]
        m = 192
        inp = [_inputs(m, i, dev, True)[rank] for i in range(2)]
        residual = [_residual(m, dev, i) for i in range(2)]
        eager = []
        for i in range(2):
            eager.append(fresh[i].slots4_ar_add_rmsnorm_mxfp8(*inp[i], residual[i], w, 1e-6, 16))
            torch.cuda.synchronize()
            assert set(fresh[i].counts[:m].tolist()) == {1}
        graphs, outputs = [], []
        for i in range(2):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=streams[i]):
                outputs.append(fresh[i].slots4_ar_add_rmsnorm_mxfp8(*inp[i], residual[i], w, 1e-6, 16))
            graphs.append(g)
        for rep in range(3):
            for i in range(2):
                with torch.cuda.stream(streams[i]):
                    if rank == 3:
                        torch.cuda._sleep(300000)
                    graphs[i].replay()
            torch.cuda.synchronize()
            for i in range(2):
                for a, b in zip(_written(outputs[i], m), _written(eager[i], m)):
                    assert torch.equal(_bits(a), _bits(b)), (rank, i, rep)
                assert set(fresh[i].counts[:m].tolist()) == {rep + 2}
    finally:
        MlirCompiler.compile = real_compile


def test_slots4_lossless_reuse_delayed_rank_and_independent_streams():
    run_on_4_gpus(_reuse_worker, timeout=240)


def test_slots4_lossless_next_to_mfma():
    run_on_4_gpus(_worker, (160, 193, 256), 16, True, True)


def test_slots4_stream_lossless_reference_replay_and_compile():
    # every edge of the streaming tiers (R8 / R16 / R32) and of the two-phase coded ranges, both code layouts,
    # sensitive rows, shrinking and growing shapes
    run_on_4_gpus(_worker, (768, 769, 1024, 1025, 1280, 1281, 1536, 1537, 1601, 1752, 2048, 2049, 1537,
                            2943, 2944, 3072, 3073, 4094, 4095, 769),
                  16, False, True)
    run_on_4_gpus(_worker, (769, 1280, 1536, 1537, 1752, 2048, 2049, 2943, 2944, 3000, 3073, 4095,
                            1601), 0, False, True)


def test_slots4_stream_lossless_next_to_mfma():
    run_on_4_gpus(_worker, (1024, 1537, 2048, 4095), 16, True, True)
    run_on_4_gpus(_worker, (1752, 3584, 4095), 0, True, True)


def _stream_reuse_worker(rank, errors, tile):
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.host import SlotsArNormComm
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import slots4_allreduce_reference
    from flydsl.compiler.jit_function import MlirCompiler

    dev = torch.device('cuda', rank)
    w = torch.randn(H, generator=torch.Generator().manual_seed(81)).bfloat16().to(dev)
    comm = SlotsArNormComm(rank, dist.group.WORLD, dev)
    comm.warm_up(w, 1e-6, 16)
    x = torch.zeros((1, H), dtype=torch.bfloat16, device=dev)
    an.gemma_add_rmsnorm_mxfp8(x, x, w, 1e-6, 16)  # compile the reference chain's norm before the guard

    def run(c, *args):
        return c.slots4_ar_add_rmsnorm_mxfp8(*args, w, 1e-6, tile)

    def reference(ranks, r):
        """The contract chain on every rank's (slots, shared): the reference all-reduce, then the one-GPU norm."""
        red = slots4_allreduce_reference([sl for sl, _ in ranks], [sh for _, sh in ranks])
        return an.gemma_add_rmsnorm_mxfp8(red, r, w, 1e-6, 16)

    real_compile = MlirCompiler.compile
    def late(*args, **kwargs):
        raise AssertionError('compiled after SlotsArNormComm.warm_up')
    MlirCompiler.compile = late
    try:
        # changing q, raw waves and tier switches (in and out of the coded tier) may not leave live reads of the coded
        # rows or of the raw arena
        for m in (1752, 4095, 1024, 2048, 3000, 2943, 3072, 3073, 1536, 1280, 1752):
            padded = torch.empty((m, H + 8), dtype=torch.bfloat16, device=dev)
            r = padded[:, :H]
            r.copy_(_residual(m, dev))
            every_rank = [_inputs(m, i, dev, True) for i in range(3)]
            for sl, sh in (pair for ranks in every_rank for pair in ranks):
                sl[:, :, :32] = 0
                sh[:, :32] = 0
                sl[::3, 0, 0:16] = 2.0**-40  # a raw wave (16 exceptions in one group)
                sl[::3, 0, 16:32] = 2.0**20
                sl[:, :, 2000:2032] = 0
                sh[:, 2000:2032] = 0
                sl[1::5, 0, 2000] = 2.0**10  # one exception in the reduced row
                sl[1::5, 0, 2001] = 2.0**-10
                sl[:, :, 4096:4128] = 0
                sh[:, 4096:4128] = 0
                sl[2::7, 0, 4096:4112] = 2.0**-40  # a raw wave in the second column piece
                sl[2::7, 0, 4112:4128] = 2.0**20
            every = [ranks[rank] for ranks in every_rank]
            sl = torch.empty_like(every[0][0])
            sh = torch.empty_like(every[0][1])
            refs = [reference(ranks, r) for ranks in every_rank]
            outs = []
            for source, shared in every:
                sl.copy_(source); sh.copy_(shared)
                if rank == 3:
                    torch.cuda._sleep(300000)
                outs.append(run(comm, sl, sh, r))
                sl.fill_(-5); sh.fill_(9)
            torch.cuda.synchronize()
            for i, (out, ref) in enumerate(zip(outs, refs)):
                for a, b in zip(_written(out, m, tile), _written(ref, m)):
                    assert torch.equal(_bits(a), _bits(b)), (rank, m, i)
            del every, every_rank, refs, outs
        # The generation crosses the signed boundary: only sub-batch leaders count.
        m = 1752
        q = (m + 3) // 4
        ranks = _inputs(m, 8, dev, True)
        sl, sh = ranks[rank]
        r = _residual(m, dev)
        ref = reference(ranks, r)
        torch.cuda.synchronize()
        comm.counts[:m].fill_(2**31 - 2)
        wrap_out = [run(comm, sl, sh, r) for _ in range(4)]
        torch.cuda.synchronize()
        want = [(-2**31 + 2) if (row % q) % 2 == 0 else 2**31 - 2 for row in range(m)]
        assert comm.counts[:m].tolist() == want, (rank, 'generation wrap')
        for out in wrap_out:
            for a, b in zip(_written(out, m, tile), _written(ref, m)):
                assert torch.equal(_bits(a), _bits(b)), (rank, 'generation wrap')
        # Two simultaneous graphs use separate communicator state, raw arenas and streams.
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        fresh = [SlotsArNormComm(rank, dist.group.WORLD, dev) for _ in range(2)]
        assert fresh[0].raw_bufs[rank] != fresh[1].raw_bufs[rank]
        assert fresh[0].input_raw_bufs[rank] != fresh[1].input_raw_bufs[rank]
        inp = [_inputs(m, i, dev, True)[rank] for i in range(2)]
        residual = [_residual(m, dev, i) for i in range(2)]
        eager = []
        for i in range(2):
            eager.append(run(fresh[i], *inp[i], residual[i]))
        torch.cuda.synchronize()
        graphs, outputs = [], []
        for i in range(2):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=streams[i]):
                outputs.append(run(fresh[i], *inp[i], residual[i]))
            graphs.append(g)
        for rep in range(3):
            for i in range(2):
                with torch.cuda.stream(streams[i]):
                    if rank == 3:
                        torch.cuda._sleep(300000)
                    graphs[i].replay()
            torch.cuda.synchronize()
            for i in range(2):
                for a, b in zip(_written(outputs[i], m, tile), _written(eager[i], m, tile)):
                    assert torch.equal(_bits(a), _bits(b)), (rank, i, rep)
                lead = [fresh[i].counts[row].item() for row in range(0, q, 2)]
                assert set(lead) == {rep + 2}, (rank, i, rep, set(lead))
    finally:
        MlirCompiler.compile = real_compile


def test_slots4_stream_lossless_reuse_delayed_rank_and_independent_streams():
    run_on_4_gpus(_stream_reuse_worker, 16, timeout=600)
    run_on_4_gpus(_stream_reuse_worker, 0, timeout=600)
