# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Joined SP MoE on rank-specific weights: explicit vs lossless compact.

All preparation, staging, gather, reconstruction, experts/shared, RS and norm
are inside timing. Both arms share every common tensor allocation and communicator.
The explicit control keeps the fixed-chunk device pull. Gather progress is
device-owned, in graphs and eagerly.
"""
import argparse
import collections
import contextlib
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_4_gpus

H = 6144


def load_weights(rank, args):
    """Synthetic by default; a caller can supply its own checkpoint loader."""
    if args.weights_factory:
        import importlib.util
        path, name = args.weights_factory.rsplit(':',1)
        spec = importlib.util.spec_from_file_location('joined_weight_provider',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module,name)(rank,args.layer)
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, SharedWeights
    from tests.moe.mxfp4_h6144_i768_e128_top4.workload import make_experts
    raw, _ = make_experts(seed=rank)
    gate_gen = torch.Generator(device='cuda').manual_seed(607)
    gen = torch.Generator(device='cuda').manual_seed(608+rank)
    def fp8(n,k):
        q = (torch.randn((n,k),device='cuda',generator=gen)*20).to(torch.float8_e4m3fn)
        scale = torch.randint(118,124,(n,k//32),device='cuda',dtype=torch.uint8,generator=gen)
        return q.view(torch.uint8),scale
    shared = SharedWeights.from_row_major(*fp8(1536,6144),*fp8(6144,768))
    gate = torch.randn((128,6144),device='cuda',generator=gate_gen)/6144**.5
    bias = torch.randn((128,),device='cuda',generator=gate_gen)*.01
    return MoeWeights.from_row_major(gate,bias,*raw,shared=shared)


class Arena:
    def __init__(self):
        self.buffers = {}
        self.orig = torch.empty
        self.origlike = torch.empty_like
        self.phase = ''
        self.index = 0
        self.counts = collections.defaultdict(int)
        self.q4 = None
        self.gather_buffers = None
        self.frozen = False

    def allocate(self, shape, kw, source):
        shape = tuple(shape)
        dtype = kw.get('dtype', torch.float32)
        if self.phase == 'gather' and source[0] == 'prepared_input.py':
            key = (shape,dtype)
            occ = self.counts[key]
            self.counts[key] += 1
            return self.gather_buffers[key][occ]
        if source[0] == 'sidecar_unpack.py':
            assert self.q4 is not None and self.q4.shape == shape and self.q4.dtype == dtype
            return self.q4
        sem = (self.index, self.phase, source, shape, dtype)
        occ = self.counts[sem]
        self.counts[sem] += 1
        key = (*sem, occ)
        if key not in self.buffers:
            assert not self.frozen, key
            self.buffers[key] = self.orig(shape, **kw)
        return self.buffers[key]

    def empty(self, *shape, **kw):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) else shape
        fr = sys._getframe(1)
        return self.allocate(shape, kw, (os.path.basename(fr.f_code.co_filename), fr.f_code.co_name))

    def like(self, x, **kw):
        fr = sys._getframe(1)
        return self.allocate(x.shape, dict(dtype=kw.get('dtype', x.dtype), device=x.device),
                             (os.path.basename(fr.f_code.co_filename), fr.f_code.co_name))

    @contextlib.contextmanager
    def active(self, phase, index):
        self.phase, self.index = phase, index
        self.counts.clear()
        torch.empty, torch.empty_like = self.empty, self.like
        try:
            yield
        finally:
            torch.empty, torch.empty_like = self.orig, self.origlike


def _worker(rank, errors, args):
    from amd_kernels.collectives.pull_collectives_ranks4 import AllGatherComm, SlotsReduceScatterComm
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeInput, MoeInputCompact, prepare_mxfp4_moe, gather_mxfp4_moe, mxfp4_moe
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import unshuffle_scales
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) >= 16:
        chunk = len(cpus)//4
        os.sched_setaffinity(0, cpus[rank*chunk:(rank+1)*chunk])
    dev = torch.device('cuda', rank)
    w = load_weights(rank, args)
    dist.barrier()
    gather = AllGatherComm(rank, dist.group.WORLD, dev, 8192*9632)
    tail = SlotsReduceScatterComm(rank, dist.group.WORLD, dev)
    gather.warm_up()
    from amd_kernels.collectives.pull_collectives_ranks4.device_sequence import compile_device_pull
    original_pull = compile_device_pull()
    served_pull = gather.pull
    # The explicit reference runs the separate stage + publish launches and the device pull (compile_device_pull).
    import amd_kernels.collectives.pull_collectives_ranks4.gather_host as gh
    real_run = gh._run_compiled
    active_arm = 'selected'
    def reference_transport(exe,*packed):
        if exe is gather.stage_six and active_arm == args.legacy_stage_arm:
            sources,offsets,sizes=packed[:6],packed[6:12],packed[12:18]
            pool,stride,counter,flag,stream=packed[23:]
            for src,off,n in zip(sources,offsets,sizes):
                real_run(gather.stage,src,pool+off,counter,stride,n,stream)
            real_run(gather.publish,flag,counter,stream)
        else:
            real_run(exe,*packed)
    gh._run_compiled = reference_transport
    tail.warm_up()
    norm_generator = torch.Generator(device=dev).manual_seed(99039)
    norm_w = (torch.randn(H, device=dev,generator=norm_generator)*.2).bfloat16()
    # The replicated next norm weights agree across ranks.
    cpu_w = norm_w.cpu()
    dist.broadcast(cpu_w, 0)
    norm_w.copy_(cpu_w)
    import importlib.metadata
    result = {'torch':torch.__version__,'flydsl':importlib.metadata.version('flydsl'),'host':os.uname().nodename,
              'gpu':torch.cuda.get_device_properties(dev).name,'rank': rank, 'pci': torch.cuda.get_device_properties(dev).pci_bus_id, 'args': vars(args), 'results': {}}
    for rows in args.tokens:
        phase_before = int(gather.counter.item())
        local = rows//4
        assert rows % 4 == 0
        count = args.inputs or (100 if rows<=2048 else 16)
        xs, residuals = [], []
        for i in range(count):
            rng = np.random.default_rng(1000*rows+i)
            full = rng.standard_normal((rows,H),dtype=np.float32)
            x = torch.from_numpy(full[rank*local:(rank+1)*local].copy()).bfloat16().cuda()
            if args.padding:
                x[-1].zero_()
            xs.append(x)
            full_residual = rng.standard_normal((rows,H),dtype=np.float32)
            residuals.append(torch.from_numpy(full_residual[rank*local:(rank+1)*local].copy()).bfloat16().cuda())
        proto = prepare_mxfp4_moe(xs[0], w)
        canonical = [torch.empty((rows,*t.shape[1:]),dtype=t.dtype,device=dev) for t in proto.tensors()]
        mask = torch.empty((rows,768), dtype=torch.uint8,device=dev)
        explicit = MoeInput(*canonical)
        compact = MoeInputCompact(canonical[1],canonical[2],canonical[3],canonical[4],canonical[5],mask)
        # Graph-safe gather and original host-counter gather have identical payload staging semantics.
        arena = Arena()
        arena.q4 = canonical[0]
        arena.gather_buffers = collections.defaultdict(list)
        for t in (*canonical,mask):
            arena.gather_buffers[(tuple(t.shape),t.dtype)].append(t)
        last_parts = {}
        def call(arm, i):
            nonlocal active_arm
            active_arm = arm
            gather.pull = original_pull if arm == 'explicit' else served_pull
            with arena.active('prepare',i):
                prepared = prepare_mxfp4_moe(xs[i],w)
            if arm == 'selected':
                with arena.active('gather',i):
                    incoming = gather_mxfp4_moe(prepared,gather)
            elif arm != 'explicit':
                with arena.active('pack',i):
                    prepared = prepared.compact()
                incoming = compact
            else:
                incoming = explicit
            if arm != 'selected':
                gather.all_gather(incoming.tensors(), prepared.tensors())
            with arena.active('experts',i):
                parts, shared = mxfp4_moe(incoming,w,return_partials=True)
            last_parts[i] = (parts,shared)
            with arena.active('tail',i):
                return tail.slots4_reduce_scatter_add_rmsnorm_mxfp8(parts,shared,residuals[i],norm_w,1e-6)
        # Warm public entries before arena recording (FlyDSL compile can allocate).
        _ = proto.compact().expand()
        _ = mxfp4_moe(proto,w,return_partials=True)
        # Global consumer warm-up with valid gathered bytes.
        gather.all_gather(explicit.tensors(),proto.tensors())
        _ = mxfp4_moe(explicit,w,return_partials=True)
        torch.cuda.synchronize()
        def valid(y):
            return y[0].view(torch.uint8),unshuffle_scales(y[1],local),y[2]
        # Compare complete routed/shared outputs and all final norm fields, not just logits/cosines.
        for i in range(count):
            y = call('explicit',i)
            saved = [t.clone() for t in (*last_parts[i],*valid(y))]
            operand = [t.clone() for t in canonical]
            z = call('selected' if 'selected' in args.arms else 'compact',i)
            torch.cuda.synchronize()
            for k,(a,b) in enumerate(zip(saved,(*last_parts[i],*valid(z)))):
                assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)), (rank,rows,i,k,'entry bits')
            for k,(a,b) in enumerate(zip(operand,canonical)):
                assert torch.equal(a,b),(rank,rows,i,k,'operand bits')
        # Record every requested arm, including diagnostic third arms, before capture.
        for warm_arm in args.arms:
            for i in range(count):
                call(warm_arm,i)
            torch.cuda.synchronize()
        arena.frozen = True
        graphs, outputs = {}, {}
        order = [v+str(i) for i in range(2) for v in args.arms]
        if args.reverse:
            order.reverse()
        for name in order:
            arm = name[:-1]
            g = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(g):
                outputs[name] = [call(arm,i) for i in range(count)]
            graphs[name] = g
        expected = [t.clone() for y in outputs[order[-1]] for t in valid(y)]
        def check(name):
            for y in outputs[name]:
                y[0].view(torch.uint8).fill_(255)
                y[1].fill_(255)
                y[2].fill_(float('nan'))
            graphs[name].replay()
            torch.cuda.synchronize()
            assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(expected,[t for y in outputs[name] for t in valid(y)])),(rank,rows,name,'replay')
        for name in order:
            check(name)
        for _ in range(10):
            for name in order:
                graphs[name].replay()
                torch.cuda.synchronize()
        samples = {name:[] for name in order}
        def measure(fn,n):
            torch.cuda.synchronize()
            dist.barrier()
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(n):
                fn()
            b.record()
            b.synchronize()
            return a.elapsed_time(b)*1000/(n*count)
        for rd in range(args.rounds):
            for name in order if rd%2==0 else order[::-1]:
                samples[name].append(measure(graphs[name].replay,args.replays))
        for name in order:
            check(name)
        if not args.no_eager:
            for arm in args.arms:
                def batch():
                    for i in range(count):
                        call(arm,i)
                batch()
                torch.cuda.synchronize()
            for rd in range(args.rounds):
                for arm in (args.arms if rd%2==0 else args.arms[::-1]):
                    def batch():
                        for i in range(count):
                            call(arm,i)
                    samples.setdefault('eager_'+arm,[]).append(measure(batch,1))
        phase_after = int(gather.counter.item())
        expected_phases = (1+(2+len(args.arms))*count
                           +len(graphs)*count*(12+args.rounds*args.replays))
        if not args.no_eager:
            expected_phases += len(args.arms)*count*(1+args.rounds)
        assert phase_after-phase_before == expected_phases,(rank,rows,phase_before,phase_after,expected_phases)
        every = [None]*4
        dist.all_gather_object(every,samples)
        maxima = {k:[max(e[k][i] for e in every) for i in range(len(samples[k]))] for k in samples}
        rr = {'samples':maxima,'medians':{k:statistics.median(v) for k,v in maxima.items()},
              'ranges':{k:[min(v),max(v)] for k,v in maxima.items()},'every_rank':every,
              'entry_and_operands_bit_equal':True,'arena_allocations':len(arena.buffers),
              'gather_phase_delta':phase_after-phase_before,'expected_gather_phases':expected_phases}
        result['results'][rows] = rr
        if rank == 0:
            print(rows,json.dumps({k:v for k,v in rr.items() if k!='every_rank'}),flush=True)
            with open(args.out,'w') as f:
                json.dump(result,f,indent=2)
        del graphs,outputs,expected,arena,last_parts,saved,operand,xs,residuals,proto,canonical,compact,explicit,mask
        torch.cuda.empty_cache()


def worker(rank, errors, args):
    try:
        _worker(rank, errors, args)
    except Exception:
        import traceback
        print(f"rank {rank} failed: {traceback.format_exc()}", flush=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-stage-arm',default='explicit')
    p.add_argument('--weights-factory',help='optional /absolute/provider.py:function(rank,layer)')
    p.add_argument('--arms',nargs='+',choices=['explicit','compact','selected'],default=['explicit','selected'])
    p.add_argument('--tokens',type=int,nargs='+',default=[4096,8192,16384,32768])
    p.add_argument('--inputs',type=int)
    p.add_argument('--rounds',type=int,default=7)
    p.add_argument('--replays',type=int,default=10)
    p.add_argument('--layer',type=int,default=39)
    p.add_argument('--reverse',action='store_true')
    p.add_argument('--padding',action='store_true')
    p.add_argument('--no-eager',action='store_true')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    run_on_4_gpus(worker,a,timeout=2700)

if __name__=='__main__':
    main()
