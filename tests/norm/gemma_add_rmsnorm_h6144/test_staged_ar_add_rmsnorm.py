# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The attention tail's staged-partial 4-rank all-reduce + add + Gemma RMSNorm, bf16 out
(``StagedArNormComm.ar_add_rmsnorm``) and MXFP8 out (``ar_add_rmsnorm_mxfp8``, tiled and row-major codes), on 4 gfx950
GPUs:

- against the contract: ``reference.allreduce_rows_reference`` then the one-GPU ``gemma_add_rmsnorm``
  (``gemma_add_rmsnorm_mxfp8``, tile 16) on it, bit for bit (out or written codes and scales, and residual out;
  row-major codes = the tiled norm's codes untiled), and the fp64 norm / MXFP8 contract of reference.py on the first
  and last call; denormals and signed zeros in the partials, cancellations into denormals;
- row counts that are not multiples of 4 (q = ceil(rows / 4)), 1 row;
- eager calls, each with its own inputs, the residual flowing from call to call; then the same calls captured in a
  graph and replayed into NaN-filled guarded outputs: bit for bit the eager results, with changed-input replays and
  eager calls interleaved; next to MFMA waves on a high-priority stream;
- rule 4: after ``warm_up``, no row count compiles anything.
"""

import contextlib

import torch
import torch.distributed as dist
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
    CODE_MISMATCH,
    MXFP8_COS,
    OUT_COS,
    OUT_MISMATCH,
    SCALE_MISMATCH,
    WORLD,
    H,
    add_rmsnorm_reference,
    allreduce_rows_reference,
    compare_mxfp8,
    compare_out,
    unshuffle_scales,
    untile_codes,
)

from tests.multi_gpu import run_on_4_gpus

EPS = 1e-6
CALLS = 4


def _inputs(rows, call, dev, sensitive=False):
    """Every rank's bf16 [rows, H] partial of call ``call`` (each rank makes all four: the reference needs them)."""
    out = []
    for p in range(WORLD):
        g = torch.Generator(device=dev).manual_seed(rows * 1000 + call * 10 + p)
        x = torch.randn(rows, H, generator=g, device=dev) * (0.5 + p)
        x[:, :32] *= 40
        x[:, 64:72] = 2.0**-133  # bf16 denormals: flushed by the rank sum
        x[:, 72:80] = -0.0
        if sensitive:
            # normal operands cancel into a denormal, which the rank sum flushes
            x[:, 128:160] = 0.0
            if p == 0:
                x[:, 128:160] = 2.0**-125
            elif p == 1:
                x[:, 128:160] = -(2.0**-125) + 2.0**-133
            # zero partials: a denormal residual must survive the restored mode
            x[:, 160:192] = 0.0
            # finite all-reduce operands whose first row's sum of squares overflows fp32: the norm returns signed zeros
            x[0, 192:224] = 1.0e20
            # the lossless all-gather code: 8 sums 2^-20 below their group (a raw wave), one such sum (an exception),
            # zeros
            x[:, 256:264] *= 2.0**-20
            x[:, 1000] *= 2.0**-20
            x[:, 2048:2080] = 0.0
        out.append(x.bfloat16())
    return out


def _residual(rows, dev, sensitive=False):
    g = torch.Generator(device=dev).manual_seed(rows + 7919)
    r = torch.randn(rows, H, generator=g, device=dev).bfloat16()
    if sensitive:
        r[:, 128:160] = 0.0
        r[:, 160:192] = 2.0**-133
    return r


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _written(o, rows, tile):
    """The bytes an output holds for its rows: (codes row-major, scales unshuffled, residual) for MXFP8, (out, residual)
    for bf16."""
    if tile is None:
        return list(o)
    q, s, res = o
    return [untile_codes(q, rows) if tile else q[:rows].view(torch.uint8), unshuffle_scales(s, rows), res]


def _worker(rank, errors, row_counts, spin=False, sensitive=False, tile=None, eager_only=False, registered=True, strided=False):
    """``tile`` None: bf16 out; else MXFP8 out with that code tile (0: row-major); ``eager_only``: no graph part (rows
    only served eagerly)."""
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn

    dev = torch.device("cuda", rank)
    if spin:
        from tests.mfma_spinner import mfma_spinner

        spinner = mfma_spinner(32)
        side = torch.cuda.Stream(priority=-1)
    w = (torch.randn(H, generator=torch.Generator().manual_seed(7)) * 0.5).bfloat16().to(dev)
    comm = arn.staged_comm(rank, dist.group.WORLD, dev)
    comm.warm_up(w, EPS)
    compiled = []
    real = MlirCompiler.compile

    def counting(fn, *args, **kwargs):
        compiled.append(type(fn).__name__)
        return real(fn, *args, **kwargs)

    mx = tile is not None

    def call(x, r, o=None):
        if mx:
            if o is None:
                return comm.ar_add_rmsnorm_mxfp8(x, r, w, EPS, tile)
            return comm.ar_add_rmsnorm_mxfp8(x, r, w, EPS, tile, q_out=o[0], scale_out=o[1], res_out=o[2])
        if o is None:
            return comm.ar_add_rmsnorm(x, r, w, EPS)
        return comm.ar_add_rmsnorm(x, r, w, EPS, out=o[0], res_out=o[1])

    for rows in row_counts:
        tag = f"rank {rank} rows {rows}" + (f" tile {tile}" if mx else "")
        every = [_inputs(rows, c, dev, sensitive) for c in range(CALLS)]
        if strided:
            for inputs in every:
                for p, x in enumerate(inputs):
                    backing = torch.full((rows, H + 16), float("nan"), dtype=x.dtype, device=dev)
                    backing[:, :H].copy_(x)
                    inputs[p] = backing[:, :H]
        mine = [every[c][rank] for c in range(CALLS)]
        r0 = _residual(rows, dev, sensitive)
        if strided:
            backing = torch.full((rows, H + 8), float("nan"), dtype=r0.dtype, device=dev)
            backing[:, :H].copy_(r0)
            r0 = backing[:, :H]
        MlirCompiler.compile = counting
        try:
            eager, r = [], r0
            for x in mine:
                o = call(x, r)
                eager.append(o)
                r = o[-1]
            torch.cuda.synchronize()
        finally:
            MlirCompiler.compile = real
        if compiled:
            errors.append(f"{tag}: compiled {compiled} after warm_up")
            compiled.clear()
        r = r0
        for c in range(CALLS):
            x = allreduce_rows_reference(every[c])
            if mx:
                ref = _written(an.gemma_add_rmsnorm_mxfp8(x, r, w, EPS, 16), rows, 16)
                names = ("codes", "scales", "res")
            else:
                ref = an.gemma_add_rmsnorm(x, r, w, EPS)
                names = ("out", "res")
            for name, a, b in zip(names, _written(eager[c], rows, tile), ref):
                if not torch.equal(_bits(a), _bits(b)):
                    k = int((_bits(a) != _bits(b)).sum())
                    errors.append(f"{tag} call {c} {name}: {k} bytes differ from the reference chain")
            y, res_ref = add_rmsnorm_reference(x, r, w, EPS)  # fp64 on this rank's GPU
            if not torch.equal(_bits(eager[c][-1]), _bits(res_ref)):
                errors.append(f"{tag} call {c}: residual out differs from the reference")
            if c in (0, CALLS - 1):
                if mx:
                    wq, ws, _ = _written(eager[c], rows, tile)
                    st = compare_mxfp8(wq, ws, y)
                    ok = st["scale_max_diff"] <= 1 and st["scale_mismatch"] <= SCALE_MISMATCH
                    ok = ok and st["code_mismatch"] <= CODE_MISMATCH and st["code_far"] == 0
                    ok = ok and st["cos"] >= MXFP8_COS
                else:
                    st = compare_out(eager[c][0], y)
                    ok = st["nonfinite_ok"] and st["bad_ulp"] == 0 and st["mismatch"] <= OUT_MISMATCH
                    ok = ok and st["cos"] >= OUT_COS
                if not ok:
                    errors.append(f"{tag} call {c}: contract {st}")
                if rank == 0 and c == 0:
                    print(f"{tag} call {c}: {st}", flush=True)
            r = eager[c][-1]
        if eager_only:
            if rank == 0:
                print(f"{tag}: eager checked", flush=True)
            continue
        # the same calls in a graph, replayed into NaN-filled guarded outputs
        guards = []

        def guarded(t):
            buf = torch.full((t.numel() * t.element_size() + 32,), 0x5A, dtype=torch.uint8, device=dev)
            guards.append(buf)
            return buf[16:-16].view(t.dtype).view(t.shape)

        outs = [[guarded(t) for t in o] for o in eager]
        graph = torch.cuda.CUDAGraph()
        MlirCompiler.compile = counting
        try:
            with (comm.capture() if registered else contextlib.nullcontext()), torch.cuda.graph(graph):
                r = r0
                for x, o in zip(mine, outs):
                    call(x, r, o)
                    r = o[-1]
        finally:
            MlirCompiler.compile = real
        if compiled:
            errors.append(f"{tag}: compiled {compiled} in graph after warm_up")
            compiled.clear()
        for rep in range(3):
            if rep == 1:
                # change every rank's inputs, then interleave eager calls
                for inputs in every:
                    for x in inputs:
                        x.neg_()
                eager, r = [], r0
                for x in mine:
                    o = call(x, r)
                    eager.append(o)
                    r = o[-1]
            for o in outs:
                for t in o:
                    t.view(torch.uint8).fill_(0xFF)
            dist.barrier()
            torch.cuda.synchronize()
            if spin:
                side.wait_stream(torch.cuda.current_stream())
                spinner(100000, side)
            graph.replay()
            torch.cuda.synchronize()
            for buf in guards:
                if not bool((buf[:16] == 0x5A).all() & (buf[-16:] == 0x5A).all()):
                    errors.append(f"{tag} replay {rep}: output guard overwritten")
            for c in range(CALLS):
                for i, (a, b) in enumerate(zip(_written(outs[c], rows, tile), _written(eager[c], rows, tile))):
                    if not torch.equal(_bits(a), _bits(b)):
                        errors.append(f"{tag} replay {rep} call {c} output {i}: != eager")
        if rank == 0:
            print(f"{tag}: eager + 3 replays checked", flush=True)
        del graph, outs, eager, every, mine


def test_staged_oneshot_rows():
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.staged_host import ONESHOT_MAX_ROWS

    run_on_4_gpus(_worker, tuple(range(1, ONESHOT_MAX_ROWS + 1)) + (8, 1))


def test_staged_oneshot_denormal_scopes_next_to_mfma():
    run_on_4_gpus(_worker, (1, 3, 4, 7, 8, 10, 11, 12, 13, 16), True, True)


def test_staged_stream_rows():
    run_on_4_gpus(_worker, (288, 289, 320, 321, 384, 385, 480, 481, 512, 641, 944, 945, 1000, 1024, 1281, 2048, 2049, 3071, 4094, 4095))


def test_staged_stream_denormal_next_to_mfma():
    run_on_4_gpus(_worker, (333, 1024, 2049, 3073, 4095), True, True)


def test_staged_mxfp8_oneshot_rows():
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.staged_host import ONESHOT_MAX_ROWS_MXFP8

    run_on_4_gpus(_worker, tuple(range(1, ONESHOT_MAX_ROWS_MXFP8 + 1)), False, False, 16)
    run_on_4_gpus(_worker, (1, 7, 8, 13, 14, 15, 16), True, True, 16)


def test_staged_mxfp8_stream_rows():
    rows = (257, 384, 480, 481, 512, 641, 1024, 1281, 2049, 4095)
    run_on_4_gpus(_worker, rows, False, False, 16)
    run_on_4_gpus(_worker, (1, 8, 13, 16) + rows, False, False, 0)
    run_on_4_gpus(_worker, (300, 1029, 4095), True, True, 0)


def test_staged_eager_between_tiers():
    # The register kernel covers all eager rows between the one-shot and stream tiers.
    run_on_4_gpus(_worker, (19, 22, 23, 24, 31, 64, 127, 128, 255, 320), False, False, None, True)
    run_on_4_gpus(_worker, (19, 97, 320), True, True, None, True)


def test_staged_register_rows():
    run_on_4_gpus(_worker, (14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 28, 29, 31, 32, 36, 37, 48, 49, 64, 65, 96, 97, 127, 128, 129, 160, 192, 224, 255, 256, 319, 320))
    run_on_4_gpus(_worker, (19, 29, 64, 97, 127, 128, 193, 256, 320), True, True)


def test_staged_mxfp8_register_rows():
    run_on_4_gpus(_worker, (16, 18, 19, 20, 21, 22, 23, 24, 28, 29, 31, 32, 36, 37, 48, 49, 64, 65, 96, 97, 127, 128, 255, 256), False, False, 16)
    run_on_4_gpus(_worker, (19, 29, 64, 97, 256), True, True, 16)



def test_staged_register_unregistered_graphs():
    rows = (14, 16, 19, 28, 29, 32, 64, 96, 97, 128, 255, 256, 320, 19)
    run_on_4_gpus(_worker, rows, True, True, None, False, False)
    run_on_4_gpus(_worker, rows[:-2], True, True, 16, False, False)



def test_staged_strided_inputs_and_residuals():
    rows = (1,3,7,8,10,13,14,16,19,32,128,320,513,1027)
    run_on_4_gpus(_worker,rows,True,True,None,False,True,True)
    run_on_4_gpus(_worker,rows,True,True,16,False,True,True)


def _reuse_worker(rank, errors, tile=None, rows_list=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,32)):
    """A producer recycles the same input allocation right after every tail (fixed distinct inputs cannot show this)."""
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
    from tests.mfma_spinner import mfma_spinner

    dev = torch.device('cuda',rank)
    w = (torch.randn(H,generator=torch.Generator().manual_seed(7))*.5).bfloat16().to(dev)
    comm = arn.staged_comm(rank,dist.group.WORLD,dev)
    comm.warm_up(w,EPS)
    spin = mfma_spinner(128)
    side = torch.cuda.Stream(priority=-1)
    calls = 24
    for rows in rows_list:
        every = [_inputs(rows,c,dev) for c in range(calls)]
        x = torch.empty((rows,H),dtype=torch.bfloat16,device=dev)
        residual = _residual(rows,dev)
        outputs = []
        graph = torch.cuda.CUDAGraph()
        with comm.capture(), torch.cuda.graph(graph):
            for inputs in every:
                x.copy_(inputs[rank])
                if tile is None:
                    out = comm.ar_add_rmsnorm(x,residual,w,EPS)
                else:
                    out = comm.ar_add_rmsnorm_mxfp8(x,residual,w,EPS,tile)
                outputs.append(out)
                # The caller may reuse this scratch immediately after return.
                x.fill_(float('nan'))
        for rep in range(2):
            if rep:
                for inputs in every:
                    for source in inputs: source.neg_()
            torch.cuda.synchronize()
            dist.barrier()
            if rank==3:
                spin(200000,side)
            graph.replay()
            torch.cuda.synchronize()
            for c,inputs in enumerate(every):
                red = allreduce_rows_reference(inputs)
                if tile is None:
                    want = an.gemma_add_rmsnorm(red,residual,w,EPS)
                else:
                    want = an.gemma_add_rmsnorm_mxfp8(red,residual,w,EPS,16)
                for k,(got,ref) in enumerate(zip(_written(outputs[c],rows,tile),_written(want,rows,16 if tile is not None else None))):
                    if not torch.equal(_bits(got),_bits(ref)):
                        errors.append(f'rank {rank} rows {rows} rep {rep} call {c} output {k}: input reuse')
        del graph,outputs,every


def test_staged_producer_reuses_input_immediately():
    run_on_4_gpus(_reuse_worker)
    run_on_4_gpus(_reuse_worker,16)
    # the lossless all-gather tier of the registered pull (rank 3 delayed)
    run_on_4_gpus(_reuse_worker,None,(127,128,200,255,256))


def test_staged_registered_stream_producer_reuses_input_immediately():
    # Registered graphs above the register tier read the peers' inputs in place.
    run_on_4_gpus(_reuse_worker,None,(321,333,512,641,1000,1281,2049,4095))
    run_on_4_gpus(_reuse_worker,16,(257,333,512,641,1000,1281,2049,3073,4095))
    run_on_4_gpus(_reuse_worker,0,(257,641,1024,1281,2049,3073,4095))


def test_staged_registered_stream_against_staged_stream():
    # the same registered capture with the in-place stream tier moved out of range (the staged stream): the same bits,
    # both outputs and code layouts
    run_on_4_gpus(_registered_stream_worker)
    run_on_4_gpus(_registered_stream_worker,16)
    run_on_4_gpus(_registered_stream_worker,0)


def _registered_stream_worker(rank, errors, tile=None):
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm import staged_host as sh

    mx = tile is not None
    attr = 'stream_pull_min_rows_mxfp8' if mx else 'stream_pull_min_rows'
    default = sh.STREAM_PULL_MIN_ROWS_MXFP8 if mx else sh.STREAM_PULL_MIN_ROWS
    dev = torch.device('cuda',rank)
    w = (torch.randn(H,generator=torch.Generator().manual_seed(7))*.5).bfloat16().to(dev)
    comm = arn.staged_comm(rank,dist.group.WORLD,dev)
    comm.warm_up(w,EPS)
    for rows in (257,321,480,481,641,1024,2049,4095):
        every = [_inputs(rows,c,dev) for c in range(3)]
        mine = [inputs[rank] for inputs in every]
        residual = _residual(rows,dev)
        outs = {}
        for name, lo in (('pull', default), ('staged', 10**9)):
            setattr(comm, attr, lo)
            graph = torch.cuda.CUDAGraph()
            with comm.capture(), torch.cuda.graph(graph):
                if mx:
                    o = [comm.ar_add_rmsnorm_mxfp8(x,residual,w,EPS,tile) for x in mine]
                else:
                    o = [comm.ar_add_rmsnorm(x,residual,w,EPS) for x in mine]
            setattr(comm, attr, default)
            graph.replay()
            torch.cuda.synchronize()
            outs[name] = [_written(tuple(t.clone() for t in oo),rows,tile) for oo in o]
            del graph
        for c in range(3):
            for k,(a,b) in enumerate(zip(outs['pull'][c],outs['staged'][c])):
                if not torch.equal(_bits(a),_bits(b)):
                    errors.append(f'rank {rank} rows {rows} tile {tile} call {c} output {k}: registered stream != staged stream')


def _flag_lines_worker(rank,errors):
    """Alternate flag layouts on one staging/count stream without CPU barriers."""
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import allreduce_rows_reference
    from tests.mfma_spinner import mfma_spinner
    dev=torch.device('cuda',rank)
    c=arn.staged_comm(rank,dist.group.WORLD,dev)
    w=torch.randn(H,device=dev).bfloat16()
    c.warm_up(w,EPS)
    rows_list=[1,3,9,8,2,13,7,8]*3
    every=[];sources=[];residuals=[]
    for i,rows in enumerate(rows_list):
        parts=[]
        for p in range(WORLD):
            g=torch.Generator().manual_seed(8123+i*17+p)
            v=torch.randn((16,H),generator=g).bfloat16()
            v[:,:32]=2.**-133 if p%2 else -2.**-133
            parts.append(v)
        every.append(parts);sources.append(parts[rank].to(dev))
        r=torch.randn((16,H),device=dev).bfloat16();r[:,:32]=2.**-133
        residuals.append(r)
    x=torch.empty_like(sources[0])
    spinner=mfma_spinner(64);compute=torch.cuda.Stream(priority=-1)
    graph=torch.cuda.CUDAGraph();got=[]
    with torch.cuda.graph(graph):
        for i,rows in enumerate(rows_list):
            x.copy_(sources[i])
            if rank==3:torch.cuda._sleep(10000)
            got.append(c.ar_add_rmsnorm(x[:rows],residuals[i][:rows],w,EPS))
            x.fill_(float('nan'))
    for rep in range(3):
        if rep==1:
            for src in sources:src.neg_()
            for parts in every:
                for part in parts:part.neg_()
        expected=[]
        for i,rows in enumerate(rows_list):
            red=allreduce_rows_reference([p[:rows] for p in every[i]])
            expected.append(gemma_add_rmsnorm(red.to(dev),residuals[i][:rows],w,EPS))
        for outs in got:
            for t in outs:t.view(torch.uint8).fill_(0xFF)
        torch.cuda.synchronize();spinner(2000000,compute)
        graph.replay();torch.cuda.synchronize()
        for a,b in zip(got,expected):
            for y,z in zip(a,b):
                if not torch.equal(y.view(torch.uint8),z.view(torch.uint8)):
                    errors.append(f'mixed flag lines rank {rank} rep {rep}')


def test_staged_oneshot_mixed_flag_lines_and_immediate_reuse():
    run_on_4_gpus(_flag_lines_worker)



def test_registered_batch_protocol_shape_changes():
    # batch widths, owner maps and both handshake protocols on one communicator (eager calls stage in sub-batches,
    # registered graphs may publish one whole batch): every reused flag position stays paired with its own next
    # generation across the shape changes
    rows = (512,321,2049,641,3072,3073,481,1281,769,4095,257,1024)
    for tile in (None,16,0):
        run_on_4_gpus(_worker, rows, True, True, tile, False, True, False, timeout=1800)
