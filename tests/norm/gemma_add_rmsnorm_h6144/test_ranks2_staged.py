# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The attention tail's two-rank all-reduce + add + Gemma RMSNorm (``allreduce_add_rmsnorm_ranks2.staged_comm``), bf16
out and MXFP8 out (tiled and row-major codes), on 2 gfx950 GPUs:

- against the contract: ``reference.allreduce_rows_reference`` of the 2 partials, then the one-GPU
  ``gemma_add_rmsnorm`` (``gemma_add_rmsnorm_mxfp8``, tile 16) on it, bit for bit (out or written codes and scales, and
  residual out; row-major codes = the tiled norm's codes untiled), and the fp64 norm / MXFP8 contract of reference.py
  on the first and last call; denormals and signed zeros in the partials, cancellations into denormals;
- every rows-per-batch tier and its edges, the CUDA graph batch sizes of a DSpark-7 decode (8 rows per request);
- eager calls, each with its own inputs, the residual flowing from call to call; then the same calls captured in a
  graph and replayed into guarded outputs: bit for bit the eager results, with changed-input replays and eager calls
  interleaved; next to MFMA waves on a high-priority stream; strided partials and residuals;
- a producer that overwrites its input right after each call, and rank skew with mixed row counts and outputs.
"""

import torch
import torch.distributed as dist
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
    CODE_MISMATCH,
    MXFP8_COS,
    OUT_COS,
    OUT_MISMATCH,
    SCALE_MISMATCH,
    H,
    add_rmsnorm_reference,
    allreduce_rows_reference,
    compare_mxfp8,
    compare_out,
    unshuffle_scales,
    untile_codes,
)

from tests.multi_gpu import run_on_2_gpus

WORLD = 2
EPS = 1e-6
CALLS = 4
DECODE_ROWS = (8, 16, 24, 64, 128, 256, 512, 1024)


def _inputs(rows, call, dev, sensitive=False):
    """Both ranks' bf16 [rows, H] partials of call ``call`` (each rank makes both: the reference needs them)."""
    out = []
    for p in range(WORLD):
        g = torch.Generator(device=dev).manual_seed(rows * 1000 + call * 10 + p)
        x = torch.randn(rows, H, generator=g, device=dev) * (0.5 + p)
        x[:, :32] *= 40
        x[:, 64:72] = 2.0**-133  # bf16 denormals: flushed by the rank sum
        x[:, 72:80] = -0.0
        if sensitive:
            # normal operands cancel into a denormal, which the rank sum flushes
            x[:, 128:160] = 2.0**-125 if p == 0 else -(2.0**-125) + 2.0**-133
            # zero partials: a denormal residual must survive
            x[:, 160:192] = 0.0
            # finite operands whose first row's sum of squares overflows fp32: the norm returns signed zeros
            x[0, 192:224] = 1.0e20
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


def _worker(rank, errors, row_counts, spin=False, sensitive=False, tile=None, strided=False):
    """``tile`` None: bf16 out; else MXFP8 out with that code tile (0: row-major)."""
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm_ranks2 as arn

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
            y, res_ref = add_rmsnorm_reference(x, r, w, EPS)
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
            r = eager[c][-1]
        guards = []

        def guarded(t):
            buf = torch.full((t.numel() * t.element_size() + 32,), 0x5A, dtype=torch.uint8, device=dev)
            guards.append(buf)
            return buf[16:-16].view(t.dtype).view(t.shape)

        outs = [[guarded(t) for t in o] for o in eager]
        graph = torch.cuda.CUDAGraph()
        MlirCompiler.compile = counting
        try:
            with comm.capture(), torch.cuda.graph(graph):
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


TIER_ROWS = (1, 2, 3, 7, 8, 9, 47, 48, 49, 63, 65, 127, 255, 256, 257, 511, 1000, 1024, 1025, 1536)
LARGE_ROWS = (2048, 2049, 3073, 4095, 4096, 8191)  # above 4095: launches of CHUNK_ROWS


def test_ranks2_bf16_rows():
    run_on_2_gpus(_worker, TIER_ROWS + DECODE_ROWS + LARGE_ROWS)


def test_ranks2_bf16_denormal_next_to_mfma():
    run_on_2_gpus(_worker, (1, 8, 48, 49, 257, 1025, 2049, 4095), True, True)


def test_ranks2_mxfp8_tiled_rows():
    run_on_2_gpus(_worker, TIER_ROWS + DECODE_ROWS + LARGE_ROWS, False, False, 16)
    run_on_2_gpus(_worker, (1, 8, 48, 49, 257, 1000), True, True, 32)


def test_ranks2_mxfp8_rowmajor_rows():
    run_on_2_gpus(_worker, TIER_ROWS + LARGE_ROWS, False, False, 0)
    run_on_2_gpus(_worker, (1, 8, 49, 300, 1029, 2049, 4095), True, True, 0)


def test_ranks2_strided_inputs_and_residuals():
    rows = (1, 3, 8, 13, 48, 49, 320, 513, 1027, 2050)
    run_on_2_gpus(_worker, rows, True, True, None, True)
    run_on_2_gpus(_worker, rows, True, True, 16, True)


def _reuse_worker(rank, errors, tile=None, rows_list=(1, 2, 7, 8, 16, 48, 49, 200, 257, 1025, 2049, 4095)):
    """A producer recycles the same input allocation right after every tail."""
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm_ranks2 as arn
    from tests.mfma_spinner import mfma_spinner

    dev = torch.device("cuda", rank)
    w = (torch.randn(H, generator=torch.Generator().manual_seed(7)) * 0.5).bfloat16().to(dev)
    comm = arn.staged_comm(rank, dist.group.WORLD, dev)
    comm.warm_up(w, EPS)
    spin = mfma_spinner(128)
    side = torch.cuda.Stream(priority=-1)
    calls = 12
    for rows in rows_list:
        every = [_inputs(rows, c, dev) for c in range(calls)]
        x = torch.empty((rows, H), dtype=torch.bfloat16, device=dev)
        residual = _residual(rows, dev)
        outputs = []
        graph = torch.cuda.CUDAGraph()
        with comm.capture(), torch.cuda.graph(graph):
            for inputs in every:
                x.copy_(inputs[rank])
                if tile is None:
                    outputs.append(comm.ar_add_rmsnorm(x, residual, w, EPS))
                else:
                    outputs.append(comm.ar_add_rmsnorm_mxfp8(x, residual, w, EPS, tile))
                x.fill_(float("nan"))
        for rep in range(2):
            if rep:
                for inputs in every:
                    for source in inputs:
                        source.neg_()
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 1:
                spin(200000, side)
            graph.replay()
            torch.cuda.synchronize()
            for c, inputs in enumerate(every):
                red = allreduce_rows_reference(inputs)
                if tile is None:
                    want = an.gemma_add_rmsnorm(red, residual, w, EPS)
                else:
                    want = an.gemma_add_rmsnorm_mxfp8(red, residual, w, EPS, 16)
                got = _written(outputs[c], rows, tile)
                for k, (a, b) in enumerate(zip(got, _written(want, rows, 16 if tile is not None else None))):
                    if not torch.equal(_bits(a), _bits(b)):
                        errors.append(f"rank {rank} rows {rows} rep {rep} call {c} output {k}: input reuse")
        del graph, outputs, every


def test_ranks2_producer_reuses_input_immediately():
    run_on_2_gpus(_reuse_worker)
    run_on_2_gpus(_reuse_worker, 16)
    run_on_2_gpus(_reuse_worker, 0, (1, 49, 1029, 4095))


def _skew_worker(rank, errors):
    """Eager calls with mixed row counts and outputs (every rows-per-batch tier, the flag words of one batch tier read
    by another) while one rank sleeps between calls: the call counts and staged copies stay paired."""
    import time

    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm_ranks2 as arn

    dev = torch.device("cuda", rank)
    comm = arn.staged_comm(rank, dist.group.WORLD, dev)
    w = torch.ones(H, dtype=torch.bfloat16, device=dev)
    comm.warm_up(w, EPS)
    history = []
    pattern = (7, 300, 64, 1, 49, 2048, 8, 257, 4095, 130, 3, 512, 3500, 48, 1025)
    for call in range(48):
        m = pattern[call % len(pattern)]
        parts = [torch.randn((m, H), generator=torch.Generator().manual_seed(1000 * call + r)).bfloat16().pin_memory()
                 for r in range(WORLD)]
        residual = torch.randn((m, H), generator=torch.Generator().manual_seed(77 + call)).bfloat16().to(dev)
        if rank == call % 2 and call % 3 == 0:
            time.sleep(0.01)
        x = parts[rank].to(dev, non_blocking=True)
        tile = (None, 16, 0)[call % 3]
        out = comm.ar_add_rmsnorm(x, residual, w, EPS) if tile is None else comm.ar_add_rmsnorm_mxfp8(
            x, residual, w, EPS, tile)
        history.append((parts, residual, out, m, tile))
    torch.cuda.synchronize()
    for parts, residual, out, m, tile in history:
        red = allreduce_rows_reference([p.to(dev) for p in parts])
        if tile is None:
            want = an.gemma_add_rmsnorm(red, residual, w, EPS)
        else:
            want = an.gemma_add_rmsnorm_mxfp8(red, residual, w, EPS, 16)
        for k, (a, b) in enumerate(zip(_written(out, m, tile), _written(want, m, 16 if tile is not None else None))):
            if not torch.equal(_bits(a), _bits(b)):
                errors.append(f"rank {rank} rows {m} tile {tile} output {k}: skewed eager call differs")


def test_ranks2_eager_rank_skew_mixed_tiers():
    run_on_2_gpus(_skew_worker, timeout=600)
