# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MoE combine fused into the two-rank all-reduce + add + Gemma RMSNorm + MXFP8 quant
(``allreduce_add_rmsnorm_ranks2.slots4_comm``) on 2 gfx950 GPUs:

- against the contract: ``reference.slots4_allreduce_reference`` of the 2 ranks' slots and shared outputs, then the
  one-GPU ``gemma_add_rmsnorm_mxfp8`` on it, bit for bit (codes, scales, residual out; written bytes only), and the
  MXFP8 tolerances of reference.py on the first and last call; denormals and signed zeros in the slots;
- every rows-per-batch tier and its edges, tiled and row-major codes, 1-4095 rows;
- eager calls, each with its own inputs, the residual flowing from call to call; then the same calls captured in a
  graph and replayed into guarded outputs: bit for bit the eager results, with changed-input replays, next to MFMA.
"""

import torch
import torch.distributed as dist
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
    CODE_MISMATCH,
    MXFP8_COS,
    SCALE_MISMATCH,
    H,
    add_rmsnorm_reference,
    compare_mxfp8,
    slots4_allreduce_reference,
    unshuffle_scales,
    untile_codes,
)

from tests.multi_gpu import run_on_2_gpus

WORLD = 2
EPS = 1e-6
CALLS = 4
TILE = 16


def _inputs(rows, call, dev, sensitive=False):
    """Both ranks' (slots [rows, 4, H], shared [rows, H]) of call ``call``, bf16."""
    out = []
    for p in range(WORLD):
        g = torch.Generator(device=dev).manual_seed(rows * 1000 + call * 10 + p)
        s = torch.randn(rows, 4, H, generator=g, device=dev) * (0.5 + p)
        sh = torch.randn(rows, H, generator=g, device=dev)
        s[:, :, :32] *= 40
        s[:, :, 64:72] = 2.0**-133  # a bf16 denormal: kept by the local sum, flushed by the rank sum
        s[:, 1, 72:80] = -0.0
        sh[:, 64:80] = -0.0
        if sensitive:
            # local denormals cross into the normal range before the rank flush
            s[:, :, 96:128] = 2.0**-128
            sh[:, 96:128] = 0.0
            # normal rank operands cancel into a denormal, which the rank sum flushes
            s[:, :, 128:160] = 0.0
            sh[:, 128:160] = 0.0
            s[:, 0, 128:160] = 2.0**-125 if p == 0 else -(2.0**-125) + 2.0**-133
            s[:, :, 160:192] = 0.0
            sh[:, 160:192] = 0.0
        out.append((s.bfloat16(), sh.bfloat16()))
    return out


def _residual(rows, dev):
    g = torch.Generator(device=dev).manual_seed(rows + 7919)
    return torch.randn(rows, H, generator=g, device=dev).bfloat16()


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _written(o, rows, tile=TILE):
    q, s, res = o
    return [untile_codes(q, rows) if tile else q[:rows].view(torch.uint8), unshuffle_scales(s, rows), res]


def _worker(rank, errors, row_counts, tile=TILE, spin=False, sensitive=False):
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm_ranks2 as arn

    dev = torch.device("cuda", rank)
    if spin:
        from tests.mfma_spinner import mfma_spinner

        spinner = mfma_spinner(32)
        side = torch.cuda.Stream(priority=-1)
    w = (torch.randn(H, generator=torch.Generator().manual_seed(7)) * 0.5).bfloat16().to(dev)
    comm = arn.slots4_comm(rank, dist.group.WORLD, dev)
    comm.warm_up(w, EPS, tile)
    compiled = []
    real = MlirCompiler.compile

    def counting(fn, *args, **kwargs):
        compiled.append(type(fn).__name__)
        return real(fn, *args, **kwargs)

    for rows in row_counts:
        tag = f"rank {rank} rows {rows} tile {tile}"
        every = [_inputs(rows, c, dev, sensitive) for c in range(CALLS)]
        mine = [every[c][rank] for c in range(CALLS)]
        r0 = _residual(rows, dev)
        if sensitive:
            r0[:, 96:160] = 0.0
            r0[:, 160:192] = 2.0**-133
        MlirCompiler.compile = counting
        try:
            eager, r = [], r0
            for s, sh in mine:
                o = comm.slots4_ar_add_rmsnorm_mxfp8(s, sh, r, w, EPS, tile)
                eager.append(o)
                r = o[2]
            torch.cuda.synchronize()
        finally:
            MlirCompiler.compile = real
        if compiled:
            errors.append(f"{tag}: compiled {compiled} after warm_up")
            compiled.clear()
        r = r0
        for c in range(CALLS):
            x = slots4_allreduce_reference([every[c][p][0] for p in range(WORLD)],
                                           [every[c][p][1] for p in range(WORLD)])
            ref = an.gemma_add_rmsnorm_mxfp8(x, r, w, EPS, TILE)
            for name, a, b in zip(("codes", "scales", "res"), _written(eager[c], rows, tile), _written(ref, rows)):
                if not torch.equal(_bits(a), _bits(b)):
                    k = int((_bits(a) != _bits(b)).sum())
                    errors.append(f"{tag} call {c} {name}: {k} bytes differ from the reference chain")
            y, res_ref = add_rmsnorm_reference(x, r, w, EPS)
            if not torch.equal(_bits(eager[c][2]), _bits(res_ref)):
                errors.append(f"{tag} call {c}: residual out differs from the reference")
            if c in (0, CALLS - 1):
                q, s, _ = eager[c]
                st = compare_mxfp8(_written(eager[c], rows, tile)[0], unshuffle_scales(s, rows), y)
                ok = (st["scale_max_diff"] <= 1 and st["scale_mismatch"] <= SCALE_MISMATCH
                      and st["code_mismatch"] <= CODE_MISMATCH and st["code_far"] == 0 and st["cos"] >= MXFP8_COS)
                if not ok:
                    errors.append(f"{tag} call {c}: contract {st}")
            r = eager[c][2]
        guards = []

        def guarded(t):
            buf = torch.full((t.numel() * t.element_size() + 32,), 0x5A, dtype=torch.uint8, device=dev)
            guards.append(buf)
            return buf[16:-16].view(t.dtype).view(t.shape)

        outs = [[guarded(t) for t in o] for o in eager]
        graph = torch.cuda.CUDAGraph()
        MlirCompiler.compile = counting
        try:
            with torch.cuda.graph(graph):
                r = r0
                for (s, sh), o in zip(mine, outs):
                    comm.slots4_ar_add_rmsnorm_mxfp8(s, sh, r, w, EPS, tile, q_out=o[0], scale_out=o[1],
                                                     res_out=o[2])
                    r = o[2]
        finally:
            MlirCompiler.compile = real
        if compiled:
            errors.append(f"{tag}: compiled {compiled} in graph after warm_up")
            compiled.clear()
        for rep in range(3):
            if rep == 1:
                for inputs in every:
                    for slots0, shared0 in inputs:
                        slots0.neg_()
                        shared0.add_(0.5)
                eager, r = [], r0
                for slots0, shared0 in mine:
                    o = comm.slots4_ar_add_rmsnorm_mxfp8(slots0, shared0, r, w, EPS, tile)
                    eager.append(o)
                    r = o[2]
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


def test_ranks2_slots4_tiled_rows():
    run_on_2_gpus(_worker, (1, 2, 5, 8, 16, 24, 47, 48, 49, 64, 65, 128, 255, 256, 257, 512, 1000, 1024, 1025,
                            2048, 2049, 4095, 4096, 7681))


def test_ranks2_slots4_rowmajor_rows():
    run_on_2_gpus(_worker, (1, 8, 13, 48, 49, 256, 257, 1025, 2049, 4095), 0)


def test_ranks2_slots4_denormal_next_to_mfma():
    run_on_2_gpus(_worker, (1, 8, 48, 49, 300, 1025, 2049, 4095), TILE, True, True)
    run_on_2_gpus(_worker, (1, 49, 1029), 0, True, True)
