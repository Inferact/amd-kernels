# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MoE combine fused into the 4-rank all-reduce + add + Gemma RMSNorm + MXFP8 quant
(``SlotsArNormComm.slots4_ar_add_rmsnorm_mxfp8``) on 4 gfx950 GPUs:

- against the contract: ``reference.slots4_allreduce_reference`` then the one-GPU ``gemma_add_rmsnorm_mxfp8`` on it,
  bit for bit (codes, scales, residual out; written bytes only), and the residual out bit for bit
  ``add_rmsnorm_reference``; the MXFP8 tolerances of reference.py on the first and last call; denormals and signed
  zeros in the slots;
- row counts that are not multiples of 4 or 16 (q = ceil(rows / 4)), 1-4095;
- eager calls, each with its own inputs, the residual flowing from call to call; then the same calls captured in a
  graph and replayed into NaN-filled outputs: bit for bit the eager results, with changed-input and repeated replays;
- rule 4: after ``warm_up``, no row count compiles anything.
"""

import torch
import torch.distributed as dist
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
    CODE_MISMATCH,
    MXFP8_COS,
    SCALE_MISMATCH,
    WORLD,
    H,
    add_rmsnorm_reference,
    compare_mxfp8,
    slots4_allreduce_reference,
    unshuffle_scales,
    untile_codes,
)

from tests.multi_gpu import run_on_4_gpus

EPS = 1e-6
CALLS = 4
TILE = 16


def _inputs(rows, call, dev, sensitive=False):
    """Every rank's (slots [rows, 4, H], shared [rows, H]) of call ``call``, bf16 (each rank makes all four: the
    reference needs them)."""
    out = []
    for p in range(WORLD):
        g = torch.Generator(device=dev).manual_seed(rows * 1000 + call * 10 + p)
        s = torch.randn(rows, 4, H, generator=g, device=dev) * (0.5 + p)
        sh = torch.randn(rows, H, generator=g, device=dev)
        s[:, :, :32] *= 40
        if rows > 4:
            tiny = 2.0**-133  # a bf16 denormal: kept by the local sum, flushed after
            s[:4, :, 64:72] = tiny
            s[1, :, 72:80] = -0.0
            sh[:4, 64:80] = -0.0
        if sensitive:
            # Local subnormals cross into the normal range before rank FTZ.
            s[:, :, 96:128] = 2.0**-128
            sh[:, 96:128] = 0.0
            # Normal rank operands cancel into a subnormal, which rank FTZ must flush.
            s[:, :, 128:160] = 0.0
            sh[:, 128:160] = 0.0
            if p == 0:
                s[:, 0, 128:160] = 2.0**-125
            elif p == 1:
                s[:, 0, 128:160] = -2.0**-125 + 2.0**-133
            # A subnormal residual must survive after the previous mode is restored.
            s[:, :, 160:192] = 0.0
            sh[:, 160:192] = 0.0
        out.append((s.bfloat16(), sh.bfloat16()))
    return out


def _residual(rows, dev, salt=0):
    g = torch.Generator(device=dev).manual_seed(rows + salt * 7919)
    return torch.randn(rows, H, generator=g, device=dev).bfloat16()


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _written(o, rows, tile=TILE):
    q, s, res = o
    return [untile_codes(q, rows) if tile else q[:rows].view(torch.uint8), unshuffle_scales(s, rows), res]


def _worker(rank, errors, row_counts, tile=TILE, spin=False, sensitive=False):
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.host import (
        SlotsArNormComm,
    )

    dev = torch.device("cuda", rank)
    if spin:
        from tests.mfma_spinner import mfma_spinner
        spinner = mfma_spinner(32)
        side = torch.cuda.Stream(priority=-1)
    w = (torch.randn(H, generator=torch.Generator().manual_seed(7)) * 0.5).bfloat16().to(dev)
    comm = SlotsArNormComm(rank, dist.group.WORLD, dev)
    compiled = []
    from flydsl.compiler.jit_function import MlirCompiler
    real = MlirCompiler.compile
    comm.warm_up(w, EPS, tile)

    def counting(fn, *args, **kwargs):
        compiled.append(type(fn).__name__)
        return real(fn, *args, **kwargs)

    for rows in row_counts:
        tag = f"rank {rank} rows {rows}"
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
            x = slots4_allreduce_reference(
                [every[c][p][0] for p in range(WORLD)],
                [every[c][p][1] for p in range(WORLD)],
            )
            ref = an.gemma_add_rmsnorm_mxfp8(x, r, w, EPS, TILE)
            for name, a, b in zip(
                ("codes", "scales", "res"), _written(eager[c], rows, tile), _written(ref, rows)
            ):
                if not torch.equal(_bits(a), _bits(b)):
                    k = int((_bits(a) != _bits(b)).sum())
                    errors.append(f"{tag} call {c} {name}: {k} bytes differ from the reference chain")
            y, res_ref = add_rmsnorm_reference(x, r, w, EPS)  # fp64 on this rank's GPU
            if not torch.equal(_bits(eager[c][2]), _bits(res_ref)):
                errors.append(f"{tag} call {c}: residual out differs from the reference")
            if c in (0, CALLS - 1):
                q, s, _ = eager[c]
                st = compare_mxfp8(untile_codes(q, rows) if tile else q[:rows].view(torch.uint8), unshuffle_scales(s, rows), y)
                ok = (
                    st["scale_max_diff"] <= 1
                    and st["scale_mismatch"] <= SCALE_MISMATCH
                    and st["code_mismatch"] <= CODE_MISMATCH
                    and st["code_far"] == 0
                    and st["cos"] >= MXFP8_COS
                )
                if not ok:
                    errors.append(f"{tag} call {c}: contract {st}")
                if rank == 0 and c == 0:
                    print(f"{tag} call {c}: {st}", flush=True)
            r = eager[c][2]
        # the same calls in a graph, replayed into NaN-filled outputs
        guards = []
        def guarded(t):
            # Preserve the dtype byte width and the output's shape.
            buf = torch.full((t.numel() * t.element_size() + 32,), 0x5A,
                             dtype=torch.uint8, device=dev)
            guards.append(buf)
            return buf[16:-16].view(t.dtype).view(t.shape)
        outs = [[guarded(t) for t in o] for o in eager]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            r = r0
            for (s, sh), o in zip(mine, outs):
                comm.slots4_ar_add_rmsnorm_mxfp8(
                    s, sh, r, w, EPS, tile, q_out=o[0], scale_out=o[1], res_out=o[2]
                )
                r = o[2]
        for rep in range(3):
            if rep == 1:
                # Change every captured rank's inputs, then interleave eager calls.
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


def test_slots4_small_rows():
    run_on_4_gpus(_worker, (1, 5, 8, 9, 16, 13, 24, 32, 66, 128))


def test_slots4_large_rows():
    run_on_4_gpus(_worker, (256, 257, 512, 1000, 1024))


def test_slots4_oneshot():
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.host import ONESHOT_MAX_ROWS

    run_on_4_gpus(_worker, tuple(range(1, ONESHOT_MAX_ROWS + 2)))


def test_slots4_oneshot_next_to_mfma():
    run_on_4_gpus(_worker, (1, 8, 14, 16, 22), TILE, True)


def test_slots4_oneshot_denormal_scopes():
    run_on_4_gpus(_worker, (1, 4, 8, 14, 16, 22), TILE, True, True)


def test_slots4_register_denormal_scopes():
    # the register kernels' rank-sum flush, scoped hardware FP32 FTZ or software masks, on both sides of
    # SCOPED_FTZ_MAX_ROWS next to MFMA waves
    run_on_4_gpus(_worker, (15, 23, 26, 28, 29, 31, 32, 48, 63, 64, 65, 80, 95, 96, 97, 128, 257, 300),
                  TILE, True, True)


def test_slots4_stream_rows():
    run_on_4_gpus(
        _worker,
        (385, 384, 640, 641, 769, 1025, 1280, 1281, 1536, 1537, 2048, 2049, 3073,
         4095, 1025, 767, 1024, 1536, 1537, 3072),
    )


def test_slots4_stream_rowmajor():
    run_on_4_gpus(_worker, (1, 8, 13, 16, 32, 64, 128, 256, 513, 640, 768, 769, 1024, 1025,
         2048, 2049, 3072, 3073, 4095), 0)


def test_slots4_stream_next_to_mfma():
    run_on_4_gpus(_worker, (641, 1024, 1281, 1536, 2049, 3072, 4095), TILE, True)
    run_on_4_gpus(_worker, (13, 641, 768, 769, 1024, 2048, 2049, 3072), 0, True)
