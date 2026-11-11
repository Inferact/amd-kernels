# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``DecoderArNorm.add_rmsnorm`` writes the bytes of the communicator call it routes to (staged bf16 out, staged
MXFP8 out, slots4 MXFP8 out) at decode and mixed-step row counts, on 4 gfx950 GPUs."""

import torch
import torch.distributed as dist
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import H, unshuffle_scales, untile_codes

from tests.multi_gpu import run_on_4_gpus

EPS = 1e-6


def _codes(q, rows, gemm_weight):
    tile = add_rmsnorm.code_tile_rows_for(rows, gemm_weight)
    return untile_codes(q, rows) if tile else q.view(torch.uint8)[:rows]


def _worker(rank, errors, rows_list):
    dev = torch.device("cuda", rank)
    g = torch.Generator(device=dev).manual_seed(7)
    weight = (torch.randn(H, generator=g, device=dev) * 0.1 + 1).bfloat16()
    comms = arn.DecoderArNorm(rank, dist.group.WORLD, dev, weight, EPS)
    gemm_weight = torch.empty((2560, H), dtype=torch.float8_e4m3fn, device=dev)  # qkv's shape
    for rows in rows_list:
        g = torch.Generator(device=dev).manual_seed(rows * 10 + rank)
        x = torch.randn(rows, H, generator=g, device=dev).bfloat16()
        slots = torch.randn(rows, 4, H, generator=g, device=dev).bfloat16()
        shared = torch.randn(rows, H, generator=g, device=dev).bfloat16()
        res = torch.randn(rows, H, generator=g, device=dev).bfloat16()
        out, r = comms.add_rmsnorm(res.clone(), weight, EPS, partial=x)
        want, wr = comms.staged.ar_add_rmsnorm(x, res.clone(), weight, EPS)
        if not (torch.equal(out, want) and torch.equal(r, wr)):
            errors.append(f"rank {rank} rows {rows}: bf16 out differs")
        for name, kw, direct in (
            ("staged", dict(partial=x),
             lambda: comms.staged.ar_add_rmsnorm_mxfp8(x, res.clone(), weight, EPS, gemm_weight=gemm_weight)),
            ("slots4", dict(slots=slots, shared=shared),
             lambda: comms.slots.slots4_ar_add_rmsnorm_mxfp8(
                 slots, shared, res.clone(), weight, EPS, gemm_weight=gemm_weight)),
        ):
            (q, s), r = comms.add_rmsnorm(res.clone(), weight, EPS, gemm_weight=gemm_weight, **kw)
            wq, ws, wr = direct()
            same = (torch.equal(_codes(q, rows, gemm_weight), _codes(wq, rows, gemm_weight))
                    and torch.equal(unshuffle_scales(s, rows), unshuffle_scales(ws, rows)) and torch.equal(r, wr))
            if not same:
                errors.append(f"rank {rank} rows {rows}: {name} MXFP8 out differs")


def test_decoder_ar_norm_routes():
    run_on_4_gpus(_worker, (1, 8, 33, 256, 1000, 1752))
