# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Unpack with o_proj_full's scale layout on two ranks: codes as the reference unpack, scales bit-equal to the GEMM's
own shuffle of the reference scales (padding rows 127), the GEMM on them bit-equal to the row-major call; both
publication forms, both pool sets on the same flags, late ranks, MFMA work, no compile after warm-up."""
import torch
import torch.distributed as dist
from amd_kernels.collectives.pull_collectives_ranks2.reference import WORLD, all_to_all_reference, unpack_reference
from tests.multi_gpu import run_on_2_gpus
from .served import collectives_slot_bytes

D = 4096  # code bytes per packed row


def _u8(shape, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8, device=dev)


def _packed(shape, seed, dev):
    """Random finite e4m3 codes, e8m0 scales 2^-12..2^-3 (the attention epilogue's range)."""
    x = _u8(shape, seed, dev)
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    x[:, :D] = torch.where((x[:, :D] & 0x7F) == 0x7F, x[:, :D] ^ 1, x[:, :D])
    x[:, D:] = torch.randint(115, 125, (shape[0], D // 32), generator=g, device=dev, dtype=torch.uint8)
    return x


def _worker(rank, errors, row_counts, wgs, joined=False, gemm_rows=(), fence=None):
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels.collectives.pull_collectives_ranks2 import Collectives, FencedAllGatherComm, Pools
    from amd_kernels.collectives.pull_collectives_ranks2.pull_unpack_q4096_shuffled import (
        all_to_all_unpack_q4096_shuffled)
    from amd_kernels.gemm.mxfp8_common.gemm import _preshuffle_activation_scale, mxfp8_gemm, preshuffle_mxfp8_weight
    dev = torch.device('cuda', rank)
    main = torch.cuda.current_stream()
    pools = Pools(rank, WORLD, dist.group.WORLD, dev, collectives_slot_bytes(), sets=2)
    if wgs is None:
        FencedAllGatherComm(pools).warm_up()
    else:
        for f in (True, False):
            c = Collectives(pools, 0).bind(main)
            x = c.alloc((WORLD * 32, D + D // 32), torch.uint8)
            x.zero_()
            all_to_all_unpack_q4096_shuffled(c, x, wgs, f)
        torch.cuda.synchronize()
    if joined:
        from tests.mfma_spinner import mfma_spinner
        spinner = mfma_spinner(128)
        side = torch.cuda.Stream(priority=-1)
    if gemm_rows:
        g = torch.Generator(device=dev).manual_seed(5)
        wq = torch.randint(0, 120, (6144, WORLD * D), generator=g, device=dev, dtype=torch.uint8)
        wq = wq.view(torch.float8_e4m3fn)
        wsc = torch.randint(118, 124, (6144, WORLD * D // 32), generator=g, device=dev, dtype=torch.uint8)
        w, ws = preshuffle_mxfp8_weight(wq, wsc)
    real = MlirCompiler.compile
    compiles = []

    def count(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)
    for rows in row_counts:
        for rep in range(4):
            every = [_packed((WORLD * rows, D + D // 32), rows * 1000 + rep * 10 + p, dev) for p in range(WORLD)]
            c = Collectives(pools, rep % 2).bind(main)
            x = c.alloc(every[rank].shape, torch.uint8)
            x.copy_(every[rank])
            if rank == rep % WORLD:
                torch.cuda._sleep(30000)
            if joined:
                side.wait_stream(main)
                spinner(100000, side)
            MlirCompiler.compile = count
            try:
                codes, scales = all_to_all_unpack_q4096_shuffled(c, x, wgs, fence)
                torch.cuda.synchronize()
            finally:
                MlirCompiler.compile = real
            if compiles:
                errors.append(f'rows {rows} rep {rep}: compile after warm-up')
                compiles.clear()
            want_c, want_s = unpack_reference(all_to_all_reference(every, rank), rows, D)
            if not torch.equal(codes.view(torch.uint8), want_c.view(torch.uint8)):
                errors.append(f'rank {rank} rows {rows} rep {rep}: codes')
            if not torch.equal(scales, _preshuffle_activation_scale(want_s)):
                errors.append(f'rank {rank} rows {rows} rep {rep}: shuffled scales')
            if rows in gemm_rows and rep == 0:
                a = mxfp8_gemm(codes, scales, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
                b = mxfp8_gemm(want_c.view(torch.float8_e4m3fn), want_s, w, ws, torch.bfloat16)
                if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                    errors.append(f'rank {rank} rows {rows}: o_proj_full on the shuffled scales differs')
        dist.barrier()


def test_shuffled_unpack_uneven_rows_and_slot_reuse():
    # served shard rows 2048..16384 (4096..32768 tokens) and uneven rows around the 32-row groups and the fence split
    rows = (1, 7, 31, 32, 33, 96, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8192, 16384, 1)
    run_on_2_gpus(_worker, rows, None, False, (33, 2048, 16384))


def test_shuffled_unpack_workgroups_next_to_mfma():
    for wgs in (16, 64):
        for fence in (True, False):
            run_on_2_gpus(_worker, (32, 1031, 2048, 4128), wgs, True, (), fence)
