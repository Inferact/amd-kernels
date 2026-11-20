# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks, one fresh ``AllGatherComm`` each: row counts that are not multiples of 32 (or of anything) up to 16384
per rank, rising and falling, through ``gather_mxfp4_moe`` of the i1536 MoE and through direct gathers whose field
count and part counts change from call to call (a field's part count follows its bytes; a call with more parts than
an earlier one must not wait on tickets the earlier one left behind). Every byte vs ``all_gather_reference``, eager
and replayed."""
import torch

from tests.multi_gpu import run_on_2_gpus
from amd_kernels.collectives.pull_collectives_ranks2.reference import all_gather_reference

# 1 -> 129 -> 2049: the sequence that hung (the first call with a 24-part field after calls of one part per field)
MOE_ROWS = (1, 129, 2049, 3071, 4097, 6143, 8191, 12289, 16383, 16384, 2049, 33, 10001, 1)
BF16_ROWS = (2049, 1, 5003, 16383, 77)


def _x(rows, rank, seed):
    g = torch.Generator(device="cuda").manual_seed(seed * 2 + rank)
    return torch.randn((rows, 6144), generator=g, device="cuda", dtype=torch.bfloat16)


def _moe_input(x):
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import MoeInput
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4.quant import mxfp4_quant
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4.shared_quant import shared_quant

    rows = x.shape[0]
    q, s = mxfp4_quant(x)
    q8, s8 = shared_quant(x)
    ids = (torch.arange(rows * 4, device=x.device, dtype=torch.int32).view(rows, 4) * 7 + rows) % 128
    weights = x[:, :4].float().contiguous()
    return MoeInput(q, s, weights, ids, q8, s8)


def _worker(rank, errors):
    from amd_kernels.collectives.pull_collectives_ranks2 import AllGatherComm
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import gather_mxfp4_moe

    dev = torch.device("cuda", rank)
    comm = AllGatherComm(rank, torch.distributed.group.WORLD, dev, (16384 * 2 * 6144 + 6 * 256 + 255) // 256 * 256)
    comm.warm_up()

    def check(got, every, what):
        torch.cuda.synchronize()
        for i, o in enumerate(got):
            want = all_gather_reference([e[i] for e in every])
            if not torch.equal(o.view(torch.uint8), want.view(torch.uint8)):
                errors.append(f"rank {rank} {what} field {i}")

    for step, rows in enumerate(MOE_ROWS):
        every = [_moe_input(_x(rows, p, step)).compact().tensors() for p in range(2)]
        local = _moe_input(_x(rows, rank, step))
        check(gather_mxfp4_moe(local, comm, world=2).tensors(), every, f"moe eager rows {rows}")
        if rows in (2049, 16383):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = gather_mxfp4_moe(local, comm, world=2)
            for t in got.tensors():
                t.view(torch.uint8).fill_(255)
            graph.replay()
            check(got.tensors(), every, f"moe replay rows {rows}")
            del graph

    # one wide field (up to 64 parts), seven fields (two launches), then the compact payload again
    for step, rows in enumerate(BF16_ROWS):
        every = [[_x(rows, p, 100 + step)] for p in range(2)]
        out = torch.empty((2 * rows, 6144), dtype=torch.bfloat16, device=dev)
        comm.all_gather([out], every[rank])
        check([out], every, f"bf16 rows {rows}")
        every7 = [[t[:, :c].contiguous() for c in (8, 16, 2048, 24, 3072, 8, 64)] for t in (e[0] for e in every)]
        outs = [torch.empty((2 * rows, t.shape[1]), dtype=t.dtype, device=dev) for t in every7[rank]]
        comm.all_gather(outs, every7[rank])
        check(outs, every7, f"7 fields rows {rows}")
    every = [_moe_input(_x(2049, p, 999)).compact().tensors() for p in range(2)]
    check(gather_mxfp4_moe(_moe_input(_x(2049, rank, 999)), comm, world=2).tensors(), every, "moe after bf16")


def test_gather_odd_rows():
    run_on_2_gpus(_worker, timeout=600)
