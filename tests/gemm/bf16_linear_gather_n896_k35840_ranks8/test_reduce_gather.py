# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Slab reduce + eight-rank gather + RMSNorm at the served row counts."""
import torch
import torch.distributed as dist
from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8.comm import ProjComm
from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8.ipc import f32_bits, run_compiled
from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8.reduce_gather import compile_reduce_gather
from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8.reference import N_FULL, N_SHARD, gather_norm_reference

from tests.gemm.bf16_linear_gather_n896_k35840_ranks8.multi_gpu8 import run_on_8_gpus

ROWS = (1, 12, 40, 65, 80, 192)
SPLITS = (1, 7, 35)


def worker(rank, errors):
    dev = torch.device("cuda", rank)
    comm = ProjComm(rank, dist.group.WORLD, dev)
    eps = 1e-6
    for split in SPLITS:
        for m in ROWS:
            g = torch.Generator().manual_seed(m * 100 + split)
            slabs = [(torch.randn(split, m, N_SHARD, generator=g) * 3).to(dev) for _ in range(8)]
            w = (torch.rand(N_FULL, generator=g) + 0.5).bfloat16().to(dev)
            shards = [s.sum(0).bfloat16() for s in slabs]
            ref = gather_norm_reference(shards, w, eps)
            out = torch.empty(m, N_FULL, dtype=torch.bfloat16, device=dev)
            k = compile_reduce_gather(split)
            for it in range(3):
                out.fill_(0)
                run_compiled(k, m, rank, slabs[rank].data_ptr(), comm.buf_table.data_ptr(), comm.counts.data_ptr(),
                             w.data_ptr(), out.data_ptr(), f32_bits(eps), torch.cuda.current_stream())
                torch.cuda.synchronize()
                err = (out.float() - ref.float()).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=1).min().item()
                if not (cos >= 0.99999 and err < 0.1):
                    errors.append(f"rank {rank} split {split} rows {m} it {it}: cos {cos} maxerr {err}")
    comm.close()


def test_reduce_gather():
    run_on_8_gpus(worker, timeout=600)
