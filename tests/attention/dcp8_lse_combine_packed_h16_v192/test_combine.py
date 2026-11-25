# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract against reference.py (and bit for bit against vLLM's unpack-combine when installed) on 8 GPUs: row counts
across the tiles, eager and graph replay with changed inputs and row counts, empty shards, NaN and +inf LSEs, two runs
bit for bit."""
import pytest
import torch
import torch.distributed as dist

from tests.multi_gpu8 import run_on_8_gpus

TOKENS = ([1, 12, 16, 17, 33], [64, 100, 128, 129, 200], [255, 288, 300, 512])


def _worker(rank, errors, tokens):
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192 import DcpPackedCombine, dcp8_lse_combine_packed
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192.reference import combine_reference

    from .common import has_vllm, make_sends, vllm_combine

    dev = torch.device("cuda", rank)
    exact = has_vllm()
    comm = DcpPackedCombine(rank, dist.group.WORLD, dev)

    def check(tag, got, sends):
        want = combine_reference(sends, rank).float()
        g = got.float()
        err = (g - want).abs().max().item()
        scale = want.abs().max().item() + 1e-6
        if not (g == g).all() or err > scale * 2**-7:
            errors.append(f"rank {rank} {tag}: max err {err:.4g} (scale {scale:.4g})")
        if exact:
            v = vllm_combine(sends, rank)
            n = (got.view(torch.int16) != v.view(torch.int16)).sum().item()
            if n:
                errors.append(f"rank {rank} {tag}: {n} of {v.numel()} values differ from vLLM")

    for t in tokens:
        for step, empty in ((0, ()), (1, (3,))):
            sends = make_sends(t, step, dev, empty)
            out = torch.full((t, 16, 192), float("nan"), dtype=torch.bfloat16, device=dev)
            dcp8_lse_combine_packed(comm, sends[rank], out)
            torch.cuda.synchronize()
            check(f"t{t} step{step} eager", out, sends)
        cap = sends[rank].clone()
        out.fill_(float("nan"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dcp8_lse_combine_packed(comm, cap, out)
        for step in range(2, 5):
            sends = make_sends(t, step, dev)
            cap.copy_(sends[rank])
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check(f"t{t} replay{step}", out, sends)
            first = out.clone()
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(first.view(torch.int16), out.view(torch.int16)):
                errors.append(f"rank {rank} t{t} replay{step}: two runs differ")
        del graph
        if t > 1:
            # A smaller row count after a larger one: stale rows of the pools must not leak.
            sends = make_sends(t // 2, 7, dev)
            out = torch.full((t // 2, 16, 192), float("nan"), dtype=torch.bfloat16, device=dev)
            dcp8_lse_combine_packed(comm, sends[rank], out)
            torch.cuda.synchronize()
            check(f"t{t // 2} after t{t}", out, sends)


@pytest.mark.parametrize("tokens", TOKENS)
def test_combine(tokens):
    run_on_8_gpus(_worker, tokens)
