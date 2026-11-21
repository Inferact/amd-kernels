# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank contract at the served row counts: eager and graph calls, replayed and changed inputs, determinism."""
import torch
import torch.distributed as dist
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 import LatentTailComm, replicated_partial_buffer
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 import allreduce_latent_rmsnorm_up_proj as call
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.reference import latent_tail_reference

from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.multi_gpu8 import run_on_8_gpus
from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.workload import SERVED_ROWS, make


def bits(t):
    return t.contiguous().view(torch.uint8)


def check(out, parts, nw, w, shared0, tag):
    y, up, _ = latent_tail_reference(parts, nw, w, shared0)
    o = out.double()
    err = (o - y).abs()
    tol = 2.0**-8 * y.abs() + 2.0**-8 * up.abs().amax(-1, keepdim=True) + 1e-6
    assert bool((err <= tol).all()), ("out", tag, float(err.max()), float((err / tol).max()))
    cos = torch.nn.functional.cosine_similarity(o, y, dim=1)
    assert float(cos.min()) >= 0.99999, ("cos", tag, float(cos.min()))


def contract(c, rank, dev, rows, tag):
    for m in rows:
        for ci in range(1):
            parts, nw, w, shared0 = make(m, m * 10 + ci, dev, rank)
            shared = torch.zeros(m, 7168, dtype=torch.bfloat16, device=dev)  # stock layout: a view of [m, 7168]
            view = shared[:, rank * 896:(rank + 1) * 896]
            view.copy_(shared0)
            call(c, parts[rank], nw, w, view)
            torch.cuda.synchronize()
            check(view, parts, nw, w, shared0, (tag, "eager", m, ci))
            first = view.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call(c, parts[rank], nw, w, view)
            for it in range(2):
                view.copy_(shared0)
                g.replay()
                torch.cuda.synchronize()
                assert torch.equal(bits(view), bits(first)), ("replay", it, tag, m, ci)
                check(view, parts, nw, w, shared0, (tag, "graph", m, ci))
            for x in parts:
                x.neg_()
            view.copy_(shared0)
            g.replay()
            torch.cuda.synchronize()
            check(view, parts, nw, w, shared0, (tag, "changed", m, ci))
        if rank == 0:
            print("contract PASS", tag, "rows", m, flush=True)


def worker(rank, errors):
    dev = torch.device("cuda", rank)
    c = LatentTailComm(rank, dist.group.WORLD, dev)
    contract(c, rank, dev, SERVED_ROWS, "fused")
    c.close()
    split_and_replicated(rank, dev)


def split_and_replicated(rank, dev):
    """The two-launch mode, and the replicated-shared layout: ``out`` a compact shard view apart from ``shared``."""
    c = LatentTailComm(rank, dist.group.WORLD, dev, fused=False)
    for m in (12, 80, 192):
        parts, nw, w, shared0 = make(m, 7000 + m, dev, rank)
        full = replicated_partial_buffer(m, dev)  # persistent -0.0 padded partial buffer
        out = full[:, rank * 896:(rank + 1) * 896]
        for it in range(3):
            call(c, parts[rank], nw, w, shared0, out)
            torch.cuda.synchronize()
            check(out, parts, nw, w, shared0, ("split", m, it))
            for x in parts:
                x.mul_(-1.0)
        other = full.clone().view(torch.int16)
        other[:, rank * 896:(rank + 1) * 896] = -32768
        assert bool((other == -32768).all()), ("columns outside the shard were written", m)
    c.close()


def test_latent_tail_contract():
    run_on_8_gpus(worker, timeout=900)
