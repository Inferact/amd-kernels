# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``ExpertParallel`` on 8 GPUs: the fused dispatch / experts / combine of every rank's tokens equals ``reference.py``
(cosine >= 0.99999 over the output, each row >= 0.9999) for uniform, skewed and all-to-rank-0 routing, MXFP8 and bf16
rows, unequal and empty ranks, ids that are not routed, 48, 49, 56 and 63 experts per rank; two runs and a CUDA graph
replay with new inputs are bitwise identical to the eager run; nothing compiles after the warm-up. Skipped with fewer
than 8 GPUs (``AKR_OVERSUBSCRIBE=1`` runs the 8 ranks on the GPUs there are: a plumbing check only)."""

import torch
import torch.distributed as dist
from amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8 import ExpertParallel, ExpertWeights
from amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8 import reference as ref
from tests.multi_gpu8 import run_on_8_gpus

from . import workload as wl

WORLD = 8
CHECKED = 24  # tokens per rank compared with the reference (a token's row is independent of the others)


def _cos(a, b):
    return (torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item(),
            torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=1).min().item())


def worker(rank, errors, epr, cases, graph_cases):
    from flydsl.compiler.jit_function import MlirCompiler

    dev = torch.device("cuda", torch.cuda.current_device())
    raws = [wl.make_experts(100 + r, dev, epr) for r in range(WORLD)]
    w = ExpertWeights.from_row_major(*raws[rank])
    ep = ExpertParallel(rank, WORLD, dist.group.WORLD, dev, epr=epr)
    ep.warm_up(w)
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        for ci, (kind, counts, bf16, bad_ids) in enumerate(cases):
            m = counts[rank]
            ids, tw = wl.make_routes(max(m, 1), WORLD, kind, 31 * ci + rank, dev, epr)
            ids, tw = ids[:m], tw[:m]
            if bad_ids and m > 2:
                ids[1, 2] = -1
                ids[2, 0] = epr * WORLD + 5
            x, codes, scales = wl.make_tokens(max(m, 1), 77 * ci + rank, dev)
            x, codes, scales = x[:m], codes[:m], scales[:m]
            args = (x, None) if bf16 else (codes, scales)
            out = ep(*args, w, tw, ids)
            torch.cuda.synchronize()
            again = ep(*args, w, tw, ids)
            torch.cuda.synchronize()
            tag = f"rank {rank} case {ci} {kind} counts {counts} {'bf16' if bf16 else 'mxfp8'}"
            if not torch.equal(out, again):
                errors.append(f"{tag}: two runs differ")
            # other tokens on the same routes (the same arena rows and slots), then the first tokens again
            x2, c2, s2 = wl.make_tokens(max(m, 1), 1000 + 77 * ci + rank, dev)
            ep(*((x2[:m], None) if bf16 else (c2[:m], s2[:m])), w, tw, ids)
            third = ep(*args, w, tw, ids)
            torch.cuda.synchronize()
            if not torch.equal(out, third):
                errors.append(f"{tag}: the run after other tokens on the same routes differs")
            k = min(m, CHECKED)
            if k:
                q, s = ref.quantize_mxfp8(x[:k]) if bf16 else (codes[:k], scales[:k])
                want = ref.ep_reference(q, s, raws, tw[:k], ids[:k])
                cos, row = _cos(out[:k], want)
                if not (cos >= 0.99999 and row >= 0.9999):
                    errors.append(f"{tag}: cosine {cos:.7f}, worst row {row:.6f}")
                if torch.isnan(out.float()).any():
                    errors.append(f"{tag}: NaN")
        # CUDA graph: replay with new inputs equals the eager run on them, bit for bit
        for graph_counts in graph_cases:
            m = graph_counts[rank]
            sx, _, _ = wl.make_tokens(max(m, 1), 5 + rank, dev)
            sids, stw = wl.make_routes(max(m, 1), WORLD, "uniform", 9 + rank, dev, epr)
            sx, sids, stw = sx[:m], sids[:m], stw[:m]
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = ep(sx, None, w, stw, sids)
            for seed in (1, 2, 3):
                nx, _, _ = wl.make_tokens(max(m, 1), seed * 10 + rank, dev)
                nids, ntw = wl.make_routes(max(m, 1), WORLD, "uniform", seed * 10 + rank, dev, epr)
                sx.copy_(nx[:m])
                sids.copy_(nids[:m])
                stw.copy_(ntw[:m])
                g.replay()
                torch.cuda.synchronize()
                eager = ep(sx, None, w, stw, sids)
                torch.cuda.synchronize()
                if not torch.equal(gout, eager):
                    errors.append(f"rank {rank}: graph of {graph_counts} replay {seed} differs from eager")
    finally:
        MlirCompiler.compile = orig


UNIFORM = [16] * WORLD


def test_ranks8_experts48():
    cases = [
        ("uniform", UNIFORM, True, False),
        ("uniform", [64] * WORLD, False, True),
        ("skew", [192] * WORLD, True, False),
        ("hot", [0, 5, 300, 64, 1, 0, 128, 33], True, False),
        ("uniform", [768] * WORLD, True, False),
    ]
    graphs = [[64] * WORLD, [768] * WORLD, [768, 1, 0, 33, 200, 64, 5, 128]]
    run_on_8_gpus(worker, 48, cases, graphs, timeout=1800)


def test_ranks8_experts49():
    # one replica slot per rank: the geometry of vLLM's shipped DeepSeek-V4.1 expert plan (S = 49), decode rows
    # up to 1024 per rank
    cases = [
        ("skew", [16] * WORLD, True, False),
        ("uniform", [848] * WORLD, True, False),
        ("hot", [1024, 0, 5, 300, 64, 1, 128, 33], True, False),
    ]
    run_on_8_gpus(worker, 49, cases, [[64] * WORLD, [848] * WORLD], timeout=1800)


def test_ranks8_experts56():
    cases = [("uniform", [64] * WORLD, True, False), ("skew", [160] * WORLD, False, False)]
    run_on_8_gpus(worker, 56, cases, [[32] * WORLD], timeout=1800)


def test_ranks8_experts63():
    cases = [("uniform", [48, 0, 17, 130, 1, 64, 300, 5], True, True)]
    run_on_8_gpus(worker, 63, cases, [[24] * WORLD], timeout=1800)
