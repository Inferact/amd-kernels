# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 on 2 ranks, fresh rank processes (this file's first call): the communicators' construction and ``warm_up``
compile every variant, then a decoder layer's phases at the served token counts (2048-32768, off the 32-row grid
too) compile nothing: the input gather, the MoE layer (attention hand-off with the full shared expert, ``mxfp4_moe``
on the gathered routed-only input, the FFN hand-off on its partials), the dense layer's hand-offs, the final reduce-scatter and the gathers; the
same for the SDMA form after its construction. No FlyDSL launch happens inside a warm-up."""

import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_2_gpus

from .workload import H, Q, WORLD, moe_weights, o_proj_weight, rand

WALK_TOKENS = (2048, 2049, 3000, 4096, 4097, 6000, 8192, 8193, 8885, 12289, 15745, 16384, 16385, 24577, 31425, 32768)
SDMA_TOKENS = (2048, 4097, 8885, 12289, 16385)


def _worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import (SdmaSeqParallelHandoffs,
                                                                               SeqParallelHandoffs, shard_rows)
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import mxfp4_moe
    from flydsl.compiler.jit_function import CompiledFunction, MlirCompiler

    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    warm_up.set_workers(8)  # 2 ranks share the lease's CPUs
    dev = torch.device("cuda", rank)
    in_warm_up = []
    launch = CompiledFunction.__call__

    def spy(self, *args):
        if warm_up._in_warm_up.get():
            in_warm_up.append(1)
        return launch(self, *args)

    CompiledFunction.__call__ = spy
    try:
        ph = SeqParallelHandoffs(rank, dist.group.WORLD, dev)
        o_w, o_s = o_proj_weight(dev)
        mw = moe_weights(dev, rank)
        nw = rand((H,), 3, dev, 0.2)
        ph.warm_up(o_w, o_s, mw)
    finally:
        CompiledFunction.__call__ = launch
    if in_warm_up:
        errors.append(f"rank {rank}: {len(in_warm_up)} launches inside a warm-up")
    # the SDMA form (one instance per DBO microbatch of a 32768-token step) compiles its local kernels when built
    sdma = SdmaSeqParallelHandoffs(rank, dist.group.WORLD, dev, SDMA_TOKENS[-1])
    late0 = warm_up.late_compiles()
    real = MlirCompiler.compile
    compiles = []

    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)

    MlirCompiler.compile = counting
    try:
        for hand, tokens in ((ph, WALK_TOKENS), (sdma, SDMA_TOKENS)):
            for t in tokens:
                rows = shard_rows(t)
                x = rand((rows, H), t + rank, dev)
                codes, scales = hand.input_gather(x)
                attn = rand((t, Q), t + 10 + rank, dev, 0.5)
                moe_in, r, shared = hand.attention_moe(attn, o_w, o_s, x, nw, 1e-6, mw)
                slots, _ = mxfp4_moe(moe_in, mw, return_partials=True)
                codes, scales, r = hand.ffn_norm_gather((slots, shared), r, nw, 1e-6)
                h = hand.ffn_reduce_scatter((slots, shared))
                del slots, shared, moe_in
                codes, scales, r = hand.attention_dense(attn, o_w, o_s, r, nw, 1e-6)
                partial = rand((WORLD * rows, H), t + 20 + rank, dev, 0.25)
                codes, scales, r = hand.ffn_norm_gather(partial, r, nw, 1e-6)
                h = hand.ffn_reduce_scatter(partial)
                hand.gather(h)
                torch.cuda.synchronize()
                if compiles:
                    errors.append(f"rank {rank}: {len(compiles)} compiles at {t} tokens ({type(hand).__name__})")
                    compiles.clear()
                if not bool(torch.isfinite(h).all()):
                    errors.append(f"rank {rank}: non-finite output at {t} tokens ({type(hand).__name__})")
    finally:
        MlirCompiler.compile = real
    if warm_up.late_compiles() != late0:
        errors.append(f"rank {rank}: {warm_up.late_compiles() - late0} late compiles under an entry")
    if ("collectives/seq_parallel_handoffs_h6144_q4096_ranks2", ("moe", True, True)) not in warm_up.warmed():
        errors.append(f"rank {rank}: the MoE hand-off mode was not warmed: {sorted(warm_up.warmed())}")


def _worker_then_reset(rank, errors):
    """``_worker``, then the first-call warm-up off again: later tests reuse these rank processes."""
    from amd_kernels import warm_up

    try:
        _worker(rank, errors)
    finally:
        warm_up.set_auto(False)
        warm_up.set_workers(None)


def test_warm_up_then_no_compile():
    run_on_2_gpus(_worker_then_reset, timeout=2400)
