# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 on 4 ranks, fresh rank processes (this file's first call): the communicators' construction and ``warm_up``
compile every variant, then a decoder layer's phases at the served token counts (4096-32768, off the 32-row grid
too) compile nothing: the input gather, the MoE layer (attention hand-off, ``mxfp4_moe`` on the gathered compact
input, the FFN hand-off on its partials), the dense layer's hand-offs, the final reduce-scatter and the gathers. No
FlyDSL launch happens inside a warm-up."""

import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_4_gpus

from .workload import H, Q, WORLD, moe_weights, o_proj_weight, rand

WALK_TOKENS = (4096, 4097, 6000, 8192, 8193, 8885, 12289, 15745, 16384, 16385, 24577, 31425, 32768)


def _worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4 import SeqParallelHandoffs, shard_rows
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    from flydsl.compiler.jit_function import CompiledFunction, MlirCompiler

    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    warm_up.set_workers(4)  # 4 ranks share the node's CPUs
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
        mw = moe_weights(dev)
        nw = rand((H,), 3, dev, 0.2)
        ph.warm_up(o_w, o_s, mw)
    finally:
        CompiledFunction.__call__ = launch
    if in_warm_up:
        errors.append(f"rank {rank}: {len(in_warm_up)} launches inside a warm-up")
    late0 = warm_up.late_compiles()
    real = MlirCompiler.compile
    compiles = []

    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)

    MlirCompiler.compile = counting
    try:
        for t in WALK_TOKENS:
            rows = shard_rows(t)
            x = rand((rows, H), t + rank, dev)
            codes, scales = ph.input_gather(x)
            attn = rand((t, Q), t + 10 + rank, dev, 0.5)
            moe_in, r = ph.attention_moe(attn, o_w, o_s, x, nw, 1e-6, mw)
            slots, shared = mxfp4_moe(moe_in, mw, return_partials=True)
            codes, scales, r = ph.ffn_norm_gather((slots, shared), r, nw, 1e-6)
            h = ph.ffn_reduce_scatter((slots, shared))
            del slots, shared, moe_in
            codes, scales, r = ph.attention_dense(attn, o_w, o_s, r, nw, 1e-6)
            partial = rand((WORLD * rows, H), t + 20 + rank, dev, 0.25)
            codes, scales, r = ph.ffn_norm_gather(partial, r, nw, 1e-6)
            h = ph.ffn_reduce_scatter(partial)
            ph.gather(h)
            torch.cuda.synchronize()
            if compiles:
                errors.append(f"rank {rank}: {len(compiles)} compiles at {t} tokens after the warm-up")
                compiles.clear()
            if not bool(torch.isfinite(h).all()):
                errors.append(f"rank {rank}: non-finite output at {t} tokens")
    finally:
        MlirCompiler.compile = real
    if warm_up.late_compiles() != late0:
        errors.append(f"rank {rank}: {warm_up.late_compiles() - late0} late compiles under an entry")
    if ("collectives/seq_parallel_handoffs_h6144_q2048_ranks4", ("moe", True, True)) not in warm_up.warmed():
        errors.append(f"rank {rank}: the MoE hand-off mode was not warmed: {sorted(warm_up.warmed())}")


def test_warm_up_then_no_compile():
    run_on_4_gpus(_worker, timeout=2400)
