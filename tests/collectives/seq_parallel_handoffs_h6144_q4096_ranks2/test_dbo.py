# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DBO as vLLM runs it on 2 ranks: two microbatches, one ``SdmaSeqParallelHandoffs`` each, two threads taking turns on
one compute stream (a phase's ``between`` is the yield, so one microbatch's pushes are in flight while the other's
kernels are queued; each microbatch's full shared expert runs on its instance's side stream across the yield), two
MoE layers then the final reduce-scatter and gathers; every output bit for bit the pull form's
(``SeqParallelHandoffs``) on the same microbatch."""

import threading

import pytest
import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_2_gpus

from .workload import H, Q, moe_weights, o_proj_weight, rand

EPS = 1e-6


def _layers(ph, attn, res, o_w, o_s, nw, mw, between=None):
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import mxfp4_moe

    kw = {} if between is None else {"between": between}
    outs = []
    for _ in range(2):
        moe_in, res, shared = ph.attention_moe(attn, o_w, o_s, res, nw, EPS, mw, **kw)
        slots, _ = mxfp4_moe(moe_in, mw, return_partials=True)
        outs += [t.clone() for t in moe_in.tensors()[:4]]
        codes, scales, res = ph.ffn_norm_gather((slots, shared), res, nw, EPS, **kw)
        outs += [codes.clone(), scales.clone(), res]
    h = ph.ffn_reduce_scatter((slots, shared), **kw)
    outs += [h, ph.gather(h, **kw), ph.gather(res, **kw)]
    return outs


def _worker(rank, errors, tokens):
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import (SdmaSeqParallelHandoffs,
                                                                               SeqParallelHandoffs, shard_rows)

    dev = torch.device("cuda", rank)
    o_w, o_s = o_proj_weight(dev)
    mw = moe_weights(dev, rank)
    nw = rand((H,), 3, dev, 0.2)
    u = [tokens // 2, tokens - tokens // 2]
    attns = [rand((n, Q), 10 * i + rank, dev, 0.5) for i, n in enumerate(u)]
    ress = [rand((shard_rows(n), H), 10 * i + 5 + rank, dev) for i, n in enumerate(u)]
    pull = SeqParallelHandoffs(rank, dist.group.WORLD, dev, max(u))
    want = [_layers(pull, attns[i], ress[i], o_w, o_s, nw, mw) for i in range(2)]
    torch.cuda.synchronize()

    sdma = [SdmaSeqParallelHandoffs(rank, dist.group.WORLD, dev, max(u)) for _ in range(2)]
    stream = torch.cuda.Stream(device=dev)
    turn = [threading.Event(), threading.Event()]
    done = [False, False]
    got, errs = [None, None], []

    def body(i):
        torch.cuda.set_device(dev)
        torch.cuda.set_stream(stream)

        def yield_():
            if not done[1 - i]:
                turn[1 - i].set()
                turn[i].wait()
                turn[i].clear()

        try:
            turn[i].wait()
            turn[i].clear()
            got[i] = _layers(sdma[i], attns[i], ress[i], o_w, o_s, nw, mw, between=yield_)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
        finally:
            done[i] = True
            turn[1 - i].set()

    stream.wait_stream(torch.cuda.current_stream())
    threads = [threading.Thread(target=body, args=(i,)) for i in range(2)]
    for th in threads:
        th.start()
    turn[0].set()
    for th in threads:
        th.join()
    torch.cuda.synchronize()
    for ph in sdma:
        ph.drain()
    errors.extend(f"rank {rank}: {e}" for e in errs)
    for i in range(2):
        if got[i] is None:
            continue
        for k, (a, b) in enumerate(zip(got[i], want[i])):
            if a.shape != b.shape or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                errors.append(f"rank {rank} microbatch {i} output {k} differs")


@pytest.mark.parametrize("tokens", [8192, 32768 + 1])
def test_dbo_sdma_equals_pull(tokens):
    run_on_2_gpus(_worker, tokens, timeout=900)
