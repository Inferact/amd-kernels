# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""A MoE decoder layer of the TP2 prefill on two ranks, sequence-parallel vs the plain tensor-parallel path, per token
count T (shard rows R = T / 2). The layer from the attention output to the next layer's QKV output; the attention
core (identical in both paths) is not run.

- ``sp`` (``SeqParallelHandoffs``, pull collectives) and ``sdma`` (``SdmaSeqParallelHandoffs``, one batch, no
  overlap): ``attention_moe`` (pack, all-to-all unpack, o_proj_full [6144, 8192] on R rows, norm, MoE preparation,
  the routed payload's gather, the full shared expert of the R own rows on the side stream), ``mxfp4_moe`` routed
  partials on all T rows, ``ffn_norm_gather`` (join, MoE tail reduce-scatter with the owner's shared output + norm +
  quant, codes gather), the QKV GEMM [4992, 6144] on all T rows.
- ``sdma_dbo``: DBO as vLLM runs it: two microbatches of T / 2 tokens (one ``SdmaSeqParallelHandoffs`` each, each
  with its shared-expert side stream), two threads taking turns on one compute stream, each phase's ``between`` the
  yield, so a microbatch's pushes run on the SDMA engine under the other microbatch's compute; ``DBO_LAYERS`` layers
  per round, us per layer.
- ``*_serial``: the same with the full shared expert on the compute stream (``shared_stream=False``): the overlap's
  reference. ``*_shard``: a shard copy of the shared expert (I 1536 on all T rows inside ``mxfp4_moe``, the compact
  7328 B/token gather; under DBO on the caller's stream, as ``set_dbo_query`` puts it in vLLM).
- ``tp_rccl`` / ``tp_aiter``: the attention output quantized + o_proj [6144, 4096] on all T rows, all-reduce (RCCL, or
  AITER's custom all-reduce where vLLM takes it: at most 64 MiB), the bf16 add + norm, ``mxfp4_moe`` (reduced) on all
  T rows, all-reduce, the add + norm + MXFP8 quant, the QKV GEMM. The same amd_kernels compute in both paths, so the
  difference is the hand-offs.

Eager (the SP phases keep host sequence state), ``ROUNDS`` rounds after a warm-up call, each bracketed by a barrier:
the max over ranks per round, the median over rounds, us. Stage columns: the SP hand-off phases alone.

Roofline: link bytes per rank at ``LINK_GBS`` (SP: R x 23456 B: a2a 4224, routed gather 3296 (shard copy: compact
7328), coded tail 9600, codes gather 6336; TP: two all-reduces of [T, H] bf16, R x 49152 B) and the compute ideal (each GEMM / MoE at max(HBM
bytes / 6.5 TB/s, MXFP4 flops / 10 PF + MXFP8 flops / 5 PF), the norms at their bytes): their sum for one batch,
their maximum for DBO.

  python3 benchmarks/collectives/seq_parallel_handoffs_h6144_q4096_ranks2/benchmark_sp_layer.py [out.json] \
      [--tokens T ...] [--paths sp sdma tp_rccl tp_aiter] [--bf16-tail]

``--bf16-tail``: the SP hand-offs with the MoE tail's reduce-scatter in BF16 (12288 B per row) instead of the code.
"""
import json
import statistics
import sys
import threading

import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_2_gpus

H, Q, WORLD, I, E = 6144, 4096, 2, 1536, 128
QKV_N = 4992
EPS = 1e-6
ROUNDS = 7
LINK_GBS = 61.2
HBM_TBS = 6.5
FP4_PF, FP8_PF = 10.0, 5.0
AITER_MAX_BYTES = 64 << 20  # vLLM's AiterCustomAllreduce.effective_max_size
TOKENS = (4096, 8192, 16384, 32768)
PATHS = ("sp", "sp_serial", "sp_shard", "sdma", "sdma_dbo", "sdma_dbo_serial", "sdma_dbo_shard", "tp_rccl",
         "tp_aiter")
DBO_LAYERS = 4
SP_LINK_BYTES = 4224 + 3296 + 9600 + 6336  # per shard row (12288 for the tail with --bf16-tail)
SHARD_GATHER_EXTRA = 7328 - 3296  # the compact payload of a shard copy


def _compute_ideal_us(t: int) -> float:
    """The layer's compute ideal at T tokens (module docstring)."""
    def gemm(m, n, k, pf):
        return max((n * k + m * k + 2 * m * n) / (HBM_TBS * 1e12), 2 * m * n * k / (pf * 1e15))

    touched = E * (1 - (1 - 4 / E) ** t)
    moe_bytes = touched * 15.04e6 + 58.4e6 + 3.1e6 + t * H * 4
    moe_flops = (t * 4 * 2 * (H * 2 * I + I * H)) / (FP4_PF * 1e15) + (t * 2 * (H * 2 * I + I * H)) / (FP8_PF * 1e15)
    moe = max(moe_bytes / (HBM_TBS * 1e12), moe_flops)
    o_proj = gemm(t // WORLD, H, WORLD * Q, FP8_PF)  # = o_proj [6144, 4096] on T rows in flops
    qkv = gemm(t, QKV_N, H, FP8_PF)
    norms = 2 * (t // WORLD) * H * 2 * 3 / (HBM_TBS * 1e12)
    return (moe + o_proj + qkv + norms) * 1e6


def _weights(dev, rank):
    from amd_kernels.gemm.mxfp8_common.gemm import preshuffle_mxfp8_weight
    from tests.collectives.seq_parallel_handoffs_h6144_q4096_ranks2.workload import moe_weights, o_proj_weight

    def fp8(n, k, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        wq = (torch.randn((n, k), device=dev, generator=g) * 20).to(torch.float8_e4m3fn)
        ws = torch.randint(116, 121, (n, k // 32), generator=g, device=dev, dtype=torch.uint8)
        return preshuffle_mxfp8_weight(wq, ws)

    return o_proj_weight(dev), fp8(H, Q, 11), fp8(QKV_N, H, 12), moe_weights(dev, rank)


def _shard_copy(dev, mw):
    """``mw`` with a shard copy of the shared expert (the same routed tensors)."""
    from dataclasses import replace

    from tests.collectives.seq_parallel_handoffs_h6144_q4096_ranks2.workload import moe_weights

    return replace(mw, shared_weights=moe_weights(dev, full=False).shared_weights)


def _worker(rank, errors, tokens, paths, out_path, coded=True):
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import (SdmaSeqParallelHandoffs,
                                                                                SeqParallelHandoffs, shard_rows)
    from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_e4m3_quantize_flydsl, mxfp8_gemm
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import mxfp4_moe, set_dbo_query
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm_mxfp8_row_major
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.host import gemma_add_rmsnorm
    from tests.collectives.seq_parallel_handoffs_h6144_q4096_ranks2.workload import rand

    dev = torch.device("cuda", rank)
    (o_w, o_s), (tp_w, tp_s), (qkv_w, qkv_s), mw_full = _weights(dev, rank)
    mw_shard = _shard_copy(dev, mw_full) if any(p.endswith("_shard") or p.startswith("tp") for p in paths) else None
    nw, nw2 = rand((H,), 3, dev, 0.2), rand((H,), 4, dev, 0.2)
    nccl = dist.new_group(backend="nccl")
    in_dbo = threading.local()
    set_dbo_query(lambda: getattr(in_dbo, "on", False))
    hand, dbo = {}, {}
    for path in ("sp", "sp_serial", "sp_shard"):
        if path in paths:
            hand[path] = SeqParallelHandoffs(rank, dist.group.WORLD, dev, max(tokens), coded_tail=coded,
                                             shared_stream=path != "sp_serial")
    if "sdma" in paths:
        hand["sdma"] = SdmaSeqParallelHandoffs(rank, dist.group.WORLD, dev, max(tokens), coded_tail=coded)
    for path in ("sdma_dbo", "sdma_dbo_serial", "sdma_dbo_shard"):
        if path in paths:
            ub = max(tokens) // 2 + 1
            dbo[path] = [SdmaSeqParallelHandoffs(rank, dist.group.WORLD, dev, ub, coded_tail=coded,
                                                 compact=path.endswith("_shard"),
                                                 shared_stream=not path.endswith("_serial")) for _ in range(2)]
    aiter = None
    if "tp_aiter" in paths:
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

        aiter = CustomAllreduce(dist.group.WORLD, dev, max_size=2 * AITER_MAX_BYTES)

    def qkv(codes, scales):
        return mxfp8_gemm(codes, scales, qkv_w, qkv_s, torch.bfloat16, activation_scale_preshuffled=True)

    def moe(ph, attn, res, mw, **kw):
        """attention_moe + the routed (and, with a shard copy, shared) partials -> (FFN output, residual)."""
        moe_in, r, shared = ph.attention_moe(attn, o_w, o_s, res, nw, EPS, mw, **kw)
        slots, sh = mxfp4_moe(moe_in, mw, return_partials=True)
        return (slots, shared if shared is not None else sh), r

    def sp_layer(ph, attn, res, ev, mw):
        ev[0].record()
        moe_in, r, shared = ph.attention_moe(attn, o_w, o_s, res, nw, EPS, mw)
        ev[1].record()
        slots, sh = mxfp4_moe(moe_in, mw, return_partials=True)
        del moe_in
        ev[2].record()
        codes, scales, r = ph.ffn_norm_gather((slots, shared if shared is not None else sh), r, nw2, EPS)
        ev[3].record()
        del slots, sh, shared
        y = qkv(codes, scales)
        ev[4].record()
        return y

    def dbo_layers(dbo, attns, ress, ev, mw):
        """Two microbatches through ``DBO_LAYERS`` layers, vLLM's DBO turn-taking (module docstring)."""
        stream = torch.cuda.current_stream()
        turn = [threading.Event(), threading.Event()]
        done = [False, False]
        errs = []

        def body(i):
            torch.cuda.set_device(dev)
            torch.cuda.set_stream(stream)

            def yield_():
                if not done[1 - i]:
                    turn[1 - i].set()
                    turn[i].wait()
                    turn[i].clear()

            try:
                in_dbo.on = True
                turn[i].wait()
                turn[i].clear()
                ph, attn, res = dbo[i], attns[i], ress[i]
                for _ in range(DBO_LAYERS):
                    ffn, r = moe(ph, attn, res, mw, between=yield_)
                    codes, scales, r = ph.ffn_norm_gather(ffn, r, nw2, EPS, between=yield_)
                    del ffn
                    qkv(codes, scales)
                    yield_()
            except Exception as e:  # noqa: BLE001
                errs.append(e)
            finally:
                done[i] = True
                turn[1 - i].set()

        ev[0].record()
        threads = [threading.Thread(target=body, args=(i,)) for i in range(2)]
        for th in threads:
            th.start()
        turn[0].set()
        for th in threads:
            th.join()
        if errs:
            raise errs[0]
        for e in ev[1:]:
            e.record()

    def tp_layer(all_reduce, attn, res_full, ev):
        ev[0].record()
        xq, xs = mxfp8_e4m3_quantize_flydsl(attn)
        h = mxfp8_gemm(xq, xs, tp_w, tp_s, torch.bfloat16, activation_scale_preshuffled=True)
        h = all_reduce(h)
        y, r = gemma_add_rmsnorm(h, res_full, nw, EPS)
        ev[1].record()
        out = mxfp4_moe(y, mw_shard)
        ev[2].record()
        out = all_reduce(out)
        codes, scales, r = gemma_add_rmsnorm_mxfp8_row_major(out, r, nw2, EPS)
        ev[3].record()
        y = qkv(codes, scales)
        ev[4].record()
        return y

    def rccl(x):
        dist.all_reduce(x, group=nccl)
        return x

    def aiter_ar(x):
        return aiter.custom_all_reduce(x)

    results = []
    for t in tokens:
        rows = shard_rows(t)
        attn = rand((t, Q), t + 10 + rank, dev, 0.5)
        res = rand((rows, H), t + 20 + rank, dev)
        res_full = rand((t, H), t + 30, dev)
        for path in paths:
            if path == "tp_aiter" and t * H * 2 > AITER_MAX_BYTES:
                continue
            mw = mw_shard if path.endswith("_shard") else mw_full
            if path in dbo:
                u = [t // 2, t - t // 2]
                attns = [attn[: u[0]].contiguous(), attn[u[0] :].contiguous()]
                ress = [rand((shard_rows(n), H), t + 40 + i + rank, dev) for i, n in enumerate(u)]
                run = lambda ev, d=dbo[path], mw=mw: dbo_layers(d, attns, ress, ev, mw)  # noqa: E731
            elif path in hand:
                if not hand[path].supports(t):
                    continue
                run = lambda ev, ph=hand[path], mw=mw: sp_layer(ph, attn, res, ev, mw)  # noqa: E731
            else:
                ar = rccl if path == "tp_rccl" else aiter_ar
                run = lambda ev, ar=ar: tp_layer(ar, attn, res_full, ev)  # noqa: E731
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            run(ev)
            torch.cuda.synchronize()
            rounds = []
            for _ in range(ROUNDS):
                dist.barrier()
                torch.cuda.synchronize()
                run(ev)
                torch.cuda.synchronize()
                rounds.append([ev[0].elapsed_time(e) * 1000 for e in ev[1:]])
            if path == "sdma":
                hand[path].drain()
            if path in dbo:
                for ph in dbo[path]:
                    ph.drain()
                rounds = [[x / DBO_LAYERS for x in r] for r in rounds]
            every = [None] * WORLD
            dist.all_gather_object(every, rounds)
            per_round = [[max(every[p][i][j] for p in range(WORLD)) for j in range(4)] for i in range(ROUNDS)]
            total = statistics.median(r[3] for r in per_round)
            stage = [statistics.median(r[j] - (r[j - 1] if j else 0) for r in per_round) for j in range(4)]
            link_bytes = rows * (4 * H * 2 if path.startswith("tp") else SP_LINK_BYTES + (0 if coded else 2688)
                                 + (SHARD_GATHER_EXTRA if path.endswith("_shard") else 0))
            link_us = link_bytes / (LINK_GBS * 1e3)
            comp_us = _compute_ideal_us(t)
            # one batch serializes the link and the compute; DBO can overlap them
            ideal = max(link_us, comp_us) if path in dbo else link_us + comp_us
            if path in dbo:
                stage = None
            results.append(dict(tokens=t, path=path, total_us=total, stage_us=stage, link_bytes=link_bytes,
                                link_us=link_us, compute_ideal_us=comp_us, ideal_us=ideal))
            if rank == 0:
                stages = " / ".join(f"{s:.1f}" for s in stage) if stage else "-"
                print(f"T {t:6d} {path:9s} total {total:9.1f} us  stages {stages}  ideal {ideal:8.1f} "
                      f"(link {link_us:.1f}, compute {comp_us:.1f})", flush=True)
        del attn, res, res_full
        torch.cuda.empty_cache()
    if rank == 0 and out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)


def main():
    args = sys.argv[1:]
    out = args.pop(0) if args and args[0].endswith(".json") else None
    tokens, paths, key = [], [], None
    for a in args:
        if a.startswith("--"):
            key = a
        elif key == "--tokens":
            tokens.append(int(a))
        elif key == "--paths":
            paths.append(a)
    run_on_2_gpus(_worker, tuple(tokens) or TOKENS, tuple(paths) or PATHS, out, "--bf16-tail" not in args,
                  timeout=3000)


if __name__ == "__main__":
    main()
