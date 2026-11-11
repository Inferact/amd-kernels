# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The attention side of a TP4 layer tail at every all-reduce row count (8-4095):
each rank's o_proj partial (bf16 [M, 6144]) + residual -> all-reduce + residual
add + Gemma RMSNorm -> bf16 rows + residual (MoE layers) or MXFP8 codes + scales
+ residual (dense layers), timed on 4 gfx950 GPUs against the baseline sequence
at that row count:

- bf16 (``ar_gemma_rms_norm``): <= 21 even rows AITER's one-stage fused all-reduce
  + Gemma norm; else AITER's custom all-reduce (registered in graphs, copy-in when
  eager) + the one-GPU ``gemma_add_rmsnorm``; above AITER's 64 MB input cap
  (4096-8191-token DBO microbatches take the all-reduce path, not sequence
  parallel) RCCL's all-reduce + the norm, eager.
- MXFP8 (dense MLP layers): AITER's custom all-reduce + the one-GPU norm-quant, codes
  tiled to 16 rows up to 512 rows (``gemma_add_rmsnorm_mxfp8``), row-major above
  (``gemma_add_rmsnorm_mxfp8_row_major``).

Arms: ``aiter`` = AITER's custom all-reduce + norm up to its cap, ``rccl`` above
(eager only); ``aiter_1stage`` = AITER's one-stage fused call; ``entry`` = the entry's
eager-and-graph communicator (``allreduce_add_rmsnorm.staged_comm(...)
.ar_add_rmsnorm[_mxfp8]``) where it serves (``supports(rows, mxfp8)``). Inputs per rank: normal bf16 partials and residuals seeded
``1000*M + 10*i + rank``; up to 2048 rows 100 distinct inputs, one graph of 100
calls; above 16. Rounds of 10 replays, arms alternated; each sample = the max over
the 4 ranks of the per-call time; printed median [min-max] over rounds. Eager
(direct calls, host dispatch included) from 1024 rows, or with ``--eager``.

    python3 benchmarks/norm/gemma_add_rmsnorm_h6144/benchmark_ar_add_rmsnorm_rows.py \\
        [--json out.json] [--rounds 5] [--eager] [--kinds bf16,mxfp8] [--checkpoint <model>] [rows ...]
"""

import argparse
import contextlib
import json
import statistics
import sys

import torch
import torch.distributed as dist

from tests.multi_gpu import WORLD, run_on_4_gpus

EPS = 1e-6
H = 6144
ROWS = [8, 16, 32, 128, 256, 512, 1024, 2048, 4095]
AITER_MAX_BYTES = 8192 * 1024 * 8  # vLLM's AiterCustomAllreduce.effective_max_size()
REPLAYS = 10
TILE = 16


def _norm_weight(layer: int, dev, checkpoint=None) -> torch.Tensor:
    """The checkpoint's post-attention norm weight (layer 39: MoE, 1: dense); random
    without ``--checkpoint``."""
    if checkpoint is None:
        g = torch.Generator().manual_seed(layer)
        return (torch.randn(H, generator=g) * 0.5).bfloat16().to(dev)
    from safetensors import safe_open

    key = f"language_model.model.layers.{layer}.post_attention_layernorm.weight"
    with open(f"{checkpoint}/model.safetensors.index.json") as fh:
        shard = json.load(fh)["weight_map"][key]
    with safe_open(f"{checkpoint}/{shard}", "pt") as f:
        return f.get_tensor(key).to(dev, torch.bfloat16).contiguous()


def _calls(rows: int) -> int:
    return 100 if rows <= 2048 else 16


def _worker(rank, errors, rows_list, kinds, rounds, eager_all, out_json, oneshot_max=None, stream_min=None,
            checkpoint=None):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn

    dev = torch.device("cuda", rank)
    group = dist.group.WORLD
    ca = CustomAllreduce(group=group, device=dev, max_size=8192 * 1024 * 8 * 2)  # vLLM's MAX_SIZE
    assert not ca.disabled
    nccl = dist.new_group(backend="nccl")
    # the entry's eager-and-graph communicator at every row count, once it exists
    factory = getattr(arn, "staged_comm", None)
    tail = factory(rank, group, dev) if callable(factory) else None
    if tail is not None and oneshot_max is not None:  # sweep: move the one-shot tier
        tail.oneshot_max_rows = oneshot_max
        tail.oneshot_max_rows_mxfp8 = oneshot_max
    if tail is not None and stream_min is not None:  # sweep: the streamed kernel below its tier
        tail.register_max_rows = stream_min - 1
        tail.register_max_rows_mxfp8 = stream_min - 1
    weights = {"bf16": _norm_weight(39, dev, checkpoint), "mxfp8": _norm_weight(1, dev, checkpoint)}
    if tail is not None:
        tail.warm_up(weights["bf16"], EPS)
    if rank == 0:
        props = torch.cuda.get_device_properties(dev)
        print(f"rank 0 pci_bus {props.pci_bus_id:#x} entry={tail is not None}", flush=True)
    results = {}
    for rows in rows_list:
        calls = _calls(rows)
        xs, rs = [], []
        for i in range(calls):
            torch.manual_seed(1000 * rows + 10 * i + rank)
            xs.append(torch.randn(rows, H, device=dev).bfloat16())
            rs.append(torch.randn(rows, H, device=dev).bfloat16())
        big = rows * H * 2 > AITER_MAX_BYTES  # served: RCCL's all-reduce (DBO calls up to 8191 rows)
        rins = [x.clone() for x in xs] if big else None  # RCCL reduces in place: its own copies
        tile = TILE if rows <= 512 else 0
        eager = eager_all or rows >= 1024
        for kind in kinds:
            w = weights[kind]
            mx = kind == "mxfp8"

            def norm(y, r):
                if not mx:
                    return an.gemma_add_rmsnorm(y, r, w, EPS)
                if tile:
                    return an.gemma_add_rmsnorm_mxfp8(y, r, w, EPS, tile)
                return an.gemma_add_rmsnorm_mxfp8_row_major(y, r, w, EPS)

            def aiter(i):
                y = ca.custom_all_reduce(xs[i])
                assert y is not None
                return norm(y, rs[i])

            def rccl(i):
                dist.all_reduce(rins[i], group=nccl)  # in place: as fast as vLLM's out-of-place call
                return norm(rins[i], rs[i])

            def aiter_1stage(i):
                return ca.custom_fused_ar_rms(xs[i], rs[i], w, EPS, use_1stage=True, gemma_norm=True)

            def entry(i):
                if mx:
                    return tail.ar_add_rmsnorm_mxfp8(xs[i], rs[i], w, EPS, tile)
                return tail.ar_add_rmsnorm(xs[i], rs[i], w, EPS)

            one_stage = not mx and rows % 2 == 0 and rows * H * 2 < 256 * 1024
            arms = {"rccl": rccl} if big else {"aiter": aiter}
            if one_stage:
                arms["aiter_1stage"] = aiter_1stage
            if tail is not None and tail.supports(rows, mx):
                arms["entry"] = entry
            today = "aiter_1stage" if one_stage else "rccl" if big else "aiter"
            eager_only = {"rccl"}

            # compile everything outside the captures
            norm(xs[0], rs[0])
            if "entry" in arms:
                entry(0)
            torch.cuda.synchronize()
            graphs = {}
            for name, fn in arms.items():
                if name in eager_only:
                    continue
                g = torch.cuda.CUDAGraph()
                with ca.capture(), (tail.capture() if name == "entry" else contextlib.nullcontext()), torch.cuda.graph(g):
                    for i in range(calls):
                        fn(i)
                graphs[name] = g
            for g in graphs.values():
                for _ in range(2):
                    g.replay()
            torch.cuda.synchronize()
            eagers = {f"eager_{k}": fn for k, fn in arms.items() if eager or k in eager_only}
            times = {k: [] for k in [*graphs, *eagers]}
            for rnd in range(rounds):
                order = list(graphs) if rnd % 2 == 0 else list(reversed(list(graphs)))
                for k in order:
                    dist.barrier()
                    torch.cuda.synchronize()
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    for _ in range(REPLAYS):
                        graphs[k].replay()
                    e.record()
                    torch.cuda.synchronize()
                    times[k].append(s.elapsed_time(e) * 1e3 / (REPLAYS * calls))
                order = list(eagers) if rnd % 2 == 0 else list(reversed(list(eagers)))
                for k in order:
                    fn = eagers[k]
                    for i in range(calls):
                        fn(i)
                    torch.cuda.synchronize()
                    dist.barrier()
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    for _ in range(REPLAYS):
                        for i in range(calls):
                            fn(i)
                    e.record()
                    torch.cuda.synchronize()
                    times[k].append(s.elapsed_time(e) * 1e3 / (REPLAYS * calls))
            every = [None] * WORLD
            dist.all_gather_object(every, times)
            if rank == 0:
                samples = {k: [max(ev[k][j] for ev in every) for j in range(rounds)] for k in times}
                parts = [f"rows={rows} {kind} today={today}"]
                for k, v in samples.items():
                    parts.append(f"{k} {statistics.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]")
                for base in (today, f"eager_{today}", "eager_aiter"):
                    other = "entry" if base == today else "eager_entry"
                    if base == "eager_aiter" and f"eager_{today}" in samples:
                        continue
                    if base not in samples:
                        continue
                    if base and other in samples:
                        t, f = samples[base], samples[other]
                        wins = sum(a < b for a, b in zip(f, t))
                        parts.append(
                            f"{other} vs {base}: {statistics.median(t) - statistics.median(f):+.2f} us "
                            f"({wins}/{rounds} rounds)"
                        )
                print(" | ".join(parts), flush=True)
                results[f"{rows}_{kind}"] = {"today": today, "samples": samples, "every": every}
            del graphs
            torch.cuda.synchronize()
        del xs, rs, rins
        torch.cuda.empty_cache()
    if rank == 0 and out_json:
        with open(out_json, "w") as fh:
            json.dump(results, fh, indent=1)
    ca.close()


def main():
    if torch.cuda.device_count() < WORLD:
        sys.exit("needs 4 GPUs")
    p = argparse.ArgumentParser()
    p.add_argument("rows", nargs="*", type=int)
    p.add_argument("--json")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--eager", action="store_true", help="also eager below 1024 rows")
    p.add_argument("--kinds", default="bf16,mxfp8")
    p.add_argument("--oneshot-max", type=int, help="override the staged one-shot's last row count")
    p.add_argument("--stream-min", type=int, help="override: the staged streamed kernel from this row count")
    p.add_argument("--checkpoint", help="MiniMax-M3.1-Flash checkpoint directory: the norm weights of layers 39 and 1 "
                   "(default: random weights)")
    a = p.parse_args()
    run_on_4_gpus(_worker, a.rows or ROWS, a.kinds.split(","), a.rounds, a.eager, a.json, a.oneshot_max,
                  a.stream_min, a.checkpoint, timeout=5400)


if __name__ == "__main__":
    main()
