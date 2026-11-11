# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compare the public QKV/insert chain against a previous insert package.

Layer 39 rank 0, >400 MB of rotating weights, 100 seeded inputs through M2048
and 16 above, three repetitions of five alternating rounds, ten graph replays.
Keep the baseline module and its compiled launchers alive until graphs die.

  python3 benchmarks/gemm/mxfp8_n2560_k6144/benchmark_qkv_insert_stores.py \
    --checkpoint /path/to/model \
    --baseline /path/to/old/amd_kernels/attention/norm_rope_nvfp4_insert_d128_rot64_p128
"""

import argparse
import gc
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

import benchmark_qkv_insert as bench
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import host
from amd_kernels.gemm.mxfp8_common import gemm


def load_baseline(path):
    spec = importlib.util.spec_from_file_location(
        "qkv_insert_baseline", path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def eager(call, count, phases):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for phase in range(phases):
        for j in range(count):
            call(j, phase * count)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (count * phases)


def stats(samples):
    return dict(median=statistics.median(samples), min=min(samples), max=max(samples), rounds=samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--ms", default="289,320,384,512,513,768,1024,1536,2048,3072,4095,24576,32768")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--insert-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    old = load_baseline(args.baseline)
    copies, norm = bench.weights(args.checkpoint, args.layer)
    kw = dict(num_heads=16, num_kv_heads=1, num_index_heads=1, eps=1e-6)
    for m in map(int, args.ms.split(",")):
        count = 100 if m <= 2048 else 16
        phases = (len(copies) + count - 1) // count
        acts, insert_args = bench.inputs(m, count, norm)
        intermediates = []
        if args.insert_only:
            for j, (codes, scales, tile) in enumerate(acts):
                w, ws = copies[j % len(copies)]
                if tile:
                    qkv = gemm.mxfp8_gemm_shuffled(codes, scales, w, ws, torch.bfloat16, m)
                else:
                    qkv = gemm.mxfp8_gemm(codes, scales, w, ws, torch.bfloat16,
                                          activation_scale_preshuffled=True)
                intermediates.append(qkv)
        calls = {}
        addresses = {"baseline": set(), "ours": set()}
        recording = False
        for name, insert in [("baseline", old.norm_rope_nvfp4_insert),
                             ("ours", host.norm_rope_nvfp4_insert)]:
            def call(j, offset, insert=insert, arm=name):
                codes, scales, tile = acts[j]
                w, ws = copies[(j + offset) % len(copies)]
                if args.insert_only:
                    qkv = intermediates[j]
                elif tile:
                    qkv = gemm.mxfp8_gemm_shuffled(codes, scales, w, ws, torch.bfloat16, m)
                else:
                    qkv = gemm.mxfp8_gemm(codes, scales, w, ws, torch.bfloat16,
                                          activation_scale_preshuffled=True)
                if recording:
                    addresses[arm].add(qkv.data_ptr())
                insert(qkv, *insert_args[j], **kw)
            calls[name] = call
        # All modified outputs, including complete pages, must have the same bits.
        calls["baseline"](0, 0)
        reference = [insert_args[0][i].clone() for i in (12, 13, 8, 9)]
        calls["ours"](0, 0)
        same = all(torch.equal(x.view(torch.uint8), insert_args[0][i].view(torch.uint8))
                   for x, i in zip(reference, (12, 13, 8, 9)))
        assert same
        graphs = {}
        # Both arms release the ephemeral GEMM output per call. Sharing their
        # graph pool keeps its address/cache color equal across the comparison.
        pool = torch.cuda.graph_pool_handle()
        for name, call in calls.items():
            arm = []
            for phase in range(phases):
                call(0, phase * count)
                graph = torch.cuda.CUDAGraph()
                recording = True
                with torch.cuda.graph(graph, pool=pool):
                    for j in range(count):
                        call(j, phase * count)
                recording = False
                arm.append(graph)
            graphs[name] = arm
        assert addresses["baseline"] == addresses["ours"], addresses
        for _ in range(10):
            for arm in graphs.values():
                for graph in arm:
                    graph.replay()
        torch.cuda.synchronize()
        if m >= 2048:
            for call in calls.values():
                eager(call, count, phases)
        gc.collect()
        samples = {name: [] for name in calls}
        eager_samples = {name: [] for name in calls}
        for _ in range(15):
            for name, arm in graphs.items():
                samples[name].append(bench.measure(arm, count))
            if m >= 2048:
                for name, call in calls.items():
                    eager_samples[name].append(eager(call, count, phases))
        result = dict(m=m, layer=args.layer, unit="insert" if args.insert_only else "chain", same_bytes=same, same_intermediate_addresses=True,
                      graph={name: stats(v) for name, v in samples.items()})
        if m >= 2048:
            result["eager"] = {name: stats(v) for name, v in eager_samples.items()}
        print(json.dumps(result), flush=True)
        if args.profile:
            from torch.profiler import profile, ProfilerActivity
            for name, arm in graphs.items():
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    arm[0].replay()
                    torch.cuda.synchronize()
                kernels = {ev.key: ev.count for ev in prof.key_averages()
                           if str(ev.device_type).endswith("CUDA")}
                print(json.dumps(dict(m=m, arm=name, kernels=kernels)), flush=True)
        del graphs, calls, acts, insert_args, reference, intermediates
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
