# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""M97-128 o_proj: immutable-A/scale LDS versus the preceding direct 64x48 row.

Layer39 rank0, 32 cold weight copies (>400 MB), 100 seeded inputs, shared graph
pool with matched temporary addresses, 3x5 alternated rounds of ten replays.
"""
import argparse
import dataclasses
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from amd_kernels.gemm.mxfp8_common import gemm as g


def measure(graph):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop)  # 100 calls x 10 replays: ms == us / call


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--ms", default="97,111,113,127,128")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"language_model.model.layers.{args.layer}.self_attn.o_proj."
    raw = []
    for field, columns in (("weight", 2048), ("weight_scale", 64)):
        with safe_open(str(args.checkpoint / index[prefix + field]), framework="pt") as f:
            raw.append(f.get_tensor(prefix + field)[:, :columns].contiguous().cuda())
    w, ws = g.preshuffle_mxfp8_weight(raw[0].view(torch.float8_e4m3fn), raw[1].view(torch.uint8))
    copies = [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(31)]
    pool = torch.cuda.graph_pool_handle()
    results = {}
    for m in map(int, args.ms.split(",")):
        assert 97 <= m <= 128
        cfg = g._select_prequant_config(m, 6144, 2048, torch.bfloat16)
        # An explicit lookahead selects the retained direct kernel with exactly
        # the preceding row's automatic 2x8 lookahead.
        old = dataclasses.replace(cfg, la_w=2, la_a=8)
        acts = []
        for i in range(100):
            x = torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal(
                (m, 2048), dtype=np.float32)).cuda().bfloat16()
            acts.append(g.mxfp8_e4m3_quantize_shuffled(x, 64))
        pointers = {arm: set() for arm in ("old", "new")}

        def call(j, arm):
            y = g.mxfp8_gemm_shuffled(
                *acts[j], *copies[j % 32], torch.bfloat16, m,
                config=old if arm == "old" else None,
            )
            if torch.cuda.is_current_stream_capturing():
                pointers[arm].add(y.data_ptr())
            return y

        for j in range(100):
            assert torch.equal(call(j, "old"), call(j, "new")), (m, j)
        graphs = {}
        for arm in pointers:
            call(0, arm)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                for j in range(100):
                    call(j, arm)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            graphs[arm] = graph
        assert pointers["old"] == pointers["new"], pointers
        samples = {arm: [] for arm in graphs}
        for _ in range(3):
            for _ in range(5):
                for arm, graph in graphs.items():
                    samples[arm].append(measure(graph))
        med = {arm: statistics.median(v) for arm, v in samples.items()}
        result = {
            "median": med,
            "range": {arm: [min(v), max(v)] for arm, v in samples.items()},
            "change_pct": 100 * (med["new"] / med["old"] - 1),
            "every_round_faster": all(a < b for a, b in zip(samples["new"], samples["old"])),
            "apart": max(samples["new"]) < min(samples["old"]),
            "rounds": samples,
        }
        results[m] = result
        print(m, json.dumps(result), flush=True)
        del graphs, acts
        torch.cuda.empty_cache()
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
