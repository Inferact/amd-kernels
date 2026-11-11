# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SP o_proj_full (N 6144 x K 8192): the table's row versus the 4-wave row, preshuffled input scales (the
sequence-parallel call). 417-1024 rows: the split-2 256x192 tile vs 4-wave 128x256; ``--previous-prefill``, 1281-2048
rows: the 256x192 tiles vs 4-wave 256x256.

Layer 39 full o_proj; 8 cold weight copies (>400 MB); 100 seeded inputs;
3 x 5 alternated rounds of 10 graph replays, plus eager entry timing.
"""
import sys
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from amd_kernels.gemm.mxfp8_common import gemm as g
from benchmark_rect_n2560_k6144 import capture, eager
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mxfp8_n2560_k6144"))  # benchmark_qkv_insert's helpers
from benchmark_qkv_insert import measure  # noqa: E402

OLD = g._Config(128, 256, 128, 1, 1, 1, "4wave", 2, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--ms", default="417,768,897,1024")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--previous-prefill", action="store_true", help="Compare the previous 256x256 row at 1281-2048.")
    args = ap.parse_args()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"language_model.model.layers.{args.layer}.self_attn.o_proj."
    raw = []
    for field in ("weight", "weight_scale"):
        with safe_open(str(args.checkpoint / index[prefix + field]), framework="pt") as f:
            raw.append(f.get_tensor(prefix + field).cuda())
    w, ws = g.preshuffle_mxfp8_weight(raw[0].view(torch.float8_e4m3fn), raw[1].view(torch.uint8))
    copies = [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(7)]
    result = {}
    for m in map(int, args.ms.split(",")):
        assert (1281 <= m <= 2048) if args.previous_prefill else (416 < m <= 1024)
        old_config = g._Config(256, 256, 128, 1, 1, 1, "4wave_nt", 2, 2) if args.previous_prefill else OLD
        rows = [torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal(
            (m, 8192), dtype=np.float32)).cuda().bfloat16() for i in range(100)]
        acts = [g.mxfp8_e4m3_quantize_aiter(x) for x in rows]
        ptrs = {'old': set(), 'new': set()}
        def project(j, config):
            out = g.mxfp8_gemm(*acts[j], *copies[j % 8], torch.bfloat16, activation_scale_preshuffled=True,
                               config=config)
            if torch.cuda.is_current_stream_capturing():
                ptrs['new' if config is None else 'old'].add(out.data_ptr())
            return out
        same = torch.equal(project(0, old_config), project(0, None))
        calls = {"old": lambda j: project(j, old_config), "new": lambda j: project(j, None)}
        pool = torch.cuda.graph_pool_handle()
        graphs = {arm: capture(fn, pool=pool) for arm, fn in calls.items()}
        assert ptrs["old"] == ptrs["new"]
        samples = {kind: {arm: [] for arm in calls} for kind in ("graph", "eager")}
        for _ in range(3):
            for _ in range(5):
                for arm in calls:
                    samples["graph"][arm].append(measure([graphs[arm]], 100))
                for arm, fn in calls.items():
                    samples["eager"][arm].append(eager(fn))
        summary = {"same_bf16": same}
        for kind, rounds in samples.items():
            summary[kind] = {
                "median": {arm: statistics.median(v) for arm, v in rounds.items()},
                "range": {arm: [min(v), max(v)] for arm, v in rounds.items()},
                "every_round_faster": all(a < b for a, b in zip(rounds["new"], rounds["old"])),
                "apart": max(rounds["new"]) < min(rounds["old"]), "rounds": rounds,
            }
        result[m] = summary
        print(m, json.dumps(summary), flush=True)
        del graphs, acts, rows
        torch.cuda.empty_cache()
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
