# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""o_proj into a caller arena: allocate + copy versus the public out= entry.

This measures the producer and an explicit staging copy, not an all-reduce tail.
Layer39 TP4 rank0, 32 rotating weight copies (>400 MB), 100 seeded inputs,
15 alternating rounds of 10 graph replays plus eager calls. Both arms write
into the same arena. Use --bf16-input to include activation quantization.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--layer', type=int, default=39)
    ap.add_argument('--ms', default='1024,1280,1536,2048,2560,3072,4095')
    ap.add_argument('--bf16-input', action='store_true')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'language_model.model.layers.{args.layer}.self_attn.o_proj.'
    raw = []
    for field, cols in [('weight', 2048), ('weight_scale', 64)]:
        with safe_open(str(args.checkpoint / index[prefix + field]), framework='pt') as f:
            raw.append(f.get_tensor(prefix + field)[:, :cols].contiguous().cuda())
    w, ws = g.preshuffle_mxfp8_weight(raw[0].view(torch.float8_e4m3fn), raw[1].view(torch.uint8))
    copies = [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(31)]
    result = {}
    for m in map(int, args.ms.split(',')):
        assert 1024 <= m <= 4095
        rows = [torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal(
            (m, 2048), dtype=np.float32)).cuda().bfloat16() for i in range(100)]
        acts = [g.mxfp8_e4m3_quantize_aiter(x) for x in rows]
        arena = torch.empty((m, 6144), dtype=torch.bfloat16, device='cuda')

        def project(j, direct):
            dst = arena if direct else None
            if args.bf16_input:
                y = g.mxfp8_linear(rows[j], *copies[j % 32], out=dst)
            else:
                y = g.mxfp8_gemm(*acts[j], *copies[j % 32], torch.bfloat16,
                                 activation_scale_preshuffled=True, out=dst)
            if direct:
                assert y is arena
            else:
                arena.copy_(y)
            return arena

        for j in range(100):
            expected = project(j, False).clone()
            arena.fill_(float('nan'))
            assert torch.equal(project(j, True), expected), (m, j)
        calls = {'copy': lambda j: project(j, False), 'direct': lambda j: project(j, True)}
        pool = torch.cuda.graph_pool_handle()
        graphs = {arm: capture(fn, pool=pool) for arm, fn in calls.items()}
        for _ in range(5):
            for graph in graphs.values():
                graph.replay()
        samples = {kind: {arm: [] for arm in calls} for kind in ('graph', 'eager')}
        for r in range(15):
            order = ('copy', 'direct') if r % 2 == 0 else ('direct', 'copy')
            for arm in order:
                samples['graph'][arm].append(measure([graphs[arm]], 100))
            for arm in order:
                samples['eager'][arm].append(eager(calls[arm]))
        summary = {}
        for kind, rounds in samples.items():
            summary[kind] = dict(
                median={arm: statistics.median(v) for arm, v in rounds.items()},
                ranges={arm: [min(v), max(v)] for arm, v in rounds.items()},
                every_round_faster=all(a < b for a,b in zip(rounds['direct'], rounds['copy'])),
                apart=max(rounds['direct']) < min(rounds['copy']), rounds=rounds)
        result[m] = summary
        print(m, json.dumps(summary), flush=True)
        del graphs, acts, rows, arena
        torch.cuda.empty_cache()
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
