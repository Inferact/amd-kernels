# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Cold-weight o_proj: public entry versus its previous row, N6144 K2048, 769-1024 rows.

Real TP4 rank-0 weights, 32 copies (>400 MB), 100 seeded prequantized inputs,
matched ephemeral output addresses, 15 alternating rounds x 10 graph replays,
and eager calls. The previous row is the 4-wave 256x128 tile (non-temporal stores). Timing uses public
mxfp8_gemm calls in both arms.
"""
import sys
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from benchmark_rect_n2560_k6144 import capture, eager
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mxfp8_n2560_k6144"))  # benchmark_qkv_insert's helpers
from benchmark_qkv_insert import measure  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--layer', type=int, default=39)
    ap.add_argument('--ms', default='769,800,832,896,897,960,1007,1024')
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
    results = {}
    for m in map(int, args.ms.split(',')):
        assert 769 <= m <= 1024
        acts = []
        for i in range(100):
            x = torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal(
                (m, 2048), dtype=np.float32)).cuda().bfloat16()
            acts.append(g.mxfp8_e4m3_quantize_aiter(x))
        previous = g._Config(256, 128, 128, 1, 1, 1, g._FOURWAVE_NT_KERNEL, 2, 2)
        pointers = {'old': set(), 'new': set()}

        def call(j, arm):
            out = g.mxfp8_gemm(*acts[j], *copies[j % 32], torch.bfloat16,
                               activation_scale_preshuffled=True,
                               config=previous if arm == 'old' else None)
            if torch.cuda.is_current_stream_capturing():
                pointers[arm].add(out.data_ptr())
            return out

        for j in range(100):
            assert ref.compare(call(j, 'old'), call(j, 'new'))['cos'] >= 0.99999, (m, j)
        calls = {arm: (lambda j, arm=arm: call(j, arm)) for arm in pointers}
        pool = torch.cuda.graph_pool_handle()
        graphs = {arm: capture(fn, pool=pool) for arm, fn in calls.items()}
        assert pointers['old'] == pointers['new'], 'unmatched output placement'
        for _ in range(10):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        samples = {mode: {arm: [] for arm in calls} for mode in ('graph', 'eager')}
        for r in range(15):
            order = ('old', 'new') if r % 2 == 0 else ('new', 'old')
            for arm in order:
                samples['graph'][arm].append(measure([graphs[arm]], 100))
            for arm in order:
                samples['eager'][arm].append(eager(calls[arm]))
        summary = {}
        for mode, rounds in samples.items():
            summary[mode] = dict(
                median={arm: statistics.median(v) for arm, v in rounds.items()},
                ranges={arm: [min(v), max(v)] for arm, v in rounds.items()},
                every_round_faster=all(n < o for o, n in zip(rounds['old'], rounds['new'])),
                apart=max(rounds['new']) < min(rounds['old']), rounds=rounds)
        results[m] = summary
        print(m, json.dumps(summary), flush=True)
        if args.out:
            args.out.write_text(json.dumps(results, indent=2) + '\n')
        del graphs, acts
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
