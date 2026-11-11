# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 o_proj: owned AGPR reads versus the preceding owner-copy epilogue.

Layer39 rank0, 32 cold weight copies (>400 MB), 100 seeded inputs,
3 x 5 alternated rounds x 10 graph replays, public entry and one graph pool.
"""
import argparse
import functools
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import compile_mxfp8_gemm_decode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--layer', type=int, default=39)
    ap.add_argument('--ms', default='193,224,256,321,384,385,448,449,512')
    args = ap.parse_args()
    index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'language_model.model.layers.{args.layer}.self_attn.o_proj.'
    raw = []
    for field, cols in [('weight', 2048), ('weight_scale', 64)]:
        with safe_open(str(args.checkpoint / index[prefix + field]), framework='pt') as f:
            raw.append(f.get_tensor(prefix + field)[:, :cols].contiguous().cuda())
    w, ws = g.preshuffle_mxfp8_weight(raw[0].view(torch.float8_e4m3fn), raw[1].view(torch.uint8))
    copies = [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(31)]
    current = g._decode_launcher

    @functools.cache
    def previous(n, k, c, a_bf16):
        return compile_mxfp8_gemm_decode(
            N=n, K=k, RT=c.tile_m // 16, NI=c.tile_n // 16,
            KW=c.waves_m, KS=c.split_k,
            LA=(c.la_w, c.la_a) if c.la_w else None,
            sched_barriers=bool(c.waves_n), W_CM=c.w_cm,
            A_BF16=a_bf16, OWNER_READ=False,
        )

    for m in map(int, args.ms.split(',')):
        assert 193 <= m <= 256 or 321 <= m <= 512
        tile = g.prequant_code_tile_rows(m, 6144, 2048)
        acts = []
        for i in range(100):
            x = torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal(
                (m, 2048), dtype=np.float32)).cuda().bfloat16()
            acts.append(g.mxfp8_e4m3_quantize_shuffled(x, tile))
        ptrs = {'previous': set(), 'current': set()}

        def call(j, arm):
            g._decode_launcher = previous if arm == 'previous' else current
            out = g.mxfp8_gemm_shuffled(*acts[j], *copies[j % 32], torch.bfloat16, m)
            if torch.cuda.is_current_stream_capturing():
                ptrs[arm].add(out.data_ptr())
            return out

        for j in range(100):
            assert torch.equal(call(j, 'previous'), call(j, 'current')), (m, j)
        pool = torch.cuda.graph_pool_handle()
        graphs = {}
        for arm in ptrs:
            call(0, arm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                for j in range(100):
                    call(j, arm)
            graphs[arm] = graph
        assert ptrs['previous'] == ptrs['current']
        for _ in range(10):
            for graph in graphs.values():
                graph.replay()
        rounds = {arm: [] for arm in ptrs}
        for r in range(15):
            for arm in (('previous', 'current') if r % 2 == 0 else ('current', 'previous')):
                start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(10):
                    graphs[arm].replay()
                stop.record()
                stop.synchronize()
                rounds[arm].append(start.elapsed_time(stop) * 1000 / 1000)
        print(json.dumps(dict(m=m, same_bf16=True,
                              median={arm: statistics.median(v) for arm, v in rounds.items()},
                              ranges={arm: [min(v), max(v)] for arm, v in rounds.items()},
                              rounds=rounds)), flush=True)
        del graphs, acts
        torch.cuda.empty_cache()
    g._decode_launcher = current


if __name__ == '__main__':
    main()
