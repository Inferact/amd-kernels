# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Deferred TP4 QKV: interleaved prime versus the preceding W-first prologue.

Layer39 rank0, 26 cold weight copies (>400 MB), 100 seeded inputs, 3 x 5
alternated rounds x 10 graph replays, matching temporary addresses. The complete
reduce/norm/RoPE/insert chain is timed. --fresh also times the activation producer.
"""
import sys
import argparse
import functools
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import compile_mxfp8_gemm_decode
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import splitk_reduce_norm_rope_nvfp4_insert
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mxfp8_n2560_k6144"))  # benchmark_qkv_insert's helpers
from benchmark_qkv_insert import weights, inputs, measure  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--layer', type=int, default=39)
    ap.add_argument('--ms', default='96,97,104,112,120,128,129')
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()
    wc, norm = weights(args.checkpoint, args.layer)
    current = g._decode_launcher

    @functools.cache
    def previous(n, k, c, a_bf16):
        return compile_mxfp8_gemm_decode(
            N=n, K=k, RT=c.tile_m // 16, NI=c.tile_n // 16, KW=c.waves_m, KS=c.split_k,
            LA=(c.la_w, c.la_a) if c.la_w else None, sched_barriers=bool(c.waves_n),
            W_CM=c.w_cm, A_BF16=a_bf16, INTERLEAVE_PRIME=False,
        )

    kw = dict(num_heads=16, num_kv_heads=1, num_index_heads=1, eps=1e-6)
    for m in map(int, args.ms.split(',')):
        assert 81 <= m <= 144
        acts, ins = inputs(m, 100, norm)
        xs = [torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal(
            (m, 6144), dtype=np.float32)).cuda().bfloat16() for i in range(100)] if args.fresh else None
        ptrs = {'previous': set(), 'current': set()}

        def call(j, arm):
            g._decode_launcher = previous if arm == 'previous' else current
            try:
                c, s, tile = acts[j]
                if xs is not None:
                    c, s = g.mxfp8_e4m3_quantize_shuffled(xs[j], tile)
                p = g.mxfp8_gemm_shuffled(c, s, *wc[j % 26], torch.bfloat16, m, defer_reduce=True)
            finally:
                g._decode_launcher = current
            if torch.cuda.is_current_stream_capturing():
                ptrs[arm].add(p.tmp.data_ptr())
            splitk_reduce_norm_rope_nvfp4_insert(p.tmp, p.split_k, *ins[j], **kw)
            return p.tmp

        for j in range(100):
            outs = []
            for arm in ptrs:
                ins[j][8].zero_()
                ins[j][9].zero_()
                out = call(j, arm)
                outs.append([out.clone(), ins[j][-2].clone(), ins[j][-1].view(torch.uint8).clone(),
                             ins[j][8].clone(), ins[j][9].clone()])
            assert all(torch.equal(x, y) for x, y in zip(*outs)), (m, j)
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
                rounds[arm].append(measure([graphs[arm]], 100))
        print(json.dumps(dict(m=m, fresh=args.fresh, byte_equal=True, matched_ptrs=True,
                              median={arm: statistics.median(v) for arm, v in rounds.items()},
                              ranges={arm: [min(v), max(v)] for arm, v in rounds.items()},
                              rounds=rounds)), flush=True)
        del graphs, acts, ins, xs, outs
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
