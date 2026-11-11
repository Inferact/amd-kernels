# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 QKV -> norm/RoPE/NVFP4 insert, standalone reduce versus deferred reduce.

MiniMax-M3.1-Flash checkpoint (MXFP8 attention weights), layer 39 rank 0, 26 preshuffled weight copies (>400 MB),
100 seeded inputs per graph through 2048 rows and 16 above, 5 alternated rounds
of 10 replays, three repetitions. No model-framework imports.

  python3 benchmarks/gemm/mxfp8_n2560_k6144/benchmark_qkv_insert.py --checkpoint /path/to/model
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
    norm_rope_nvfp4_insert,
    splitk_reduce_norm_rope_nvfp4_insert,
)


def weights(root, layer):
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'language_model.model.layers.{layer}.self_attn.'

    def read(name):
        key = prefix + name
        with safe_open(str(root / index[key]), framework='pt') as f:
            return f.get_tensor(key)

    codes, scales = [], []
    for name, rows in [('q_proj', 2048), ('k_proj', 128), ('v_proj', 128),
                       ('index_q_proj', 128), ('index_k_proj', 128)]:
        codes.append(read(name + '.weight')[:rows])
        scales.append(read(name + '.weight_scale')[:rows])
    w, ws = g.preshuffle_mxfp8_weight(torch.cat(codes).view(torch.float8_e4m3fn).cuda(),
                                   torch.cat(scales).view(torch.uint8).cuda())
    norm = [read(n + '.weight').cuda().contiguous() for n in
            ('q_norm', 'k_norm', 'index_q_norm', 'index_k_norm')]
    return [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(25)], norm


def inputs(m, count, norm):
    rng = np.random.default_rng(7 * m + 1)
    plans, total = [], 0
    for _ in range(count):
        lens = [8] * (m // 8) + ([m % 8] if m % 8 else []) if m <= 512 else [m]
        pos, slots = [], []
        for length in lens:
            start = int(rng.integers(0, 131072 - length))
            p = np.arange(start, start + length)
            pos.extend(p)
            slots.extend((total + p // 128 - start // 128) * 128 + p % 128)
            total += (start + length - 1) // 128 - start // 128 + 1
        plans.append((torch.tensor(pos, device='cuda'), torch.tensor(slots, device='cuda')))
    perm = torch.tensor(rng.permutation(total), device='cuda')
    plans = [(p, perm[s // 128] * 128 + s % 128) for p, s in plans]
    kv = torch.zeros(total, 2, 128, 72, dtype=torch.uint8, device='cuda')
    idx = torch.zeros(total, 128, 72, dtype=torch.uint8, device='cuda')
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    f = torch.arange(131072, dtype=torch.float64)[:, None] * inv
    cs = torch.cat([f.cos(), f.sin()], 1).to(torch.bfloat16).cuda().contiguous()
    one = torch.tensor(1., device='cuda')
    a, arguments = [], []
    for i, (p, s) in enumerate(plans):
        x = torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal(
            (m, 6144), dtype=np.float32)).cuda().to(torch.bfloat16)
        tile = g.prequant_code_tile_rows(m, 2560, 6144)
        if tile:
            c, sc = g.mxfp8_e4m3_quantize_shuffled(x, tile)
        else:
            c, sc = g.mxfp8_e4m3_quantize_aiter(x)
        a.append((c, sc, tile))
        q = torch.empty(m, 2048, dtype=torch.bfloat16, device='cuda')
        iq = torch.empty(m, 128, dtype=torch.float8_e4m3fn, device='cuda')
        arguments.append((*norm, cs, p, s, s, kv, idx, one, one, q, iq))
    return a, arguments


def measure(graphs, count):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for replay in range(10):
        graphs[replay % len(graphs)].replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (10 * count)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--layer', type=int, default=39)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--previous-prefill', action='store_true',
                    help='Compare the previous 256x256 prefill row with the current public entry.')
    ap.add_argument('--ms', default='8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768')
    args = ap.parse_args()
    wc, norm = weights(args.checkpoint, args.layer)
    kw = dict(num_heads=16, num_kv_heads=1, num_index_heads=1, eps=1e-6)
    names = ('previous_prefill', 'current_prefill') if args.previous_prefill else ('standalone_reduce', 'deferred_reduce')
    for m in map(int, args.ms.split(',')):
        count = 100 if m <= 2048 else 16
        acts, ins = inputs(m, count, norm)

        def call(j, deferred, weight_offset=0):
            a, asc, tile = acts[j]
            w, ws = wc[(j + weight_offset) % len(wc)]
            config = (g._Config(256, 256, 128, 1, 1, 1, g._FOURWAVE_NT_KERNEL, 2, 2)
                      if args.previous_prefill and not deferred and 3073 <= m <= 4800 else None)
            reduce = deferred and not args.previous_prefill
            if tile:
                qkv = g.mxfp8_gemm_shuffled(a, asc, w, ws, torch.bfloat16, m,
                                           defer_reduce=reduce, config=config)
            else:
                qkv = g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16,
                                  activation_scale_preshuffled=True, config=config)
            if args.previous_prefill and torch.cuda.is_current_stream_capturing():
                ptrs[int(deferred)].add(qkv.data_ptr())
            if isinstance(qkv, g.SkPartials):
                splitk_reduce_norm_rope_nvfp4_insert(qkv.tmp, qkv.split_k, *ins[j], **kw)
            else:
                norm_rope_nvfp4_insert(qkv, *ins[j], **kw)

        graphs = []
        pool = torch.cuda.graph_pool_handle()
        ptrs = [set(), set()]
        # Sixteen inputs cannot visit 26 weights in a single graph. Cycle graph
        # instances with different weight offsets so each timed set exceeds 400 MB.
        phases = (len(wc) + count - 1) // count
        for deferred in (False, True):
            arm = []
            for phase in range(phases):
                offset = phase * count
                call(0, deferred, offset)
                call(count - 1, deferred, offset)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    for j in range(count):
                        call(j, deferred, offset)
                arm.append(graph)
            graphs.append(arm)
        if args.previous_prefill:
            assert ptrs[0] == ptrs[1], 'A/B must reuse the same GEMM output addresses'
        for arm in graphs:
            measure(arm, count)
        for rep in range(3):
            rounds = [[], []]
            eager_rounds = [[], []]
            for _ in range(5):
                for arm, arm_graphs in enumerate(graphs):
                    rounds[arm].append(measure(arm_graphs, count))
                if m >= 2048:
                    for arm, deferred in enumerate((False, True)):
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        for phase in range(max(2, phases)):
                            for j in range(count):
                                call(j, deferred, phase * count)
                        b.record()
                        b.synchronize()
                        eager_rounds[arm].append(a.elapsed_time(b) * 1000 / (max(2, phases) * count))
            result = {name: dict(median=statistics.median(values), min=min(values), max=max(values), rounds=values)
                      for name, values in zip(names, rounds)}
            if m >= 2048:
                for name, values in zip(names, eager_rounds):
                    result[name].update(eager_median=statistics.median(values), eager_min=min(values),
                                        eager_max=max(values), eager_rounds=values)
            print(json.dumps(dict(layer=args.layer, m=m, rep=rep, us=result)), flush=True)
        if args.profile:
            from torch.profiler import profile, ProfilerActivity
            for name, arm in zip(names, graphs):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    arm[0].replay()
                    torch.cuda.synchronize()
                kernels = {ev.key: ev.count for ev in prof.key_averages()
                           if ev.device_type is not None and str(ev.device_type).endswith('CUDA')}
                print(json.dumps(dict(m=m, arm=name, kernels=kernels)), flush=True)
        del graphs, acts, ins
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
