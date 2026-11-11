# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry (``mxfp8_qkv_norm_rope_nvfp4_insert``) vs the deferred chain (``gemm.mxfp8_gemm_shuffled(defer_reduce=True)``
-> ``SkPartials`` -> ``splitk_reduce_norm_rope_nvfp4_insert``), graph replay, one process, rounds alternated.

  python3 benchmarks/attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128/benchmark_chain.py \
      --checkpoint <M3.1-Flash checkpoint> --ms 1,4,8,16,32 --out /tmp/qkv_chain

Inputs (benchmarks/gemm/mxfp8_n2560_k6144/benchmark_qkv_insert.py's helpers): layer 39 rank 0 of the checkpoint, 26 preshuffled
weight copies (26 x 16.2 MB = 421 MB > 256 MB MALL; the cyclic order keeps >=24 other copies between uses), 100 distinct inputs per graph (normal
bf16 rows quantized by the producer's rule into its (16, 16) code tile; M/8 requests of 8 tokens at random positions,
own pages). Timing: one CUDA graph of 100 calls per arm, 2 warm-up replays, then 5 rounds x 10 replays, arm order
rotated; us per call = round time / 1000; median and [min-max]. Every timing input and every page byte are checked eagerly and after poisoned graph replay.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / 'benchmarks/gemm/mxfp8_n2560_k6144'))
import benchmark_qkv_insert as B  # noqa: E402  (weights(), inputs(): the library bench's helpers)
from amd_kernels.gemm.mxfp8_common import gemm as g  # noqa: E402
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (  # noqa: E402
    norm_rope_nvfp4_insert,
    splitk_reduce_norm_rope_nvfp4_insert,
)
from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128 import (  # noqa: E402
    mxfp8_qkv_norm_rope_nvfp4_insert,
)

KW = dict(num_heads=16, num_kv_heads=1, num_index_heads=1, eps=1e-6)
N, K = 2560, 6144



def weight_rotation(weights, count=100):
    """Visit all 26 weights with >=24 distinct intervening weights, including the graph's cyclic wrap.

    A simple j % 26 leaves only 21 intervening weights at the 100-call wrap. Keep its histogram (22 weights
    four times, four weights three times), but place the first 22 on fixed columns of a 25-call cycle and deal
    the remaining four across the three unoccupied columns. Every weight reuse then spans >=389 MB.
    """
    assert count == 100 and len(weights) == 26
    special = (0, 8, 16)
    ordinary = [j for j in range(25) if j not in special]
    order = [22 + (3 * (j // 25) + special.index(j % 25)) % 4 if j % 25 in special
             else ordinary.index(j % 25) for j in range(count)]
    sizes = [sum(t.numel() * t.element_size() for t in weight) for weight in weights]
    distances = []
    for k in range(26):
        hits = [j for j, w in enumerate(order) if w == k]
        assert len(hits) == (4 if k < 22 else 3)
        for a, b in zip(hits, hits[1:] + [hits[0] + count]):
            others = {order[j % count] for j in range(a + 1, b)}
            distances.append(sum(sizes[w] for w in others))
    assert min(distances) >= 384_000_000, min(distances)
    return order, min(distances)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--layer', type=int, default=39)
    ap.add_argument('--ms', default='8,16,32,64,128,256')
    ap.add_argument('--profile-ms', default='')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--out', required=True, help='path prefix: writes <out>.json and <out>.md')
    args = ap.parse_args()
    torch.cuda.set_device(0)
    dev = torch.cuda.get_device_properties(0)
    meta = dict(device=dev.name, pci_bus=getattr(dev, 'pci_bus_id', None), torch=torch.__version__,
                layer=args.layer, weight_copies=26, inputs=100, protocol='graph of 100 calls, 2 warm-up replays, '
                f'{args.rounds} rounds x 10 replays, arm order rotated')
    try:
        import flydsl
        meta['flydsl'] = getattr(flydsl, '__version__', '?')
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(dict(meta=meta)), flush=True)
    wc, norm = B.weights(args.checkpoint, args.layer)
    count = 100
    weight_order, min_distance = weight_rotation(wc, count)
    print(json.dumps(dict(min_weight_reuse_bytes=min_distance)), flush=True)
    profile_ms = {int(x) for x in args.profile_ms.split(',') if x}
    results = []
    for m in map(int, args.ms.split(',')):
        acts, ins = B.inputs(m, count, norm)
        tick = torch.zeros(64, dtype=torch.int32, device='cuda')

        def gemm(j, deferred):
            a, asc, tile = acts[j]
            w, ws = wc[weight_order[j]]
            assert tile, f'no shuffled-A row at M {m}'
            return g.mxfp8_gemm_shuffled(a, asc, w, ws, torch.bfloat16, m, defer_reduce=deferred)

        def chain(j, deferred):
            qkv = gemm(j, deferred)
            if isinstance(qkv, g.SkPartials):
                splitk_reduce_norm_rope_nvfp4_insert(qkv.tmp, qkv.split_k, *ins[j], **KW)
            else:
                norm_rope_nvfp4_insert(qkv, *ins[j], **KW)
            return qkv

        def entry(j):
            a, asc, tile = acts[j]
            w, ws = wc[weight_order[j]]
            mxfp8_qkv_norm_rope_nvfp4_insert(a, asc, w, ws, m, *ins[j], **KW)

        arms = {
            'deferred': lambda j: chain(j, True),
            'entry': entry,
            'null': lambda j: tick.add_(1),
        }
        # what each arm runs at this M
        cfg_std = g._select_prequant_config(m, N, K, torch.bfloat16)
        cfg_def = g._deferred_qkv_config(m, N, K, cfg_std)
        probe = gemm(0, True)
        info = dict(code_tile=acts[0][2], standalone_config=str(cfg_std), deferred_config=str(cfg_def),
                    deferred_split_k=probe.split_k if isinstance(probe, g.SkPartials) else 0,
                    standalone_split_k=cfg_std.split_k)
        # Every timing input, including stale/unwritten page bytes, has the deferred chain's expected output.
        kv, idx = ins[0][8], ins[0][9]
        kv.fill_(0xA5)
        idx.fill_(0x5A)
        stale_pages = kv.clone(), idx.clone()
        for j in range(count):
            chain(j, True)
        torch.cuda.synchronize()
        expected_q = [args[-2].clone() for args in ins]
        expected_iq = [args[-1].view(torch.uint8).clone() for args in ins]
        expected_pages = kv.clone(), idx.clone()

        def poison():
            kv.copy_(stale_pages[0])
            idx.copy_(stale_pages[1])
            for args in ins:
                args[-2].fill_(7)
                args[-1].view(torch.uint8).fill_(0x77)

        def check():
            torch.cuda.synchronize()
            diffs = [sum(int((args[-2] != want).sum()) for args, want in zip(ins, expected_q)),
                     sum(int((args[-1].view(torch.uint8) != want).sum()) for args, want in zip(ins, expected_iq)),
                     int((kv != expected_pages[0]).sum()), int((idx != expected_pages[1]).sum())]
            assert not any(diffs), (m, diffs)
            return dict(zip(('q', 'index_q', 'kv_bytes', 'index_bytes'), diffs))

        poison()
        for j in range(count):
            entry(j)
        info['entry_vs_deferred_differing'] = check()
        info['checked_inputs'] = count
        # graphs
        pool = torch.cuda.graph_pool_handle()
        graphs = {}
        for name, fn in arms.items():
            fn(0)
            fn(count - 1)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, pool=pool):
                for j in range(count):
                    fn(j)
            graphs[name] = gr
        for name, gr in graphs.items():
            poison()
            for _ in range(2):
                gr.replay()
            if name != 'null':
                check()
        info['graph_checked_inputs'] = count
        torch.cuda.synchronize()
        rounds = {name: [] for name in arms}
        names = list(arms)
        for r in range(args.rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            for name in order:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(10):
                    graphs[name].replay()
                b.record()
                b.synchronize()
                rounds[name].append(a.elapsed_time(b) * 1000 / (10 * count))
        us = {name: dict(median=statistics.median(v), min=min(v), max=max(v), rounds=v) for name, v in rounds.items()}
        row = dict(m=m, us=us, **info)
        if m in profile_ms:
            from torch.profiler import ProfilerActivity, profile
            prof_out = {}
            for name in ('deferred', 'entry', 'null'):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    graphs[name].replay()
                    torch.cuda.synchronize()
                ks = {}
                for ev in prof.key_averages():
                    dt = str(getattr(ev, 'device_type', ''))
                    if not dt.endswith('CUDA'):
                        continue
                    tot = getattr(ev, 'device_time_total', None)
                    if tot is None:
                        tot = getattr(ev, 'cuda_time_total', 0)
                    if ev.count:
                        ks[ev.key] = dict(count=ev.count, avg_us=tot / ev.count)
                prof_out[name] = ks
            row['profile'] = prof_out
        results.append(row)
        print(json.dumps(row), flush=True)
        del graphs, acts, ins
        torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(dict(meta=meta, results=results), indent=1))
    lines = ['# Entry vs deferred chain (us per call; graph of 100 calls on 100 inputs, 26 rotated weight copies, '
             f'2 warm-up + {args.rounds} x 10 replays; median [min-max])', '', f'{json.dumps(meta)}', '',
             '| M | deferred chain | entry | change | null | entry vs deferred bytes (q, iq, kv, idx diffs) |',
             '|---|---|---|---|---|---|']

    def c(r, k):
        u = r['us'][k]
        return f"{u['median']:.3f} [{u['min']:.3f}-{u['max']:.3f}]"

    for r in results:
        d = r['entry_vs_deferred_differing']
        ch = r['us']['entry']['median'] / r['us']['deferred']['median'] - 1
        lines.append(f"| {r['m']} | {c(r, 'deferred')} | {c(r, 'entry')} | {100 * ch:+.1f}% | {c(r, 'null')} | "
                     f"{d['q']}, {d['index_q']}, {d['kv_bytes']}, {d['index_bytes']} |")
    lines += ['', '## Per-kernel durations (torch.profiler, one replay of the 100-call graph, avg us per launch)', '']
    for r in results:
        if 'profile' not in r:
            continue
        lines.append(f"### M {r['m']}")
        for arm, ks in r['profile'].items():
            lines.append(f'- {arm}: ' + '; '.join(f'`{k}` x{v["count"]} {v["avg_us"]:.2f}' for k, v in ks.items()))
        lines.append('')
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('wrote', out.with_suffix('.json'), out.with_suffix('.md'), flush=True)


if __name__ == '__main__':
    main()
