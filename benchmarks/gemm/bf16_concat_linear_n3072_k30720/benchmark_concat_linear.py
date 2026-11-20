# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One GPU, cold weights: the N-shard GEMM + slab reduce (3 rotated packed 3072-column copies, 566 MB) vs torch.cat +
F.linear of the replicated weight (2 rotated copies, 755 MB; what vLLM runs at TP2), 20 distinct inputs per graph, 5
alternated rounds x 10 replays, us per call (median [min-max]). The 2-rank gather is not included (see
benchmark_comm.py). ``--sweep`` times tier candidates (kernel, row tiles, column tiles per wave, K slabs) instead.

    python3 benchmarks/gemm/bf16_concat_linear_n3072_k30720/benchmark_concat_linear.py [--rows ...] [--sweep]
"""
import argparse
import statistics
import torch
from amd_kernels.gemm.bf16_concat_linear_n3072_k30720 import bf16_concat_linear, pack_weight
from amd_kernels.gemm.bf16_concat_linear_n3072_k30720.host import config_for

CANDIDATES = {
    'eightwave': [(8, nt, s) for nt in (2, 3, 4, 6) for s in (1, 2, 3, 4, 6, 8)],
}


def _lds_mt(m, mh):
    tiles = (m + 128 * mh - 1) // (128 * mh)
    return max(1, (m + 16 * mh * tiles - 1) // (16 * mh * tiles))


def _candidates(kind, m):
    if kind in ('lds', 'lds1', 'lds3'):
        mh = 1 if kind == 'lds1' else 2
        return [(_lds_mt(m, mh), nr, s) for nr in (8, 12, 16, 24) for s in (5, 10, 15, 20, 30, 40)]
    return CANDIDATES[kind]


def _valid(kind, mt, nt, s):
    if kind in ('lds', 'lds1', 'lds3'):
        mh = 1 if kind == 'lds1' else 2
        ring = 3 if kind == 'lds3' else 2
        nb = 480 // s
        return 96 % nb == 0 and 192 % nt == 0 and nt % 4 == 0 and ring * (nt + mt * mh) * 2048 <= 160 * 1024
    nw = 8 if kind == 'eightwave' else 4
    steps = 960 // s
    return 960 % s == 0 and steps % 2 == 0 and 3072 % (16 * nt * nw) == 0


def _time(graphs, calls):
    for gr in graphs:
        gr.replay()
    torch.cuda.synchronize()
    ts = [[] for _ in graphs]
    for r in range(5):
        order = range(len(graphs)) if r % 2 == 0 else reversed(range(len(graphs)))
        for k in order:
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(10):
                graphs[k].replay()
            e.record()
            e.synchronize()
            ts[k].append(s.elapsed_time(e) * 1e3 / (10 * calls))
    return ts


def _graph(fn, calls):
    fn(0)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for i in range(calls):
            fn(i)
    return gr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, nargs='+', default=[8, 56, 64, 80, 128, 256, 448, 512, 1024, 2048])
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--kinds', nargs='+', default=list(CANDIDATES) + ['lds', 'lds1', 'lds3'])
    a = p.parse_args()
    g = torch.Generator(device='cuda').manual_seed(0)
    full = [(torch.randn((6144, 30720), device='cuda', generator=g) * 0.02).bfloat16() for _ in range(1 if a.sweep else 2)]
    packed = [pack_weight(full[0][:3072].clone()) for _ in range(3)]
    calls = 20
    for m in a.rows:
        xs = [[torch.randn((m, 6144), device='cuda', generator=g).bfloat16() for _ in range(5)] for _ in range(calls)]
        outs = [torch.empty((m, 3072), device='cuda', dtype=torch.bfloat16) for _ in range(calls)]
        f = lambda v: '%.1f [%.1f-%.1f]' % (statistics.median(v), min(v), max(v))  # noqa: E731
        if a.sweep:
            res = {}
            for kind in a.kinds:
                for mt, nt, s in _candidates(kind, m):
                    if not _valid(kind, mt, nt, s) or (mt * 16 > 2 * m and kind not in ('lds', 'lds1', 'lds3')):
                        continue
                    cfg = (kind, mt, nt, s)
                    try:
                        gr = _graph(lambda i: bf16_concat_linear(xs[i], packed[i % 3], out=outs[i], config=cfg), calls)
                    except Exception as ex:  # noqa: BLE001
                        print(f'rows {m} {cfg}: {type(ex).__name__}', flush=True)
                        continue
                    res[cfg] = statistics.median(_time([gr], calls)[0])
                    del gr
            best = sorted(res.items(), key=lambda kv: kv[1])[:8]
            print(f'rows {m}: tier {config_for(m)} {res.get(config_for(m), float("nan")):.1f} | best ' +
                  ' '.join(f'{k}:{v:.1f}' for k, v in best), flush=True)
            continue

        def old(i):
            return torch.nn.functional.linear(torch.cat(xs[i], 1), full[i % 2])

        def new(i):
            return bf16_concat_linear(xs[i], packed[i % 3], out=outs[i])
        ts = _time([_graph(old, calls), _graph(new, calls)], calls)
        flops = 2 * m * 3072 * 30720
        floor = max((3072 * 30720 * 2 + m * 61440) / 6.5e12, flops / 2.5e15) * 1e6
        print(f'rows {m}: cat + F.linear (replicated) {f(ts[0])} us, N-shard GEMM + reduce {f(ts[1])} us, '
              f'roofline {floor:.1f} us ({floor / statistics.median(ts[1]):.0%}), tier {config_for(m)}', flush=True)
        del xs


if __name__ == '__main__':
    main()
