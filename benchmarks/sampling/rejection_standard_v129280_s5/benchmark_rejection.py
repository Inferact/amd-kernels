# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``rejection_sample`` against vLLM's Triton path (the standard method, a replicated
draft-logits cache) per request count: distinct inputs per graph (a working set larger than the LLC), target rows of
LM-like logits and draft rows that are noisy copies of them in bf16, tokens drawn from the draft rows, temperatures 0,
0.7 and 1.0 mixed. ``--target bf16`` (default): the lm_head GEMM's bf16 rows go straight in; the baseline is what vLLM
runs on them (fp32 copy, ``apply_temperature``, ``rejection_sample``). ``--target fp32``: processed fp32 rows; the
baseline is ``rejection_sample`` alone. Rounds A B B A, median of 5 rounds x 5 replays, us per call.
Run: python3 benchmarks/sampling/rejection_standard_v129280_s5/benchmark_rejection.py [--target fp32] [--reqs 8 32]
"""

import argparse
import statistics

import torch
from amd_kernels.sampling.rejection_standard_v129280_s5 import V, rejection_sample, warmup_rejection_sample
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample as vllm_rejection_sample

WORKING_SET = 2 << 30


class Cache:
    row_mapping = None

    def __init__(self, values):
        self.values = values


def make_input(b, gen, dev, bf16_target):
    base = torch.randn(b * 6, V, generator=gen, device=dev) * 2
    base.scatter_(1, torch.randint(0, V, (b * 6, 16), generator=gen, device=dev),
                  torch.rand(b * 6, 16, generator=gen, device=dev) * 10 + 6)
    temp = torch.tensor([0.0, 0.7, 1.0], device=dev)[torch.arange(b, device=dev) % 3]
    scale = temp.repeat_interleave(6).unsqueeze(1)
    tgt = base.bfloat16() if bf16_target else torch.where(scale > 0, base / scale.clamp_min(1e-6), base)
    rows = base.view(b, 6, V)[:, :5]
    draft = (rows + torch.randn(rows.shape, generator=gen, device=dev)).to(torch.bfloat16)
    probs = torch.softmax(draft.float() / temp.clamp_min(0.5).view(b, 1, 1), dim=-1)
    tok = torch.multinomial(probs.view(b * 5, V), 1, generator=gen).view(b, 5)
    tok = torch.where(temp.view(b, 1) == 0, draft.argmax(-1), tok)
    ds = torch.full((b, 6), -1, dtype=torch.int64, device=dev)
    ds[:, 1:] = tok
    pos = torch.arange(b * 6, device=dev, dtype=torch.int64) + 1000
    seed = torch.randint(0, 2**40, (b,), generator=gen, device=dev, dtype=torch.int64)
    return tgt, draft, ds.view(-1), pos, torch.arange(b, device=dev, dtype=torch.int32), temp, seed


def timed(g, n):
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (5 * n))
    return statistics.median(ts)


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--target", choices=("bf16", "fp32"), default="bf16")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    warmup_rejection_sample(dev)
    bf16 = args.target == "bf16"
    print(f"target logits {args.target}: requests | baseline us | this entry us | change")
    for b in args.reqs:
        gen = torch.Generator(device=dev).manual_seed(b)
        per_input = b * 6 * V * (2 if bf16 else 4) + b * 5 * V * 2
        n = max(4, min(100, -(-WORKING_SET // per_input)))
        inputs = [make_input(b, gen, dev, bf16) for _ in range(n)]
        cu = (torch.arange(b + 1, device=dev) * 6).int()
        elp = torch.arange(6, device=dev).repeat(b).int()
        eidx = inputs[0][4].repeat_interleave(6)

        def ours():
            for tgt, dl, ds, pos, idx, temp, seed in inputs:
                rejection_sample(tgt, dl, ds, pos, idx, temp, seed)

        def base():
            for tgt, dl, ds, pos, idx, temp, seed in inputs:
                if bf16:
                    tgt = tgt.float()
                    apply_temperature(tgt, eidx, temp)
                vllm_rejection_sample(tgt, Cache(dl), ds, cu, pos, idx, eidx, elp, temp, seed, 5)

        go, gb = graph(ours), graph(base)
        t_o, t_b = [], []
        for first in (go, gb, gb, go):
            (t_o if first is go else t_b).append(timed(first, n))
        o, bl = statistics.median(t_o), statistics.median(t_b)
        print(f"{b} | {bl:.1f} | {o:.1f} | {(o / bl - 1) * 100:+.1f}% (A B B A: ours {t_o[0]:.1f}/{t_o[1]:.1f}, "
              f"base {t_b[0]:.1f}/{t_b[1]:.1f}; {n} inputs)", flush=True)
        del go, gb, inputs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
