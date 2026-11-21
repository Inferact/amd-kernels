# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of the whole Engram step of a decode batch (both layers: hash + row pull, then per layer the
``wkv`` GEMM [N 25600, K 6144] and the gated update) against vLLM's chain (Triton hash + lookups to BF16 rows, then per
layer the MXFP8 linear from the BF16 rows [activation quantizer + the GEMM with the generic untuned rows] and its Triton
gated update), and the GEMM alone: from the pulled MXFP8 rows with the tuned rows vs from BF16 rows with the generic
ones. One GPU, the shards one allocation at the served size (local
HBM, no xGMI); the two layers' wkv weights (157 MB each) alternate, so the
GEMMs read them cold. Ten distinct batches per graph, median of 5 rounds x 5 replays, us per step.
Run: python3 benchmarks/engram/ngram_rows_mxfp8_d256_c24_ranks8/benchmark_wkv_chain.py [--rows 16 64 ...]
"""

import argparse
import contextlib
import statistics

import torch
from amd_kernels.engram.gated_post_bf16_h5120_hc4 import gated_post
from amd_kernels.engram.ngram_rows_mxfp8_d256_c24_ranks8 import NgramRows
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8
from benchmarks.engram.gated_post_bf16_h5120_hc4.benchmark_gated_post import graph
from benchmarks.engram.ngram_rows_mxfp8_d256_c24_ranks8.benchmark_ngram_rows import vllm_chain
from tests.engram.ngram_rows_mxfp8_d256_c24_ranks8 import workload as wl

NIN = 10
H, HC, N, K = 5120, 4, 25600, 6144


def timed(gr):
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            gr.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (5 * NIN))
    return statistics.median(ts)


@contextlib.contextmanager
def untuned():
    """The dispatcher without the wkv rows: what the shape ran before they were added (the generic fallback)."""
    tuned = g._load_tuned_configs()
    g._load_tuned_configs = lambda: {k: v for k, v in tuned.items() if k[:2] != (N, K)}
    try:
        yield
    finally:
        g._load_tuned_configs = lambda: tuned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[16, 32, 64, 128, 192, 256, 384, 512, 768])
    args = ap.parse_args()
    dev = torch.device("cuda")
    try:
        from vllm.models.deepseek_v41.common.engram import _fused_engram_post_wkv_kernel as triton_post
        from vllm.triton_utils import triton
    except ImportError:
        raise SystemExit("needs vLLM's DeepSeek-V4.1 module for the baseline")
    primes, offsets, mult = wl.layout()
    totals = [int(primes[layer].sum()) for layer in range(2)]
    peer_rows = -(-max(totals) // wl.RANKS)
    tmap = wl.token_map(3).to(dev)
    pad_id = int(tmap[wl.PAD_ID_TOKEN])
    bufs = [torch.empty(peer_rows * (256 + 8), dtype=torch.uint8, device=dev) for _ in range(2)]
    for b in bufs:
        b.view(torch.int64).random_()
    mult_d, primes_d, offsets_d = mult.to(dev), primes.to(dev), offsets.to(dev)
    ours = NgramRows(tmap, mult_d, primes_d, offsets_d, pad_id, peer_rows, [[b.data_ptr()] * 8 for b in bufs],
                     [[b.data_ptr() + peer_rows * 256] * 8 for b in bufs], max_rows=max(args.rows), image_id=wl.IMAGE_ID,
                     keepalive=bufs)
    ours.warm_up()
    peer_ptrs = [torch.tensor([b.data_ptr()] * 8, dtype=torch.int64, device=dev) for b in bufs]
    chain = vllm_chain(tmap, mult_d, primes_d.view(-1), offsets_d.view(-1), pad_id, peer_ptrs, peer_rows, totals)
    weights = []
    for layer in range(2):
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
        codes, exps = quantize_mxfp8(w)
        weights.append(g.preshuffle_mxfp8_weight(codes, exps))
        del w, codes, exps
    qw = [torch.randn(HC, H, device=dev).bfloat16() * 0.3 for _ in range(2)]
    kw = [torch.randn(HC, H, device=dev).bfloat16() * 0.3 for _ in range(2)]
    print("rows | step: this entry us | vLLM chain us | wkv GEMM tuned, pulled rows us | generic, BF16 rows us | rows pull us | vLLM hash+lookup us")
    for m in args.rows:
        reqs = max(1, m // 6)
        ins = []
        for i in range(NIN):
            ids, pos, qsl, lb, _ = wl.decode_batch(reqs, 6, m, 7000 * m + i, slots=160)
            ins.append([t.to(dev) for t in (ids, pos, qsl, lb)])
        hs = [torch.randn(m, HC, H, device=dev).bfloat16() for _ in range(NIN)]
        mask = torch.rand(m, device=dev) > 0.01
        hashes = torch.empty(m, 2, 24, dtype=torch.int32, device=dev)
        rows = [torch.empty(m, 24, 256, dtype=torch.bfloat16, device=dev) for _ in range(2)]
        outs = [torch.empty(m, HC, H, dtype=torch.bfloat16, device=dev) for _ in range(2)]

        tile = g.prequant_code_tile_rows(m, N, K)

        def ours_step(parts=("pull", "gemm", "post")):
            for (ids, pos, qsl, lb), h in zip(ins, hs):
                if "pull" in parts:
                    ours(ids, pos, qsl, lb, tile)
                for layer in range(2):
                    act, sc = ours.activation(layer, m, tile)
                    if "gemm" in parts:
                        kv = g.mxfp8_gemm_prequant(act, sc, *weights[layer], torch.bfloat16, m)
                    if "post" in parts:
                        gated_post(h, kv, qw[layer], kw[layer], mask, out=outs[layer])

        def base_step(parts=("pull", "gemm", "post")):
            for (ids, pos, qsl, lb), h in zip(ins, hs):
                if "pull" in parts:
                    chain(ids, pos, qsl, lb, hashes, rows)
                for layer in range(2):
                    if "gemm" in parts:
                        kv = g.mxfp8_linear(rows[layer].view(m, K), *weights[layer])
                    if "post" in parts:
                        triton_post[(m * HC,)](
                            h, kv, qw[layer], kw[layer], mask, outs[layer], m, h.stride(0), h.stride(1), h.stride(2),
                            kv.stride(0), kv.stride(1), qw[layer].stride(0), qw[layer].stride(1), kw[layer].stride(0),
                            kw[layer].stride(1), mask.stride(0), outs[layer].stride(0), outs[layer].stride(1),
                            outs[layer].stride(2), 1e-20, 1e-6, DIM=H, HC_MULT=HC, BLOCK_SIZE=triton.next_power_of_2(H),
                            HAS_MASK=True, num_warps=8)

        kv0 = torch.randn(m, (HC + 1) * H, device=dev).bfloat16()

        def measure(fn, parts, baseline=False):
            def body():
                fn(parts)

            with untuned() if baseline else contextlib.nullcontext():
                gr = graph(body)
            t = [timed(gr), timed(gr)]
            del gr
            return statistics.median(t)

        # the post and GEMM parts need a kv: the full step provides it, the partial ones reuse a fixed one
        full_o, full_b = measure(ours_step, ("pull", "gemm", "post")), measure(base_step, ("pull", "gemm", "post"), True)
        pull_o, pull_b = measure(ours_step, ("pull",)), measure(base_step, ("pull",), True)
        gemm_o = measure(ours_step, ("gemm",)) / 2
        gemm_b = measure(base_step, ("gemm",), True) / 2
        print(f"{m} | {full_o:.1f} | {full_b:.1f} | {gemm_o:.1f} | {gemm_b:.1f} | {pull_o:.1f} | {pull_b:.1f}", flush=True)
        del ins, hs, kv0
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
