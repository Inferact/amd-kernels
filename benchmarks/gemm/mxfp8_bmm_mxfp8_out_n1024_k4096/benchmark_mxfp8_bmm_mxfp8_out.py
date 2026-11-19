# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``mxfp8_bmm_mxfp8_out`` (B 8 x 1024 x 4096) against the unfused pair it replaces:
AITER's batched GEMM (``batched_gemm_a8w8_mxscale_bpreshuffle``, its tuned dispatch) to bf16, then vLLM's Triton
MXFP8 quantizer (``_mxfp8_e4m3_quantize_triton``, needs vLLM installed); the AITER GEMM alone is printed too.
Weights in 8 rotated copies (256 MB), inputs rotated past 512 MB; one graph per side, median of 5 rounds, the
sides alternated twice (the faster round printed), us per call.

    python3 benchmarks/gemm/mxfp8_bmm_mxfp8_out_n1024_k4096/benchmark_mxfp8_bmm_mxfp8_out.py [M ...]
"""

import statistics
import sys

import torch
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle

from amd_kernels.gemm.mxfp8_bmm_mxfp8_out_n1024_k4096 import mxfp8_bmm_mxfp8_out

G, N, K = 8, 1024, 4096
ROWS = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
W_COPIES = 8
X_BYTES = 512 << 20


def _time(graph, calls, rounds=5, replays=5):
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(replays):
            graph.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / (replays * calls) * 1e3)
    return statistics.median(out)


def main(rows):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_triton

    ws = [
        (
            torch.randint(-100, 100, (G, N, K), dtype=torch.int8, device="cuda").view(torch.float8_e4m3fn),
            torch.randint(118, 128, (G, N // 32, K // 32), dtype=torch.uint8, device="cuda"),
        )
        for _ in range(W_COPIES)
    ]
    for m in rows:
        n_x = min(64, max(2, -(-X_BYTES // (m * G * K))))
        calls = max(n_x, W_COPIES)
        xs = [
            (
                (torch.randn((m, G, K), device="cuda") * 2).to(torch.float8_e4m3fn),
                torch.randint(118, 128, (m, G, K // 32), dtype=torch.uint8, device="cuda"),
            )
            for _ in range(n_x)
        ]

        def aiter_bmm(i):
            (x, xsc), (w, wsc) = xs[i % n_x], ws[i % W_COPIES]
            return batched_gemm_a8w8_mxscale_bpreshuffle(x, w, xsc, wsc, torch.bfloat16).view(m, G * N)

        def today(i):
            return _mxfp8_e4m3_quantize_triton(aiter_bmm(i))

        def ours(i):
            (x, xsc), (w, wsc) = xs[i % n_x], ws[i % W_COPIES]
            return mxfp8_bmm_mxfp8_out(x.view(m, G * K), xsc.view(m, -1), w, wsc)

        graphs = {}
        for name, fn in (("bmm", aiter_bmm), ("today", today), ("ours", ours)):
            fn(0)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(calls):
                    fn(i)
            graphs[name] = g
        t = {k: [] for k in graphs}
        for _ in range(2):
            for k, g in graphs.items():
                t[k].append(_time(g, calls))
        b = {k: min(v) for k, v in t.items()}
        print(
            f"M={m:6d}: AITER bmm {b['bmm']:8.2f} | bmm + Triton quant {b['today']:8.2f} | "
            f"mxfp8_bmm_mxfp8_out {b['ours']:8.2f} us | {b['ours'] - b['today']:+8.2f} us "
            f"({(b['ours'] / b['today'] - 1) * 100:+.1f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or ROWS)
