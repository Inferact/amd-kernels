# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay A/B of the TP2 draft attention -> o_proj chain (2 KV heads, 32 query heads, o_proj N 6144 x K 4096),
K / V and weights cold: where the MXFP8 epilogue (``oproj_input.MXFP8_MIN_ROWS``) pays.

- ``bf16``: the attention writes bf16; ``gemm.mxfp8_linear`` quantizes it and runs the GEMM.
- ``mxfp8``: the attention's MXFP8 epilogue writes the codes in ``gemm.prequant_code_tile_rows``' layout and the
  AITER-layout scales; the GEMM reads them.

7-row (DSpark 7) and 3-row (DSpark 3) non-causal blocks, window 4096 (``benchmark_decode.make_launches``: a 1 GiB
pool, disjoint pages per call). One graph per chain holds ``--calls`` launches, each with its own inputs and one of
``--weights`` o_proj weights (40 x 25 MB > the 256 MB MALL); rounds alternate the chains, median us per call. Before
timing the codes / scales equal the served quantizer of the bf16 output.

    python3 benchmarks/attention/swa_decode_fp8kv_gqa16_d128/benchmark_oproj_input.py [--ql 7 3] [--reqs 1 2 4]
"""

import argparse
import statistics

import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
    aiter_scale_offsets,
    tile_code_offsets,
)
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import swa_decode_fp8kv
from amd_kernels.gemm.mxfp8_common import gemm as G

from benchmarks.attention.swa_decode_fp8kv_gqa16_d128 import benchmark_decode as swa

N, K, HKV = 6144, 4096, 2
SM_SCALE = 128**-0.5
REQS = {7: [1, 2, 3, 4, 8, 16, 32, 48, 64], 3: [1, 2, 3, 4, 8, 16, 32, 64]}


def make_weights(count, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(count):
        w = (torch.randn((N, K), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        s = torch.randint(118, 124, (N, K // 32), generator=gen, device="cuda").to(torch.uint8)
        out.append(G.preshuffle_mxfp8_weight(w, s))
    return out


def gemm_codes(codes, scales, tile, rows, w, ws):
    c8 = codes.view(torch.float8_e4m3fn)
    if tile:
        return G.mxfp8_gemm_shuffled(c8, scales, w, ws, torch.bfloat16, rows)
    return G.mxfp8_gemm(c8[:rows], scales, w, ws, torch.bfloat16, activation_scale_preshuffled=True)


def capture(fns):
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    torch.cuda.synchronize()
    return g


def run_case(ql, reqs, calls, weights, rounds):
    rows = reqs * ql
    tile = G.prequant_code_tile_rows(rows, N, K, torch.bfloat16)
    pad = max(tile, 16)
    rows_p = (rows + pad - 1) // pad * pad
    codes = torch.empty((rows_p, K), dtype=torch.uint8, device="cuda")
    scales = torch.empty(((rows_p + 31) // 32 * 32, K // 32), dtype=torch.uint8, device="cuda")
    out = torch.empty((rows, HKV * 16, 128), dtype=torch.bfloat16, device="cuda")
    pool, launches = swa.make_launches(reqs, ql, HKV, 128, calls, seed=reqs * 100 + ql)
    one = torch.ones((), dtype=torch.float32, device="cuda")

    def attn(i, quant):
        bt, seq, qsl, q = launches[i]
        swa_decode_fp8kv(q, pool, bt, seq, qsl, ql, SM_SCALE, 4096, False, one, one, None if quant else out,
                         quant_out=(codes, scales, tile) if quant else None)

    def chain_bf16(i):
        attn(i, False)
        w, ws = weights[i % len(weights)]
        G.mxfp8_linear(out.view(rows, K), w, ws, torch.bfloat16)

    def chain_mxfp8(i):
        attn(i, True)
        w, ws = weights[i % len(weights)]
        gemm_codes(codes, scales, tile, rows, w, ws)

    attn(0, False)
    out2d = out.view(rows, K).clone()
    codes.fill_(0xA5)
    scales.fill_(0xA5)
    attn(0, True)
    torch.cuda.synchronize()
    want_c, want_s = G.mxfp8_e4m3_quantize_shuffled(out2d, tile) if tile else G.mxfp8_e4m3_quantize_aiter(out2d)
    if tile:
        offs = tile_code_offsets(rows, K, codes.device)
        got_c, ref_c = codes.view(-1)[offs], want_c.view(torch.uint8).view(-1)[offs]
    else:
        got_c, ref_c = codes[:rows], want_c.view(torch.uint8)[:rows]
    so = aiter_scale_offsets(rows, K, codes.device)
    assert torch.equal(got_c, ref_c) and torch.equal(scales.view(-1)[so], want_s.view(torch.uint8).view(-1)[so])
    graphs = {
        "bf16": capture([lambda i=i: chain_bf16(i) for i in range(calls)]),
        "mxfp8": capture([lambda i=i: chain_mxfp8(i) for i in range(calls)]),
    }
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    times = {k: [] for k in graphs}
    for _ in range(rounds):
        for name, g in graphs.items():
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(10):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            times[name].append(s.elapsed_time(e) * 1e3 / (10 * calls))
    bf16, mx = statistics.median(times["bf16"]), statistics.median(times["mxfp8"])
    print(f"ql={ql} R={reqs:3d} rows={rows:4d} tile={tile:3d}: bf16 {bf16:7.2f} us, mxfp8 {mx:7.2f} us "
          f"({100 * (mx / bf16 - 1):+.1f}%)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ql", type=int, nargs="*", default=[7, 3])
    p.add_argument("--reqs", type=int, nargs="*")
    p.add_argument("--calls", type=int, default=100)
    p.add_argument("--weights", type=int, default=40)
    p.add_argument("--rounds", type=int, default=7)
    a = p.parse_args()
    assert "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    weights = make_weights(a.weights, 5)
    with torch.inference_mode():
        for ql in a.ql:
            for reqs in a.reqs or REQS[ql]:
                run_case(ql, reqs, a.calls, weights, a.rounds)


if __name__ == "__main__":
    main()
