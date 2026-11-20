# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay A/B of decode attention -> o_proj with the attention's MXFP8 epilogue on or off, at the served
compute (fp16) and per-GPU heads: ``--hkv 1`` TP4 (o_proj N 6144 x K 2048), ``--hkv 2`` TP2 (K 4096); the crossover
behind the layer's ``QUANT_OUT_MIN_ROWS``.

- ``bf16``: the attention writes bf16, ``gemm.mxfp8_linear`` quantizes and multiplies.
- ``mxfp8``: the attention writes the codes in ``gemm.prequant_code_tile_rows``' layout and the AITER-layout scales;
  the GEMM reads them.

Cases: ``benchmark_sparse_decode.make_calls`` (60K-100K contexts, the measured top-k overlap, one copy of the pages per
call); one graph per chain of ``--calls`` launches, each with its own inputs and one of ``--weights`` o_proj weights
(beyond the MALL). Rounds alternate the chains; median us per call. Before timing the MXFP8 bytes are checked against
the served quantizer of the bf16 output.

    python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_oproj_input.py --hkv 2 \
        [--ql 8 4] [--reqs 1 2 4]
"""

import argparse
import statistics

import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import nvfp4_sparse_decode
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
    aiter_scale_offsets,
    tile_code_offsets,
)
from amd_kernels.gemm.mxfp8_common import gemm as G

from benchmarks.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import benchmark_sparse_decode as sd

N = 6144
REQS = {8: [1, 2, 3, 4, 6, 8, 16, 32, 64], 4: [1, 2, 4, 8, 16, 32, 64], 1: [1, 2, 3, 4, 6, 8]}


def make_weights(count, k, seed=5):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(count):
        w = (torch.randn((N, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        s = torch.randint(118, 124, (N, k // 32), generator=gen, device="cuda").to(torch.uint8)
        out.append(G.preshuffle_mxfp8_weight(w, s))
    return out


def gemm_codes(codes, scales, tile, rows, w, ws):
    c8 = codes.view(torch.float8_e4m3fn)
    if tile:
        return G.mxfp8_gemm_shuffled(c8, scales, w, ws, torch.bfloat16, rows)
    return G.mxfp8_gemm(c8[:rows], scales, w, ws, torch.bfloat16, activation_scale_preshuffled=True)


def mismatches(codes, scales, tile, out2d):
    rows, k = out2d.shape
    if tile:
        want_c, want_s = G.mxfp8_e4m3_quantize_shuffled(out2d, tile)
        offs = tile_code_offsets(rows, k, codes.device)
        got_c, ref_c = codes.view(-1)[offs], want_c.view(torch.uint8).view(-1)[offs]
    else:
        want_c, want_s = G.mxfp8_e4m3_quantize_aiter(out2d)
        got_c, ref_c = codes[:rows], want_c.view(torch.uint8)[:rows]
    so = aiter_scale_offsets(rows, k, codes.device)
    return int((got_c != ref_c).sum()) + int((scales.view(-1)[so] != want_s.view(torch.uint8).view(-1)[so]).sum())


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


def time_graphs(graphs, calls, rounds, replays=10):
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    times = {k: [] for k in graphs}
    for _ in range(rounds):
        for name, g in graphs.items():
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(replays):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            times[name].append(s.elapsed_time(e) * 1e3 / (replays * calls))
    return {k: statistics.median(v) for k, v in times.items()}


def run_case(hkv, ql, reqs, calls, weights, rounds):
    k = hkv * 2048
    rows = reqs * ql
    shared, per_call = sd.make_calls(f"kv{hkv}_f16_ql{ql}", reqs, calls, seed=reqs * 7 + ql)
    codes, scales, tile = G.prequant_buffers(rows, N, k, "cuda")
    out = torch.empty((rows, 16 * hkv, 128), dtype=torch.bfloat16, device="cuda")

    def attn(i, quant):
        c = per_call[i]
        nvfp4_sparse_decode(
            c["q"], shared["kv_cache"], shared["topk_idx"], c["block_table"], shared["seq_lens"], sd.SM_SCALE, out,
            k_scale=shared["k_scale"], v_scale=shared["v_scale"], decode_query_len=ql,
            quant_out=(codes, scales, tile) if quant else None, compute_type=torch.float16,
        )

    def chain_bf16(i):
        attn(i, False)
        w, ws = weights[i % len(weights)]
        G.mxfp8_linear(out.view(rows, k), w, ws, torch.bfloat16)

    def chain_mxfp8(i):
        attn(i, True)
        w, ws = weights[i % len(weights)]
        gemm_codes(codes, scales, tile, rows, w, ws)

    attn(0, False)
    out2d = out.view(rows, k).clone()
    attn(0, True)
    torch.cuda.synchronize()
    bad = mismatches(codes, scales, tile, out2d)
    graphs = {
        "bf16": capture([lambda i=i: chain_bf16(i) for i in range(calls)]),
        "mxfp8": capture([lambda i=i: chain_mxfp8(i) for i in range(calls)]),
    }
    t = time_graphs(graphs, calls, rounds)
    delta = 100 * (t["mxfp8"] / t["bf16"] - 1)
    print(f"| {hkv} | {ql} | {reqs} | {rows} | {tile} | {t['bf16']:.2f} | {t['mxfp8']:.2f} | {delta:+.1f}% | {bad} |",
          flush=True)
    del graphs, shared, per_call
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hkv", type=int, default=2)
    p.add_argument("--ql", type=int, nargs="*", default=[8, 4])
    p.add_argument("--reqs", type=int, nargs="*")
    p.add_argument("--calls", type=int, default=100)
    p.add_argument("--weights", type=int, default=24)
    p.add_argument("--rounds", type=int, default=7)
    a = p.parse_args()
    props = torch.cuda.get_device_properties(0)
    assert "gfx950" in props.gcnArchName, props.gcnArchName
    print(f"{props.name}, o_proj N {N} x K {a.hkv * 2048}")
    print("| KV heads | ql | requests | rows | code tile | bf16 chain us | mxfp8 chain us | mxfp8 vs bf16 "
          "| byte mismatches |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    weights = make_weights(a.weights, a.hkv * 2048)
    with torch.inference_mode():
        for ql in a.ql:
            for reqs in a.reqs or REQS[ql]:
                run_case(a.hkv, ql, reqs, a.calls, weights, a.rounds)


if __name__ == "__main__":
    main()
