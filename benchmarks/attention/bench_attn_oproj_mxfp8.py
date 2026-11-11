# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay A/B of the decode attention -> o_proj chain at the M3.1-Flash TP4 shapes
(1 KV head, 16 query heads, o_proj N 6144 x K 2048), K / V and weights cold.

- ``bf16``: the attention writes bf16; ``gemm.mxfp8_linear`` quantizes it (its table row's
  quantizer: the shuffled Triton quantizer on the decode rows, none on the fused-quant rows
  of 1-32 rows) and runs the GEMM.
- ``mxfp8``: the attention's MXFP8 epilogue writes the codes in the layout of
  ``gemm.prequant_code_tile_rows`` (the (16, 16) tiles of the decode row, row-major else) and
  the AITER-layout scales; the GEMM reads them (``mxfp8_gemm_shuffled`` /
  ``mxfp8_gemm(activation_scale_preshuffled=True)``).
- ``attn_bf16`` / ``attn_mxfp8``: the attention alone in each mode (the breakdown).

Target: ``nvfp4_sparse_decode`` (ql 8 = DSpark 7, ql 4 = DSpark 3), contexts ``--ctx``, the
top-k with the measured serving overlap (``benchmark_sparse_decode.make_calls``' picks, one
copy of the selected pages per call). Draft: ``swa_decode_fp8kv`` (ql 7 / 3, window 4096,
non-causal like the served DSpark draft; ``benchmark_decode.make_launches``: a 1 GiB pool,
disjoint pages per call). One graph per chain holds ``--calls`` launches, each with its
own inputs and one of ``--weights`` o_proj weights (40 x 12.6 MB > the 256 MB MALL).
Rounds alternate the chains (bf16, mxfp8, attn_bf16, attn_mxfp8, ...); median of the
rounds, us per call. Before timing: the mxfp8 codes / scales equal the served quantizer
of the bf16 output (``mxfp8_e4m3_quantize_shuffled`` / ``_aiter``) on the real
rows, and the two chains' o_proj outputs are compared (bitwise where both run the same
GEMM row).

    python3 benchmarks/attention/bench_attn_oproj_mxfp8.py \
        [--kinds target draft] [--ctx 65536 131072] [--rounds 7]
"""

import argparse
import json
import statistics

import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import nvfp4_sparse_decode
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
    aiter_scale_offsets,
    tile_code_offsets,
)
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import swa_decode_fp8kv
from amd_kernels.gemm.mxfp8_common import gemm as G

from benchmarks.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import (
    benchmark_sparse_decode as sd,
)
from benchmarks.attention.swa_decode_fp8kv_gqa16_d128 import benchmark_decode as swa

N, K = 6144, 2048
SM_SCALE = 128**-0.5
TARGET = {8: [1, 2, 4, 6, 8, 16, 32, 64], 4: [1, 8, 16, 32, 64]}
DRAFT = {7: [1, 2, 4, 7, 8, 16, 32, 64], 3: [16, 32, 64]}


def make_weights(count, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(count):
        w = (torch.randn((N, K), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        s = torch.randint(118, 124, (N, K // 32), generator=gen, device="cuda").to(torch.uint8)
        out.append(G.preshuffle_mxfp8_weight(w, s))
    return out


def quant_bufs(rows):
    tile = G.prequant_code_tile_rows(rows, N, K, torch.bfloat16)
    pad = max(tile, 16)
    rows_p = (rows + pad - 1) // pad * pad
    u8 = dict(dtype=torch.uint8, device="cuda")
    codes = torch.empty((rows_p, K), **u8)
    scales = torch.empty(((rows_p + 31) // 32 * 32, K // 32), **u8)
    return codes, scales, tile


def gemm_codes(codes, scales, tile, rows, w, ws):
    c8 = codes.view(torch.float8_e4m3fn)
    if tile:
        return G.mxfp8_gemm_shuffled(c8, scales, w, ws, torch.bfloat16, rows)
    return G.mxfp8_gemm(c8[:rows], scales, w, ws, torch.bfloat16, activation_scale_preshuffled=True)


def served_quant(out2d, tile):
    """The quantizer mxfp8_linear runs for the decode rows (rows < 576: Triton)."""
    if tile:
        return G.mxfp8_e4m3_quantize_shuffled(out2d, tile)
    return G.mxfp8_e4m3_quantize_aiter(out2d)


def compare_bytes(codes, scales, tile, out2d):
    rows = out2d.shape[0]
    want_c, want_s = served_quant(out2d, tile)
    if tile:
        offs = tile_code_offsets(rows, K, codes.device)
        got_c, ref_c = codes.view(-1)[offs], want_c.view(torch.uint8).view(-1)[offs]
    else:
        got_c, ref_c = codes[:rows], want_c.view(torch.uint8)[:rows]
    so = aiter_scale_offsets(rows, K, codes.device)
    got_s, ref_s = scales.view(-1)[so], want_s.view(torch.uint8).view(-1)[so]
    return int((got_c != ref_c).sum()), int((got_s != ref_s).sum())


def time_graphs(graphs, calls, rounds, replays=10, warmup=3):
    for g in graphs.values():
        for _ in range(warmup):
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
    return {k: (statistics.median(v), min(v), max(v)) for k, v in times.items()}


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


def run_case(kind, ql, reqs, ctx, calls, weights, rounds):
    rows = reqs * ql
    codes, scales, tile = quant_bufs(rows)
    out = torch.empty((rows, 16, 128), dtype=torch.bfloat16, device="cuda")
    if kind == "target":
        config = {8: "kv1_bf16_ql8", 4: "kv1_bf16_ql4"}[ql]
        lo = max(ctx - 4096, 4096)
        orig = sd.torch.randint

        def randint(low, high, size, generator=None):  # make_calls' context draw -> ctx
            return orig(lo, ctx + 1, size, generator=generator)

        sd.torch.randint = randint
        try:
            shared, per_call = sd.make_calls(config, reqs, calls, seed=reqs * 7 + ql)
        finally:
            sd.torch.randint = orig

        def attn(i, quant):
            c = per_call[i]
            nvfp4_sparse_decode(
                c["q"], shared["kv_cache"], shared["topk_idx"], c["block_table"],
                shared["seq_lens"], SM_SCALE, out, k_scale=shared["k_scale"],
                v_scale=shared["v_scale"], decode_query_len=ql,
                quant_out=(codes, scales, tile) if quant else None,
            )  # fmt: skip
    else:
        pool, launches = swa.make_launches(reqs, ql, 1, 128, calls, seed=reqs * 100 + ql)
        one = torch.ones((), dtype=torch.float32, device="cuda")

        def attn(i, quant):
            bt, seq, qsl, q = launches[i]
            swa_decode_fp8kv(
                q, pool, bt, seq, qsl, ql, SM_SCALE, 4096, False, one, one,
                None if quant else out, quant_out=(codes, scales, tile) if quant else None,
            )  # fmt: skip

    def chain_bf16(i):
        attn(i, False)
        w, ws = weights[i % len(weights)]
        G.mxfp8_linear(out.view(rows, K), w, ws, torch.bfloat16)

    def chain_mxfp8(i):
        attn(i, True)
        w, ws = weights[i % len(weights)]
        gemm_codes(codes, scales, tile, rows, w, ws)

    # checks on call 0: bytes = served quantizer of the bf16 output; o_proj outputs
    attn(0, False)
    w, ws = weights[0]
    ref = G.mxfp8_linear(out.view(rows, K), w, ws, torch.bfloat16).clone()
    out2d = out.view(rows, K).clone()
    codes.fill_(0xA5)
    scales.fill_(0xA5)
    attn(0, True)
    got = gemm_codes(codes, scales, tile, rows, w, ws)
    torch.cuda.synchronize()
    bad_c, bad_s = compare_bytes(codes, scales, tile, out2d)
    diff = (got.float() - ref.float()).abs()
    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
    bf16_row = G._select_config(rows, N, K, torch.bfloat16).kernel
    graphs = {
        "bf16": capture([lambda i=i: chain_bf16(i) for i in range(calls)]),
        "mxfp8": capture([lambda i=i: chain_mxfp8(i) for i in range(calls)]),
        "attn_bf16": capture([lambda i=i: attn(i, False) for i in range(calls)]),
        "attn_mxfp8": capture([lambda i=i: attn(i, True) for i in range(calls)]),
    }
    t = time_graphs(graphs, calls, rounds)
    rec = {
        "kind": kind, "ql": ql, "reqs": reqs, "rows": rows, "ctx": ctx if kind == "target" else 0,
        "bf16_row": bf16_row, "code_tile": tile,
        "code_mismatch": bad_c, "scale_mismatch": bad_s,
        "oproj_bitwise": bool(torch.equal(got, ref)), "oproj_max_abs": float(diff.max()),
        "oproj_rel_l2": rel,
        **{f"{k}_us": round(v[0], 3) for k, v in t.items()},
        **{f"{k}_range": [round(v[1], 3), round(v[2], 3)] for k, v in t.items()},
    }
    rec["delta_pct"] = round(100 * (rec["mxfp8_us"] / rec["bf16_us"] - 1), 2)
    print("RESULT " + json.dumps(rec), flush=True)
    del graphs
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kinds", nargs="*", default=["target", "draft"])
    p.add_argument("--ctx", type=int, nargs="*", default=[65536, 131072])
    p.add_argument("--calls", type=int, default=100)
    p.add_argument("--weights", type=int, default=40)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--target-ql", type=int, nargs="*", default=[8, 4])
    p.add_argument("--draft-ql", type=int, nargs="*", default=[7, 3])
    p.add_argument("--reqs", type=int, nargs="*")
    a = p.parse_args()
    props = torch.cuda.get_device_properties(0)
    assert "gfx950" in props.gcnArchName, props.gcnArchName
    print(f"{props.name} ({props.gcnArchName}), {props.multi_processor_count} CUs", flush=True)
    weights = make_weights(a.weights, 5)
    with torch.inference_mode():
        if "target" in a.kinds:
            for ctx in a.ctx:
                for ql in a.target_ql:
                    for reqs in a.reqs or TARGET[ql]:
                        run_case("target", ql, reqs, ctx, a.calls, weights, a.rounds)
        if "draft" in a.kinds:
            for ql in a.draft_ql:
                for reqs in a.reqs or DRAFT[ql]:
                    run_case("draft", ql, reqs, 0, a.calls, weights, a.rounds)


if __name__ == "__main__":
    main()
