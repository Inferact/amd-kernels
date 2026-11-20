# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP2 QKV GEMM (N 4992 x K 6144): the dispatched row versus vLLM's TP2 fallback (Triton ``tl.dot_scaled``,
``_M31_TILES[(4992, 6144)]``), both from pre-quantized activations, and the roofline; ``chain`` adds the TP2
norm / RoPE / NVFP4 insert (32 / 2 / 2 heads): ``mxfp8_gemm_qkv_norm_rope_nvfp4_insert`` (deferred split-K partials
where its row has them) versus the Triton GEMM + the insert.

MiniMax-M3.1-Flash checkpoint, layer 39, TP2 rank 0 slice (q 4096 | k 256 | v 256 | index-q 256 | index-k 128 rows);
14 weight copies per arm (> 400 MB, cold); up to 100 seeded inputs per graph; 3 x 5 alternated rounds of 10
replays, median us per call. Above 2048 rows (outside vLLM's graphs) the eager calls are timed too.
The akr arm gets the codes in the layout ``prequant_code_tile_rows`` names (what the fused norm + quant producer
writes) and calls ``mxfp8_gemm_prequant``; the Triton arm gets row-major codes and scales.

  python3 benchmarks/gemm/mxfp8_n4992_k6144/benchmark_n4992_k6144.py --checkpoint /path/to/model
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.reference import compare

N, K = 4992, 6144
COPIES = 14
ROWS = (("q_proj", 4096), ("k_proj", 256), ("v_proj", 256), ("index_q_proj", 256), ("index_k_proj", 128))


def insert_inputs(root, layer, m):
    """Norm weights of the layer, cos / sin, and NVFP4 pages for ``m`` tokens (8 per request up to 2048 rows)."""
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"language_model.model.layers.{layer}.self_attn."
    norms = []
    for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
        with safe_open(str(root / index[prefix + name + ".weight"]), framework="pt") as f:
            norms.append(f.get_tensor(prefix + name + ".weight").cuda().contiguous())
    rng = np.random.default_rng(7 * m + 1)
    pos, slots, total = [], [], 0
    for length in [8] * (m // 8) + ([m % 8] if m % 8 else []) if m <= 2048 else [m]:
        start = int(rng.integers(0, 131072 - length))
        p = np.arange(start, start + length)
        pos.extend(p)
        slots.extend((total + p // 128 - start // 128) * 128 + p % 128)
        total += (start + length - 1) // 128 - start // 128 + 1
    s = torch.tensor(slots, device="cuda")
    perm = torch.tensor(rng.permutation(total), device="cuda")
    s = perm[s // 128] * 128 + s % 128
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    f = torch.arange(131072, dtype=torch.float64)[:, None] * inv
    one = torch.tensor(1.0, device="cuda")
    return (*norms, torch.cat([f.cos(), f.sin()], 1).to(torch.bfloat16).cuda().contiguous(),
            torch.tensor(pos, device="cuda"), s, s,
            torch.zeros(total, 4, 128, 72, dtype=torch.uint8, device="cuda"),
            torch.zeros(total, 128, 72, dtype=torch.uint8, device="cuda"), one, one)


def weights(root, layer):
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"language_model.model.layers.{layer}.self_attn."

    def read(name):
        with safe_open(str(root / index[prefix + name]), framework="pt") as f:
            return f.get_tensor(prefix + name)

    codes = torch.cat([read(n + ".weight")[:r] for n, r in ROWS]).view(torch.float8_e4m3fn).cuda()
    scales = torch.cat([read(n + ".weight_scale")[:r] for n, r in ROWS]).view(torch.uint8).cuda()
    assert codes.shape == (N, K) and scales.shape == (N, K // 32)
    w, ws = g.preshuffle_mxfp8_weight(codes, scales)
    shuffled = [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(COPIES - 1)]
    raw = [(codes, scales)] + [(codes.clone(), scales.clone()) for _ in range(COPIES - 1)]
    return shuffled, raw


def measure(graphs, calls):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for replay in range(10):
        graphs[replay % len(graphs)].replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (10 * calls)


def eager(call, count, phases):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for phase in range(phases):
        for j in range(count):
            call(j, phase * count)
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (phases * count)


def roofline_us(m):
    moved = (N + m) * K * (1 + 1 / 32) + 2 * m * N
    return max(moved / 6.5e12, 2 * m * N * K / 5e15) * 1e6


def main():
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import _mxfp8_dot_scaled_gemm
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
        mxfp8_gemm_qkv_norm_rope_nvfp4_insert, qkv_norm_rope_nvfp4_insert)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--ms", default="1,8,16,32,64,128,192,256,320,384,448,512,640,768,1024,1536,2048,"
                                    "3072,4096,8192,16384,32768")
    ap.add_argument("--units", default="gemm,chain")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    units = args.units.split(",")
    heads = dict(num_heads=32, num_kv_heads=2, num_index_heads=2, eps=1e-6)
    shuffled, raw = weights(args.checkpoint, args.layer)
    rows = []
    for m in map(int, args.ms.split(",")):
        count = max(2, min(100, (400 << 20) // (m * K * 3)))
        phases = (COPIES + count - 1) // count
        tile = g.prequant_code_tile_rows(m, N, K)
        akr_in, tri_in = [], []
        for i in range(count):
            x = torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal(
                (m, K), dtype=np.float32)).cuda().to(torch.bfloat16)
            akr_in.append(g.mxfp8_e4m3_quantize_shuffled(x, tile) if tile else g.mxfp8_e4m3_quantize_aiter(x))
            tri_in.append(mxfp8_e4m3_quantize(x))
            del x

        def akr(j, off):
            w, ws = shuffled[(j + off) % COPIES]
            return g.mxfp8_gemm_prequant(*akr_in[j], w, ws, torch.bfloat16, m)

        def triton(j, off):
            w, ws = raw[(j + off) % COPIES]
            return _mxfp8_dot_scaled_gemm(*tri_in[j], w, ws, torch.bfloat16)

        metrics = compare(akr(0, 0), triton(0, 0).float())
        assert metrics["cos"] >= 0.99999, metrics
        arms = {}
        if "gemm" in units:
            arms.update(akr=akr, triton=triton)
        if "chain" in units:
            ins = insert_inputs(args.checkpoint, args.layer, m)

            def akr_chain(j, off):
                w, ws = shuffled[(j + off) % COPIES]
                return mxfp8_gemm_qkv_norm_rope_nvfp4_insert(*akr_in[j], w, ws, m, torch.bfloat16, *ins, **heads,
                                                             one_launch=False)

            def triton_chain(j, off):
                return qkv_norm_rope_nvfp4_insert(triton(j, off), *ins, **heads)

            q_akr, q_tri = akr_chain(0, 0)[0], triton_chain(0, 0)[0]
            assert compare(q_akr, q_tri.float())["cos"] >= 0.9999
            arms.update(akr_chain=akr_chain, triton_chain=triton_chain)
        graphs = {}
        pool = torch.cuda.graph_pool_handle()
        for name, call in arms.items():
            graphs[name] = []
            for phase in range(phases):
                call(0, phase * count)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    for j in range(count):
                        call(j, phase * count)
                graphs[name].append(graph)
            measure(graphs[name], count)
        samples = {name: [] for name in arms}
        eager_samples = {name: [] for name in arms}
        for _ in range(3):
            for _ in range(5):
                for name in arms:
                    samples[name].append(measure(graphs[name], count))
                if m > 2048:
                    for name, call in arms.items():
                        eager_samples[name].append(eager(call, count, max(2, phases)))
        row = dict(m=m, tile=tile, config=str(g._select_prequant_config(m, N, K, torch.bfloat16)),
                   roofline_us=roofline_us(m), cos=metrics["cos"])
        for name in arms:
            row[name] = statistics.median(samples[name])
            row[name + "_range"] = [min(samples[name]), max(samples[name])]
            if eager_samples[name]:
                row[name + "_eager"] = statistics.median(eager_samples[name])
        rows.append(row)
        print(json.dumps(row), flush=True)
        del graphs, akr_in, tri_in
        torch.cuda.empty_cache()
    for unit in units:
        a, b = ("akr", "triton") if unit == "gemm" else ("akr_chain", "triton_chain")
        print(f"| M | {unit} us | Triton {unit} us | speedup | GEMM roofline us | GEMM fraction |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['m']} | {r[a]:.2f} | {r[b]:.2f} | {r[b] / r[a]:.2f}x | {r['roofline_us']:.2f} | "
                  f"{100 * r['roofline_us'] / r[a]:.0f}% |")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
