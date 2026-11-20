# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP2 QKV chain (N 4992 x K 6144, 32 / 2 / 2 heads): the one-launch entry (``mxfp8_qkv_norm_rope_nvfp4_insert``)
vs the deferred chain (``gemm.mxfp8_gemm_shuffled(defer_reduce=True)`` -> split-K reduce-insert or bf16 GEMM ->
``norm_rope_nvfp4_insert``) vs vLLM's TP2 stock path (Triton ``tl.dot_scaled`` on row-major codes -> vLLM's FlyDSL
``minimax_m3_norm_rope_nvfp4_insert``), and the roofline.

  python3 benchmarks/attention/mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128/benchmark_chain.py \
      --checkpoint <M3.1-Flash checkpoint> --out /tmp/qkv_fused

Layer 39 TP2 rank 0 weights in 14 cold copies (> 400 MB), up to 100 seeded inputs per graph (decode rows: requests
of 8 tokens at random positions, own pages); graph replay, 15 rounds of 10 replays (the amd_kernels arms alternate,
then the stock arm), median us per call.
Every arm's bytes are checked against the deferred chain before timing (q, index-q, pages; stock: reported only).
Past the shuffled-code tile (769+ rows) the entry arm is the insert package's dispatcher (row-major codes).
Roofline: 6.5 TB/s over the weight, the codes and the per-token insert bytes (q 8192, index-q 256, positions / slots
24, cos / sin 128, five NVFP4 rows 360), or 5 PF/s, whichever is longer.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gemm/mxfp8_n4992_k6144"))
import benchmark_n4992_k6144 as B  # noqa: E402  (weights(), insert_inputs(): the TP2 GEMM bench's helpers)
from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128 import (  # noqa: E402
    host,
    mxfp8_qkv_norm_rope_nvfp4_insert,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (  # noqa: E402
    norm_rope_nvfp4_insert,
    splitk_reduce_norm_rope_nvfp4_insert,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import qkv as dispatch  # noqa: E402
from amd_kernels.gemm.mxfp8_common import gemm as g  # noqa: E402

N, K = 4992, 6144
HEADS = dict(num_heads=32, num_kv_heads=2, num_index_heads=2, eps=1e-6)
TOKEN_BYTES = 8192 + 256 + 24 + 128 + 360


def roofline_us(m):
    moved = (N + m) * K * (1 + 1 / 32) + m * TOKEN_BYTES
    return max(moved / 6.5e12, 2 * m * N * K / 5e15) * 1e6


def main():
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import _mxfp8_dot_scaled_gemm
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.models.minimax_m3.amd.ops.norm_rope_nvfp4_insert.host import minimax_m3_norm_rope_nvfp4_insert

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--ms", default="1,2,4,8,12,16,24,32,48,64,96,128,192,256,320,384,416,448,512,640,768,1024,1536")
    ap.add_argument("--arms", default="entry,deferred,stock")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    shuffled, raw = B.weights(args.checkpoint, args.layer)
    copies = len(shuffled)
    rows = []
    for m in map(int, args.ms.split(",")):
        count = max(2, min(100, (400 << 20) // (m * K * 3)))
        phases = (copies + count - 1) // count
        tile = g.prequant_code_tile_rows(m, N, K)
        akr_in, tri_in = [], []
        for i in range(count):
            x = torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal(
                (m, K), dtype=np.float32)).cuda().to(torch.bfloat16)
            akr_in.append(g.mxfp8_e4m3_quantize_shuffled(x, tile) if tile else g.mxfp8_e4m3_quantize_aiter(x))
            tri_in.append(mxfp8_e4m3_quantize(x))
            del x
        ins = B.insert_inputs(args.checkpoint, args.layer, m)
        q = torch.empty((m, 4096), dtype=torch.bfloat16, device="cuda")
        iq = torch.empty((m, 256), dtype=torch.float8_e4m3fn, device="cuda")

        def entry(j, off):
            w, ws = shuffled[(j + off) % copies]
            if tile:
                mxfp8_qkv_norm_rope_nvfp4_insert(*akr_in[j], w, ws, m, *ins, q, iq, **HEADS)
            else:  # rows past the shuffled-code tile: the dispatcher's GEMM + insert
                dispatch.mxfp8_gemm_qkv_norm_rope_nvfp4_insert(*akr_in[j], w, ws, m, torch.bfloat16, *ins, **HEADS)

        def deferred(j, off):
            w, ws = shuffled[(j + off) % copies]
            if not tile:
                out = g.mxfp8_gemm(*akr_in[j], w, ws, torch.bfloat16, activation_scale_preshuffled=True)
                norm_rope_nvfp4_insert(out, *ins, q, iq, **HEADS)
                return
            out = g.mxfp8_gemm_shuffled(*akr_in[j], w, ws, torch.bfloat16, m, defer_reduce=True)
            if isinstance(out, g.SkPartials):
                splitk_reduce_norm_rope_nvfp4_insert(out.tmp, out.split_k, *ins, q, iq, **HEADS)
            else:
                norm_rope_nvfp4_insert(out, *ins, q, iq, **HEADS)

        def stock(j, off):
            w, ws = raw[(j + off) % copies]
            qkv = _mxfp8_dot_scaled_gemm(*tri_in[j], w, ws, torch.bfloat16)
            minimax_m3_norm_rope_nvfp4_insert(qkv, *ins, q, iq, **HEADS)

        every = dict(entry=entry, deferred=deferred, stock=stock)
        arms = {name: every[name] for name in args.arms.split(",")}
        kv, idx = ins[8], ins[9]

        def snapshot():
            torch.cuda.synchronize()
            return q.clone(), iq.view(torch.uint8).clone(), kv.clone(), idx.clone()

        checks = {}
        for j in (0, count - 1):
            kv.zero_(), idx.zero_()
            deferred(j, 0)
            want = snapshot()
            for name, call in arms.items():
                kv.zero_(), idx.zero_()
                q.fill_(7)
                call(j, 0)
                got = snapshot()
                diff = [int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) for a, b in zip(got, want)]
                checks.setdefault(name, []).append(diff)
                if name == "entry" and not tile:  # the dispatcher allocates its own q / index-q
                    diff = diff[2:]
                if name != "stock":
                    assert not any(diff), (name, m, diff)
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
            B.measure(graphs[name], count)
        samples = {name: [] for name in arms}
        # the amd_kernels arms alternate, order rotated every round; the stock arm runs alone afterwards (interleaved,
        # the next arm after it reads 0.5-0.9 us slow at 256-768 rows)
        names = [name for name in arms if name != "stock"]
        for r in range(15):
            for name in names[r % len(names):] + names[:r % len(names)]:
                samples[name].append(B.measure(graphs[name], count))
        if "stock" in arms:
            for _ in range(15):
                samples["stock"].append(B.measure(graphs["stock"], count))
        row = dict(m=m, plan=str(host._plan(m)) if m <= host.FUSED_MAX_ROWS else "deferred chain",
                   roofline_us=roofline_us(m), checks=checks)
        for name in arms:
            row[name] = statistics.median(samples[name])
            row[name + "_range"] = [min(samples[name]), max(samples[name])]
        rows.append(row)
        print(json.dumps(row), flush=True)
        del graphs, akr_in, tri_in
        torch.cuda.empty_cache()
    names = list(arms)
    print("| M | " + " | ".join(f"{n} us" for n in names) + " | roofline us | entry % of roofline |")
    print("|---|" + "---|" * (len(names) + 2))
    for r in rows:
        cells = " | ".join(f"{r[n]:.2f}" for n in names)
        frac = f"{100 * r['roofline_us'] / r['entry']:.0f}%" if "entry" in r else "-"
        print(f"| {r['m']} | {cells} | {r['roofline_us']:.2f} | {frac} |")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
