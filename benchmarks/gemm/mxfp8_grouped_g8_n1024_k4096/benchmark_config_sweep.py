# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The sweep behind ``host._ROWS``: graph-replay time of kernel configs (``kernel.Config``) at row counts, one JSON
line per (config, rows), with the cosine against AITER's batched GEMM. Configs come from ``--configs <file>`` (one
``Config(...)`` per line) or ``--random N`` seeded draws over tile, waves, stages, split K, B path and XCD order;
a config the kernel cannot run is skipped. Weights and inputs as ``benchmark_grouped_gemm.py``.
Run: python3 benchmarks/gemm/mxfp8_grouped_g8_n1024_k4096/benchmark_config_sweep.py --rows 16,192,768 --random 200
"""

import argparse
import json
import random

import torch
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.gemm.mxfp8_common.reference import dequantize_mxfp8
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096.kernel import Config, check_config, compile_grouped_mxfp8
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096.reference import BLOCK, G, N
from benchmark_grouped_gemm import NIN, NW, graph, inputs, timed, weights


def random_configs(count, seed):
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < count:
        splits = rng.choice([1, 1, 2, 4, 8])
        direct = splits == 1 and rng.random() < 0.4
        config = Config(
            rng.choice([16, 32, 48, 64, 96, 128, 192, 256]),
            rng.choice([32, 64, 128, 256]),
            rng.choice([128, 256, 512]),
            *rng.choice([(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1), (2, 4), (4, 2)]),
            rng.choice([2, 3, 4, 5]),
            splits,
            direct,
            rng.choice([1, 2, 3]) if direct else 1,
            rng.random() < 0.5,
            splits == 1 and rng.random() < 0.6,
            0,
            rng.random() < 0.8,
        )
        try:
            check_config(config)
        except ValueError:
            continue
        if config not in seen:
            seen.add(config)
            out.append(config)
    return out


def time_config(config, m, xs, ss, ws):
    dev = xs[0].device
    exe = compile_grouped_mxfp8(config)
    tiles = -(-m // config.tile_m) * (N // config.tile_n) * G
    floats = max(1, tiles * config.splits * config.tile_m * config.tile_n)
    partials = torch.empty(floats, dtype=torch.float32, device=dev)
    counters = torch.zeros(1 << 16, dtype=torch.int32, device=dev)
    out = torch.empty(m, G * N, dtype=torch.bfloat16, device=dev)
    codes = torch.empty(m, G * N, dtype=torch.uint8, device=dev)
    scales = torch.empty(m, G * N // BLOCK, dtype=torch.uint8, device=dev)

    def call(i):
        w, s = ws[i % NW]
        _run_compiled(
            exe, out.data_ptr(), codes.data_ptr(), scales.data_ptr(), xs[i].data_ptr(), w.data_ptr(), ss[i].data_ptr(),
            s.data_ptr(), partials.data_ptr(), counters.data_ptr(), m, torch.cuda.current_stream(),
        )

    call(0)
    ref = batched_gemm_a8w8_mxscale_bpreshuffle(xs[0], ws[0][0], ss[0], ws[0][1], torch.bfloat16).view(m, -1).float()
    got = dequantize_mxfp8(codes.view(torch.float8_e4m3fn), scales) if config.mxfp8_out else out.float()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    return timed(graph(lambda: [call(i) for i in range(NIN)])), cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--configs")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    configs = [eval(line) for line in open(args.configs) if line.strip()] if args.configs else []
    configs += random_configs(args.random, args.seed)
    dev = torch.device("cuda", 0)
    ws = weights(dev)
    data = {m: inputs(m, dev) for m in map(int, args.rows.split(","))}
    for config in configs:
        try:
            check_config(config)
        except ValueError:
            continue
        for m, (xs, ss) in data.items():
            t, cos = time_config(config, m, xs, ss, ws)
            print(json.dumps({"rows": m, "us": round(t, 3), "cos": round(cos, 6), "config": config._asdict()}), flush=True)


if __name__ == "__main__":
    main()
