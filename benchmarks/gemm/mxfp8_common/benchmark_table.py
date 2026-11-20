# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of the table's rows of one (N, K) against vLLM's Triton ``tl.dot_scaled`` MXFP8 linear and
the roofline, weights HBM-cold (``sweep_rows.Bench``: rotating copies above the 256 MB MALL).

Both sides start from the bf16 activation: ``mxfp8_linear`` (the row's quantizer + GEMM) and vLLM's
``rocm_native._mxfp8_dot_scaled_linear`` (its quantizer + GEMM with ``_M31_TILES`` / ``_select_cfg``, row-major
weights). Roofline: max(bytes / 6.5 TB/s, flops / 5 PF), bytes = weight codes + scales, bf16 activation, bf16 output.
Needs vLLM on the path (``PYTHONPATH=<vllm>``).

    python3 benchmarks/gemm/mxfp8_common/benchmark_table.py --shape 3072,6144 --ms 1,8,64,256,1024,8192
"""

import argparse

from benchmarks.gemm.mxfp8_common.sweep_rows import Bench, _key, roofline_us

from amd_kernels.gemm.mxfp8_common import gemm as G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True)
    p.add_argument("--ms", default="1,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768")
    p.add_argument("--no-baseline", action="store_true")
    args = p.parse_args()
    n, k = map(int, args.shape.split(","))
    bench = Bench(n, k, not args.no_baseline)
    print(f"N {n} x K {k}, {bench.slots} weight copies")
    print("| M | row | us | Triton dot_scaled us | vs Triton | roofline us | of roofline |")
    print("|---|---|---|---|---|---|---|")
    for m in map(int, args.ms.split(",")):
        bench.set_m(m)
        us = bench.time(None, "linear", fast=False)
        base = None if args.no_baseline else bench.time("baseline", "linear", fast=False)
        roof = roofline_us(m, n, k)
        vs = "" if base is None else f"{(us / base - 1) * 100:+.0f}%"
        print(f"| {m} | {_key(G._select_config(m, n, k))} | {us:.2f} | {'' if base is None else f'{base:.2f}'} | {vs} "
              f"| {roof:.2f} | {roof / us * 100:.0f}% |", flush=True)


if __name__ == "__main__":
    main()
