# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of the eight-wave tile for every (weight, activation) ring depth the 160 KB LDS holds: the sweep that
fixed large.RINGS. One M tile of 32 mt rows per variant, 100 distinct activation sets.
Run: python3 benchmarks/gemm/bf16_n129280_k5120/sweep_rings.py [mt ...]
"""
import sys

import torch

from amd_kernels.gemm.bf16_n129280_k5120 import large, shuffle_weight
from amd_kernels.gemm.bf16_n129280_k5120.host import _run
from amd_kernels.gemm.bf16_n129280_k5120.reference import K, N

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from benchmark_gemm import capture, replay_us  # noqa: E402


def main():
    mts = [int(v) for v in sys.argv[1:]] or list(large.RINGS)
    dev = torch.device("cuda")
    ws = shuffle_weight((torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16))
    best = dict(large.RINGS)
    for mt in mts:
        m = 32 * mt
        xs = [torch.randn(m, K, device=dev).to(torch.bfloat16) for _ in range(100)]
        out = torch.empty(m, N, dtype=torch.bfloat16, device=dev)
        for rw in range(2, 5):
            for rx in range(2, 6):
                if rw * large.NR * 2048 + rx * 2 * mt * 2048 > 160 * 1024:
                    continue
                large.RINGS[mt] = (rw, rx)
                exe = large.compile_large_bf16_gemm_n129280_k5120(mt, 1)
                graph = capture([lambda x=x: _run(exe, x.data_ptr(), ws.data_ptr(), out.data_ptr(), m, 1,
                                                  torch.cuda.current_stream()) for x in xs])
                print(f"mt {mt} ring (w {rw}, x {rx}): {min(replay_us(graph, len(xs)) for _ in range(3)):.1f} us",
                      flush=True)
        large.RINGS[mt] = best[mt]


if __name__ == "__main__":
    main()
