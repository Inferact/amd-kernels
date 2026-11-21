# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Timeline of the one-launch path: start / end of every prologue, walk and output workgroup (``s_memrealtime``, 100 MHz) and,
for one sequence, when the walk of head 0 enters each chunk and when each chunk's record is published.

Inputs rotate over eight cases so the traffic is cold; the last of ``--calls`` calls is reported.

  python3 benchmarks/attention/kda_prefill_h12_d128/benchmark_timeline.py --lens 4096
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from amd_kernels.attention.kda_prefill_h12_d128 import host
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R
from amd_kernels.attention.kda_prefill_h12_d128 import walk as W

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.attention.kda_prefill_h12_d128.inputs import make_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="4096")
    ap.add_argument("--calls", type=int, default=10)
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]
    cases = [make_case(lens, seed=s) for s in range(8)]
    cu = cases[0]["cu_seqlens"]
    table, offs = R.chunk_table(cu)
    ci = torch.tensor([[n, k] for n, k, _, _ in table], dtype=torch.int32, device="cuda")
    co = torch.tensor(offs, dtype=torch.int64, device="cuda")
    nc = ci.shape[0]
    out = torch.zeros(sum(lens), R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    ts = torch.zeros(1 << 22, dtype=torch.int32, device="cuda")
    nwg = min(((len(lens) * R.NH + 7) // 8) * W.UNIT, host.MAX_WALK_WGS)
    for i in range(args.calls):
        c = cases[i % 8]
        ts.zero_()
        host.prefill_mega(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["conv_weight"],
                          c["conv_state"].clone(), c["state_cache"].clone(), c["state_indices"], c["has_initial_state"], cu, ci, co,
                          c["norm_weight"], c["norm_eps"], out, c["lower_bound"], timestamps=ts)
        torch.cuda.synchronize()
    t = ts.cpu().numpy().astype(np.int64) & 0xFFFFFFFF
    grid = nwg + 2 * nc * R.NH
    rec = t[:grid * 4].reshape(grid, 4)
    t0 = rec[:, 2].min()
    rec[:, 2] -= t0
    rec[:, 3] -= t0
    us = lambda x: x / 100.0
    print(f"lens {lens}: {nc} chunks, {nwg} walk workgroups")
    for role, name in ((0, "walk"), (1, "prologue"), (2, "output")):
        r = rec[:nwg] if role == 0 else rec[rec[:, 0] == role]
        r = r[r[:, 3] > 0]
        print(f"{name:>9}: {len(r):5d} workgroups, start {us(r[:, 2].min()):6.1f}..{us(r[:, 2].max()):6.1f} us, "
              f"end {us(r[:, 3].min()):6.1f}..{us(r[:, 3].max()):6.1f} us, mean duration {us((r[:, 3] - r[:, 2]).mean()):5.1f} us")
    if len(lens) == 1:
        step = max(1, nc // 16)
        ct = t[1 << 20:(1 << 20) + R.NH * nc * W.WALK_GROUPS].reshape(R.NH, nc, W.WALK_GROUPS)[0, :, 0].astype(np.int64) - t0
        print("walk enters chunk (head 0, every %d-th), us:" % step, [round(float(x) / 100, 1) for x in ct[::step]])
        p = rec[rec[:, 0] == 1]
        p = p[np.argsort(p[:, 1])]
        print("prologue record done (head 0), us:", [round(float(x) / 100, 1) for x in p[::R.NH, 3][::step]])


if __name__ == "__main__":
    main()
