# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The small-tile crossovers (``host._local_tile``): every fp16 tile forced at one token count and KV head count, timed
as ``benchmark_sparse_prefill.py`` times (graph replay, 100 cold inputs, 16K prefix, plain bf16 output), plus the
dispatch's own choice (``auto``). The first input is checked against ``reference.py``.

    python3 benchmarks/attention/nvfp4_sparse_prefill_gqa16_d128_p128_top16/sweep_tiles.py --hkv 2 --tokens 128,129,256
"""

import argparse
import statistics

import numpy as np
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import host
from benchmarks.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.benchmark_sparse_prefill import (
    check,
    make_inputs,
    n_inputs,
    timeit_graph,
)
from tests.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.cases import SM_SCALE

# tq1 (eager), tq2 / shared-Q tq2 / tq3 (lazy), the four-query kernel one round (eager) and two rounds (lazy)
TILES = {
    "tq1": lambda: host._get_bf16_local(1, False, False, True),
    "tq2": lambda: host._get_bf16_local(2, False, False, True, lazy=True),
    "tq2_qlds": lambda: host._get_bf16_local(2, False, False, True, lazy=True, q_shared=True),
    "tq3": lambda: host._get_bf16_local(3, False, False, True, lazy=True),
    "tq4_r1": lambda: host._get_kernel("fp16", False, 1, False, sched=False),
    "tq4_r2": lambda: host._get_kernel("fp16", False, 2, False, sched=False, lazy=True),
}


def forced(exe, inp, cache, req, pos, hkv, out):
    """One call of ``exe`` with ``nvfp4_sparse_prefill``'s plain-output arguments."""
    q, unit = inp["q"], host._unit_scale(inp["q"].device)
    host._run_compiled(
        exe, q.data_ptr(), cache.data_ptr(), inp["topk"].data_ptr(), inp["table"].data_ptr(), req.data_ptr(),
        pos.data_ptr(), out.data_ptr(), unit.data_ptr(), unit.data_ptr(), 0, q.shape[0], q.shape[1], hkv,
        inp["topk"].stride(0), inp["table"].stride(0), cache.stride(0),
        float(np.float32(SM_SCALE) * host.LOG2E_F32), torch.cuda.current_stream(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hkv", type=int, default=2)
    ap.add_argument("--tokens", default="64,128,129,256,257,384,385,512,513,768,769,1024")
    ap.add_argument("--tiles", default=",".join(TILES) + ",auto")
    ap.add_argument("--inputs", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2, help="alternated rounds")
    args = ap.parse_args()
    hkv, names = args.hkv, args.tiles.split(",")
    print("hkv | tokens | " + " | ".join(f"{v} us" for v in names) + " | fastest", flush=True)
    for t in (int(x) for x in args.tokens.split(",")):
        cache, pos, req, inputs = make_inputs(hkv, t, n_inputs(hkv, t, len(names), args.inputs))
        calls = {}
        for v in names:
            calls[v] = []
            for i, inp in enumerate(inputs):
                out = torch.empty_like(inp["q"])
                if v == "auto":
                    f = (lambda inp=inp, out=out: host.nvfp4_sparse_prefill(
                        inp["q"], cache, inp["topk"], inp["table"], req, pos, hkv, SM_SCALE, out,
                        lazy_rescale=True))
                else:
                    exe = TILES[v]()
                    f = (lambda inp=inp, out=out, exe=exe: forced(exe, inp, cache, req, pos, hkv, out))
                if i == 0 and t >= 8:  # the reference's sampled rows need a few tokens
                    f()
                    check(cache, pos, req, inp, hkv, "fp16", out)
                calls[v].append(f)
        times = {v: [] for v in names}
        for r in range(args.rounds):
            for v in (names if r % 2 == 0 else names[::-1]):
                times[v].append(timeit_graph(calls[v]) * 1e3)
        med = {v: statistics.median(times[v]) for v in names}
        best = min((v for v in names if v != "auto"), key=med.get)
        print(f"{hkv} | {t} | " + " | ".join(f"{med[v]:.2f}" for v in names) + f" | {best}", flush=True)
        del calls, cache, inputs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
