# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The prefill scorer plan's crossovers (``score_prefill.host._score_plan``) at ``--heads``: the dispatch's choice and
the pinned kernel (``PIN_SHARED``: shared-dequant or the 512-row ``kernel.py``), tile rows (``PIN_MQ``) and workgroup
target (``PIN_TARGET``) of one request of M new rows after a prefix, scorer alone (graph replay, a page pool above the
MALL). Every pin's visible scores equal the first's bit for bit (asserted).

    python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/sweep_prefill_plan.py --heads 2 --cases 2048:0,1024:32768
"""

import argparse
import statistics

import torch

from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill import host
from benchmarks.indexer.indexer_nvfp4_d128_p128_top16.benchmark_index_decode_h2 import capture, time_graph
from benchmarks.indexer.indexer_nvfp4_d128_p128_top16.benchmark_indexer import page_pool, pool_pages_for, tables

# name: (PIN_SHARED, PIN_MQ, PIN_TARGET)
PLANS = {
    "auto": (None, None, None),
    "shared_mq1": (True, 1, None),
    "shared_mq2": (True, 2, None),
    "tq512": (False, None, None),
    "tq512_t1024": (False, None, 1024),
    "tq512_t4096": (False, None, 4096),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--cases", default="1024:0,2048:0,4096:0,8192:0,16384:0,1024:32768,2048:32768,4096:32768,"
                                       "8192:65536,16384:131072,4096:262144", help="M:prefix,...")
    ap.add_argument("--plans", default=",".join(PLANS))
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    from amd_kernels import warm_up

    warm_up.set_auto(False)
    heads, names = args.heads, args.plans.split(",")
    print("M | prefix | " + " | ".join(f"{p} us" for p in names) + " | fastest", flush=True)
    for case in args.cases.split(","):
        m, prefix = (int(x) for x in case.split(":"))
        msl = m + prefix
        copies = min(24, max(4, (4 << 30) // (m * heads * 128)))
        pool = pool_pages_for([msl])
        cache = page_pool(pool)
        tabs = tables([msl], copies, pool)
        qs = [(torch.randn(m, heads, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn) for _ in range(copies)]
        seq = torch.tensor([msl], dtype=torch.int32, device="cuda")
        cu = torch.tensor([0, m], dtype=torch.int32, device="cuda")
        score = torch.empty(heads, m, score_width(msl), device="cuda")
        nb = (msl + 127) // 128
        live = torch.arange(nb, device="cuda")[None, :] <= (torch.arange(prefix, msl, device="cuda") // 128)[:, None]
        us, first = {}, None
        for p in names:
            host.PIN_SHARED, host.PIN_MQ, host.PIN_TARGET = PLANS[p]
            host.nvfp4_index_score_prefill(qs[0], cache, score, tabs[0], cu, seq, 0, 1, m, msl)
            got = score[:, :, :nb][:, live].view(torch.int32).clone()
            first = got if first is None else first
            assert torch.equal(got, first), p
            g = capture([lambda i=i: host.nvfp4_index_score_prefill(qs[i], cache, score, tabs[i], cu, seq, 0, 1, m,
                                                                    msl) for i in range(copies)])
            us[p] = statistics.median(time_graph(g, copies, 3) for _ in range(args.rounds))
            del g
        host.PIN_SHARED = host.PIN_MQ = host.PIN_TARGET = None
        print(f"{m} | {prefix} | " + " | ".join(f"{us[p]:.1f}" for p in names) + f" | {min(names, key=us.get)}",
              flush=True)
        del cache, tabs, qs, score
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
