# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of the KDA prefill prologue against the stock sequence and its HBM floor.

One HIP graph holds ``--copies`` launches over distinct inputs (the set is larger than the 256 MB MALL), median and min of 15
rounds in us per launch. Arms: ``fused`` (this kernel), ``skeleton`` (``skeleton.py``: the same loads and stores, no math)
and ``stock`` (AITER ``causal_conv1d_split_qkv_triton_tile_fn`` + ``torch.ops._C.fused_kda_prologue``). Roofline columns: HBM
bytes (reads of the projection bands, gate and beta; workspace writes) at the measured read / write ceilings.

  python3 benchmarks/attention/kda_prefill_h12_d128/benchmark_prologue.py --tokens 2048,4096,10240,16384
"""

import argparse
import statistics
import sys
import types
from pathlib import Path

import torch

from amd_kernels.attention.kda_prefill_h12_d128 import prologue as P
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.attention.kda_prefill_h12_d128.inputs import make_case  # noqa: E402

HBM_READ = 6.5e12
HBM_WRITE = 6.0e12
MIN_SET_BYTES = 800_000_000


def chunk_indices(cu):
    table, _ = R.chunk_table(cu)
    return torch.tensor([[n, k] for (n, k, _, _) in table], dtype=torch.int32, device=cu.device)


def stock_arm(c, cu, ci):
    import vllm.platforms.rocm  # noqa: F401  (registers torch.ops._C.fused_kda_prologue)
    from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_triton_tile_fn as conv_fn
    from vllm.models.kimi_k3.amd.ops.kda_chunk import fused_kda_prologue

    T = c["mixed"].shape[0]
    w2 = c["conv_weight"].permute(0, 2, 1).reshape(3 * R.LP, 4).contiguous()
    cs = c["conv_state"].clone()
    meta = types.SimpleNamespace(nums_dict={})

    def run():
        q, k, v = conv_fn(c["mixed"].transpose(0, 1), w2, None, cs, cu, R.LP, R.LP, cache_indices=c["state_indices"],
                          has_initial_state=c["has_initial_state"], activation="silu", block_m=128, metadata=meta, preserve_input_dtype=True)
        view = lambda x: x.view(1, T, R.NH, R.D)
        return fused_kda_prologue(q=view(q), k=view(k), v=view(v), raw_g=c["raw_g"].view(1, T, R.NH, R.D),
                                  raw_beta=c["beta_logit"].view(1, T, R.NH), A_log=c["A_log"], dt_bias=c["dt_bias"],
                                  scale=R.SCALE, lower_bound=c["lower_bound"], cu_seqlens=cu, chunk_indices=ci)

    return run


def fused_arm(c, cu, ci, ws, fn):
    cs = c["conv_state"].clone()
    table = P.chunk_table(cu, ci, c["state_indices"], c["has_initial_state"])
    return lambda: fn(c["mixed"], c["raw_g"], c["beta_logit"], c["A_log"], c["dt_bias"], c["lower_bound"], R.SCALE, c["conv_weight"],
                      cs, c["state_indices"], c["has_initial_state"], cu, ci, ws=ws, table=table)


def graph_time(calls, rounds=15, replays=5):
    for f in calls:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(replays):
            g.replay()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / replays / len(calls))
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="2048,4096,10240,16384")
    ap.add_argument("--ragged", default="")
    ap.add_argument("--arms", default="fused,stock")
    a = ap.parse_args()
    arms = a.arms.split(",")
    batches = [[int(t)] for t in a.tokens.split(",") if t] + ([[int(t) for t in a.ragged.split(",")]] if a.ragged else [])
    print(f"{'tokens':>22} {'arm':>9} {'us':>8} {'min us':>8} {'GB/s':>8} {'floor us':>9} {'% of floor':>10}")
    for seq in batches:
        T = sum(seq)
        per_copy = T * (R.LP * 3 * 2 + R.LP * 2 + 24 + R.NH * R.D * 2) + len(seq) * 0
        copies = max(2, -(-MIN_SET_BYTES // (per_copy + T // 64 * R.NH * R.RECORD_BYTES)))
        cases = [make_case(seq, seed=i) for i in range(copies)]
        cis = [chunk_indices(c["cu_seqlens"]) for c in cases]
        nc = cis[0].shape[0]
        reads = T * (3 * R.LP * 2 + R.LP * 2 + 24)
        writes = nc * R.NH * R.RECORD_BYTES
        floor = reads / HBM_READ * 1e6 + writes / HBM_WRITE * 1e6
        for arm in arms:
            if arm == "fused":
                wss = [torch.empty(R.NH, nc, R.RECORD_BYTES, dtype=torch.uint8, device="cuda") for _ in range(copies)]
                calls = [fused_arm(c, c["cu_seqlens"], ci, ws, P.prologue) for c, ci, ws in zip(cases, cis, wss)]
                traffic = reads + writes
            elif arm == "skeleton":
                from skeleton import skeleton_prologue

                wss = [torch.empty(R.NH, nc, R.RECORD_BYTES, dtype=torch.uint8, device="cuda") for _ in range(copies)]
                calls = [fused_arm(c, c["cu_seqlens"], ci, ws, skeleton_prologue) for c, ci, ws in zip(cases, cis, wss)]
                traffic = reads + writes
            else:
                calls = [stock_arm(c, c["cu_seqlens"], ci) for c, ci in zip(cases, cis)]
                traffic = reads * 2 + T * 3 * R.LP * 2 + writes
            med, mn = graph_time(calls)
            print(f"{'+'.join(map(str, seq))[:22]:>22} {arm:>9} {med:8.1f} {mn:8.1f} {traffic / (mn * 1e-6) / 1e9:8.0f} {floor:9.1f} {floor / mn * 100:9.0f}%")
        del cases


if __name__ == "__main__":
    main()
