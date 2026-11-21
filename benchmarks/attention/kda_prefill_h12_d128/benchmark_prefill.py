# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of the KDA prefill of one layer (conv + prologue, chunk scan, gated RMSNorm) against the stock sequence.

One HIP graph holds one launch per input set (the sets together are larger than the 256 MB MALL), min of 15 rounds in us per layer call. ``stock`` is what vLLM runs with kda_prefill_backend=fused: AITER ``causal_conv1d_split_qkv_triton_tile_fn``,
``torch.ops._C.fused_kda_prologue`` / ``fused_kda_chunk`` and the Triton ``FusedRMSNormGated``. Roofline columns: the layer's
unavoidable HBM traffic (reads of the projection bands, gate and beta, the gate for the norm, the output write, state and
conv state) at the measured ceilings (read 6.5 TB/s, write 6.0 TB/s), the same plus this implementation's workspace
write and re-reads, and the MFMA time of the chunk matrices at the measured bf16 peak (1.98 PF, 1.5 PF library).

  python3 benchmarks/attention/kda_prefill_h12_d128/benchmark_prefill.py --tokens 2048,4096,10240,16384
"""

import argparse
import functools
import statistics
import sys
import types
from pathlib import Path

import torch

from amd_kernels.attention.kda_prefill_h12_d128 import host
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.attention.kda_prefill_h12_d128.inputs import make_case

HBM_READ = 6.5e12
HBM_WRITE = 6.0e12
MFMA_PEAK = 1.98e15
MIN_SET_BYTES = 800_000_000


def chunk_indices(cu):
    table, offs = R.chunk_table(cu)
    ci = torch.tensor([[n, k] for (n, k, _, _) in table], dtype=torch.int32, device=cu.device)
    return ci, torch.tensor(offs, dtype=torch.int64, device=cu.device)


def stock_arm(c, cu, ci, co):
    import vllm.platforms.rocm  # noqa: F401  (registers torch.ops._C.fused_kda_prologue / fused_kda_chunk)
    from aiter.ops.triton.gated_delta_net import (
        causal_conv1d_split_qkv_triton_tile_fn as conv_fn,
    )
    from vllm.models.kimi_k3.amd.ops.kda_chunk import (
        fused_kda_chunk,
        fused_kda_prologue,
    )
    from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

    t = c["mixed"].shape[0]
    w2 = c["conv_weight"].permute(0, 2, 1).reshape(3 * R.LP, 4).contiguous()
    cs = c["conv_state"].clone()
    state = c["state_cache"].clone()
    meta = types.SimpleNamespace(nums_dict={})
    out = torch.empty(1, t, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    gate = c["gate"]

    def run():
        q, k, v = conv_fn(c["mixed"].transpose(0, 1), w2, None, cs, cu, R.LP, R.LP, cache_indices=c["state_indices"],
                          has_initial_state=c["has_initial_state"], activation="silu", block_m=128, metadata=meta, preserve_input_dtype=True)
        view = lambda x: x.view(1, t, R.NH, R.D)
        ws = fused_kda_prologue(q=view(q), k=view(k), v=view(v), raw_g=c["raw_g"].view(1, t, R.NH, R.D),
                                raw_beta=c["beta_logit"].view(1, t, R.NH), A_log=c["A_log"], dt_bias=c["dt_bias"], scale=R.SCALE,
                                lower_bound=c["lower_bound"], cu_seqlens=cu, chunk_indices=ci)
        fused_kda_chunk(qg=ws["qg"], w=ws["w"], u=ws["u"], kg_t=ws["kg_t"], aqk=ws["aqk"], decay=ws["decay"], out=out, scale=R.SCALE,
                        cu_seqlens=cu, initial_state=None, output_final_state=False, chunk_offsets=co, state_cache=state,
                        state_indices=c["state_indices"], has_initial_state=c["has_initial_state"])
        out.copy_(rms_norm_gated(out, gate, c["norm_weight"], None, "sigmoid", eps=c["norm_eps"]))

    return run


def fused_arm(c, cu, ci, co, counts, impl=None):
    table = host._prologue.chunk_table(cu, ci, c["state_indices"], c["has_initial_state"])
    cs = c["conv_state"].clone()
    state = c["state_cache"].clone()
    out = torch.empty(c["mixed"].shape[0], R.NH, R.D, dtype=torch.bfloat16, device="cuda")

    def run():
        host.prefill(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["conv_weight"], cs, state,
                     c["state_indices"], c["has_initial_state"], cu, ci, co, c["norm_weight"], c["norm_eps"], out, c["lower_bound"],
                     table=table, chunk_counts=counts, impl=impl)

    return run


def stage_arms(c, cu, ci, co, counts):
    """(fused stages, stock stages): [(name, closure)] sharing one workspace / output so each stage runs on its real input."""
    import vllm.platforms.rocm  # noqa: F401
    from aiter.ops.triton.gated_delta_net import (
        causal_conv1d_split_qkv_triton_tile_fn as conv_fn,
    )
    from vllm.models.kimi_k3.amd.ops.kda_chunk import (
        fused_kda_chunk,
        fused_kda_prologue,
    )
    from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

    t = c["mixed"].shape[0]
    cs, state = c["conv_state"].clone(), c["state_cache"].clone()
    out = torch.empty(t, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    ws = {}
    table = host._prologue.chunk_table(cu, ci, c["state_indices"], c["has_initial_state"])

    def f_pro():
        ws["ws"] = host._prologue.prologue(c["mixed"], c["raw_g"], c["beta_logit"], c["A_log"], c["dt_bias"], c["lower_bound"], R.SCALE,
                                           c["conv_weight"], cs, c["state_indices"], c["has_initial_state"], cu, ci, table=table)

    f_pro()
    fused = [("conv + prologue", f_pro),
             ("scan", lambda: host.chunk_scan(ws["ws"], cu, co, state, c["state_indices"], c["has_initial_state"], out, counts)),
             ("gated norm", lambda: host.gated_norm(out, c["gate"], c["norm_weight"], c["norm_eps"], out))]
    w2 = c["conv_weight"].permute(0, 2, 1).reshape(3 * R.LP, 4).contiguous()
    cs2, state2 = c["conv_state"].clone(), c["state_cache"].clone()
    meta = types.SimpleNamespace(nums_dict={})
    sws, qkv = {}, {}
    out2 = torch.empty(1, t, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    view = lambda x: x.view(1, t, R.NH, R.D)

    def s_conv():
        qkv["qkv"] = conv_fn(c["mixed"].transpose(0, 1), w2, None, cs2, cu, R.LP, R.LP, cache_indices=c["state_indices"],
                             has_initial_state=c["has_initial_state"], activation="silu", block_m=128, metadata=meta, preserve_input_dtype=True)

    def s_pro():
        q, k, v = qkv["qkv"]
        sws["ws"] = fused_kda_prologue(q=view(q), k=view(k), v=view(v), raw_g=c["raw_g"].view(1, t, R.NH, R.D),
                                       raw_beta=c["beta_logit"].view(1, t, R.NH), A_log=c["A_log"], dt_bias=c["dt_bias"], scale=R.SCALE,
                                       lower_bound=c["lower_bound"], cu_seqlens=cu, chunk_indices=ci)

    def s_chunk():
        w = sws["ws"]
        fused_kda_chunk(qg=w["qg"], w=w["w"], u=w["u"], kg_t=w["kg_t"], aqk=w["aqk"], decay=w["decay"], out=out2, scale=R.SCALE,
                        cu_seqlens=cu, initial_state=None, output_final_state=False, chunk_offsets=co, state_cache=state2,
                        state_indices=c["state_indices"], has_initial_state=c["has_initial_state"])

    s_conv()
    s_pro()
    stock = [("conv", s_conv), ("prologue", s_pro), ("chunk", s_chunk),
             ("gated norm", lambda: out2.copy_(rms_norm_gated(out2, c["gate"], c["norm_weight"], None, "sigmoid", eps=c["norm_eps"])))]
    return fused, stock


def skeleton_scan_arms(cases, cu, co, counts):
    """The chunk scan with the same DMAs, loads, stores and launches but no math (the scan's empirical floor)."""
    fns = []
    for c in cases:
        ws = torch.zeros(R.NH, int(co[-1]), R.RECORD_BYTES, dtype=torch.uint8, device="cuda")
        state = c["state_cache"].clone()
        out = torch.empty(c["mixed"].shape[0], R.NH, R.D, dtype=torch.bfloat16, device="cuda")
        fns.append(lambda ws=ws, state=state, out=out, c=c: host.chunk_scan(
            ws, cu, co, state, c["state_indices"], c["has_initial_state"], out, counts))
    return fns


def skeleton_patch():
    import importlib

    scan_mod = importlib.import_module("amd_kernels.attention.kda_prefill_h12_d128.scan")
    host._walk_launcher = functools.cache(
        lambda ckpt=False, foldwalk=False: scan_mod.compile_scan(host.WALK_WAVES, host.WALK_NVT, False, host.WALK_STAGES, ckpt, True,
                                                                  foldwalk))
    host._chain_launcher = functools.cache(
        lambda: scan_mod.compile_scan(host.CHAIN_WAVES, host.CHAIN_NVT, True, host.CHAIN_STAGES, False, True))


def timeit_graph(calls, replays=10, rounds=15, warmup=3):
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(warmup):
        graph.replay()
    for start, end in events:
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
    torch.cuda.synchronize()
    per = [s.elapsed_time(e) * 1e3 / (replays * len(calls)) for s, e in events]
    return min(per), statistics.median(per)


def floors(tokens, chunks):
    """(unavoidable us, with workspace us, mfma us) of the layer."""
    lp = R.LP
    reads = tokens * (3 * lp * 2 + lp * 2 + R.NH * 2 + lp * 2)
    writes = tokens * lp * 2
    state = R.NH * R.D * R.D * 4 * 2
    essential = (reads + state) / HBM_READ + writes / HBM_WRITE
    ws = chunks * R.NH * R.RECORD_BYTES
    with_ws = essential + ws / HBM_WRITE + 2 * ws / HBM_READ
    flops = chunks * R.NH * (12.5e6)
    return essential * 1e6, with_ws * 1e6, flops / MFMA_PEAK * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="2048,4096,10240,16384")
    ap.add_argument("--seqs", type=int, default=1)
    ap.add_argument("--lens", default="", help="ragged batch, e.g. 3000,1500,700,60 (replaces --tokens)")
    ap.add_argument("--no-stock", action="store_true")
    ap.add_argument("--stages", action="store_true", help="also time each stage alone (fused and stock)")
    ap.add_argument("--skeleton", action="store_true", help="also time the scan's no-math skeleton (run last: it patches the launchers)")
    args = ap.parse_args()
    print(f"{'T':>6} {'seqs':>4} {'path':>6} {'v2 us':>8} {'staged':>8} {'stock':>8} {'v2/stock':>9} {'v2/staged':>10} {'io floor':>9} {'%io':>5} "
          f"{'+ws floor':>10} {'%+ws':>5} {'mfma us':>8}")
    batches = [[int(x) for x in args.lens.split(",")]] if args.lens else None
    for k, tokens in enumerate(sum(b) for b in batches) if batches else enumerate(int(x) for x in args.tokens.split(",")):
        if batches:
            lens = batches[k]
        else:
            lens = [tokens // args.seqs] * args.seqs
            lens[-1] += tokens - sum(lens)
        c0 = make_case(lens)
        cu = c0["cu_seqlens"]
        ci, co = chunk_indices(cu)
        counts = [-(-n // 64) for n in lens]
        set_bytes = tokens * (R.LP * 2 * 5 + 6288 * 2)
        copies = max(2, min(12, -(-MIN_SET_BYTES // set_bytes)))
        cases = [c0] + [make_case(lens, seed=i + 1) for i in range(copies - 1)]
        path = "mega" if host.uses_mega(counts) else "staged"
        v2 = timeit_graph([fused_arm(c, cu, ci, co, counts) for c in cases])[0]
        staged = timeit_graph([fused_arm(c, cu, ci, co, counts, "v1") for c in cases])[0] if path == "mega" else v2
        stock = float("nan")
        if not args.no_stock:
            stock = timeit_graph([stock_arm(c, cu, ci, co) for c in cases])[0]
        io, with_ws, mfma = floors(tokens, int(ci.shape[0]))
        if args.skeleton:
            skeleton_patch()
            sk = timeit_graph(skeleton_scan_arms(cases, cu, co, counts))[0]
            print(f"    scan skeleton us (min): {sk:.1f}")
        if args.stages:
            fs, ss = [], []
            sets = [stage_arms(c, cu, ci, co, counts) for c in cases]
            for k, (name, _) in enumerate(sets[0][0]):
                fs.append((name, timeit_graph([a[0][k][1] for a in sets])[0]))
            for k, (name, _) in enumerate(sets[0][1]):
                ss.append((name, timeit_graph([a[1][k][1] for a in sets])[0]))
            print("    staged stages us (min): " + ", ".join(f"{n} {v:.1f}" for n, v in fs))
            print("    stock stages us (min): " + ", ".join(f"{n} {v:.1f}" for n, v in ss))
        print(f"{tokens:>6} {len(lens):>4} {path:>6} {v2:>8.1f} {staged:>8.1f} {stock:>8.1f} {stock / v2:>8.2f}x {staged / v2:>9.2f}x {io:>9.1f} "
              f"{100 * io / v2:>4.0f}% {with_ws:>10.1f} {100 * with_ws / v2:>4.0f}% {mfma:>8.1f}", flush=True)


if __name__ == "__main__":
    main()
