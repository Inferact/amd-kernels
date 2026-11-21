# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of the fused DCP8 combine against vLLM's v_up-first sequence (mask LSE, AITER LSE all-gather, bmm_vup,
scale, AITER reduce-scatter; plus the Triton output gate with ``--gate``) on 8 GPUs: inputs rotating over > 300 MB, MAX
over ranks, median and min of 9 replays, us per call, and a table against the measured ceiling model.

    python3 benchmarks/attention/dcp8_lse_combine_vup_mla_h12_kv512/benchmark_combine.py [--gate] [--tokens 12,24,...]

Ceiling model (constants measured on MI355X): launch 1.6 us + first-tile load (0.25 us + (W_UV + first
16-token tile) / 6.4 TB/s HBM read) + max(flag one-way 0.8 us + bytes sent / 343 GB/s all-to-all egress, remaining x /
6.4 TB/s) + combine tail 0.6 us.
"""
import argparse
import contextlib
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
CALLS = 16
ROUNDS = 9
W_BYTES = 96 * 128 * 512 * 2
HBM_READ = 6.4e12
EGRESS = 343e9


def ceiling_us(m):
    x_tile = 16 * 96 * 512 * 2
    x_rest = max(0, m * 96 * 512 * 2 - x_tile)
    sent = 7 * m * 12 * 260
    first = 0.25 + (W_BYTES + min(m * 96 * 512 * 2, x_tile)) / HBM_READ * 1e6
    return 1.6 + first + max(0.8 + sent / EGRESS * 1e6, x_rest / HBM_READ * 1e6) + 0.6


def _worker(rank, port, tokens, gate, out_q):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    from amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512 import DcpVupCombine, dcp8_lse_combine_vup, prepare_w_uv
    from vllm.distributed.device_communicators.aiter_custom_all_reduce import AiterCustomAllreduce
    from vllm.models.kimi_k3.amd.dsa import _gate_mul_kernel
    from vllm.v1.attention.ops.dcp import dcp_vup_rs_combine
    import triton

    dev = torch.device("cuda", rank)
    comm = DcpVupCombine(rank, dist.group.WORLD, dev)
    aiter = AiterCustomAllreduce(group=dist.group.WORLD, device=dev)
    g = torch.Generator(device=dev).manual_seed(rank)
    rows_out = []
    for m in tokens:
        nrot = max(4, min(CALLS, -(-300 * 2**20 // (m * 96 * 1024 + W_BYTES))))
        xs = [(torch.randn((m, 96, 512), generator=g, device=dev) * 0.5).bfloat16() for _ in range(nrot)]
        lses = [torch.randn((m, 96), generator=g, device=dev) * 2 + 3 for _ in range(nrot)]
        w_uv = [(torch.randn((96, 512, 128), generator=g, device=dev) * 0.05).bfloat16() for _ in range(nrot)]
        wt = [prepare_w_uv(w) for w in w_uv]
        gates = [torch.randn((m, 1536), generator=g, device=dev).bfloat16() for _ in range(nrot)]
        outs = [torch.empty((m, 12, 128), dtype=torch.bfloat16, device=dev) for _ in range(nrot)]
        gouts = [torch.empty((m, 1536), dtype=torch.bfloat16, device=dev) for _ in range(nrot)]
        nseq = max(1, m // 2)
        qsl = torch.linspace(0, m, nseq + 1).round().int().to(dev)
        sls = torch.randint(1, 5000, (nseq,), generator=g, device=dev).int()

        def ours(i):
            dcp8_lse_combine_vup(comm, xs[i], lses[i], wt[i], gouts[i].view(m, 12, 128) if gate else outs[i], sls, qsl,
                                 gate=gates[i] if gate else None)

        def stock(i):
            dcp_vup_rs_combine(xs[i], lses[i], w_uv[i], outs[i], aiter, rank, WORLD, True, sls, qsl)
            if gate:
                _gate_mul_kernel[(triton.cdiv(m * 1536, 1024),)](outs[i], gates[i], gouts[i], m * 1536, BLOCK=1024)

        ours(0)
        stock(0)
        torch.cuda.synchronize()
        got, ref = (gouts if gate else outs)[0].float(), None
        ours_out = got.clone()
        stock(0)
        torch.cuda.synchronize()
        ref = (gouts if gate else outs)[0].float()
        cos = torch.nn.functional.cosine_similarity(ours_out.flatten(), ref.flatten(), dim=0).item()

        graphs = {}
        for name, fn, ctx in (("ours", ours, contextlib.nullcontext), ("stock", stock, aiter.capture)):
            for i in range(nrot):
                fn(i)
            torch.cuda.synchronize()
            dist.barrier()
            gr = torch.cuda.CUDAGraph()
            with ctx():
                with torch.cuda.graph(gr):
                    for c in range(CALLS):
                        fn(c % nrot)
            graphs[name] = gr
        times = {k: [] for k in graphs}
        for rd in range(ROUNDS):
            for name in (("ours", "stock") if rd % 2 == 0 else ("stock", "ours")):
                graphs[name].replay()
                torch.cuda.synchronize()
                dist.barrier()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                graphs[name].replay()
                e.record()
                torch.cuda.synchronize()
                t = torch.tensor([s.elapsed_time(e) * 1000 / CALLS])
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                times[name].append(t.item())
        rows_out.append((m, {k: (statistics.median(v), min(v)) for k, v in times.items()}, cos))
        torch.cuda.empty_cache()
    if rank == 0:
        out_q.put(rows_out)
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="12,24,40,65,80,128,192")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, port, tokens, a.gate, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    rows = q.get(timeout=900)
    for p in procs:
        p.join()
    print(f"\nDCP8 combine + W_UV{' + output gate' if a.gate else ''}: graph replay, us per call (median/min of {ROUNDS}), "
          "MAX over 8 ranks")
    print(f"{'M':>4} {'sent MB':>8} {'HBM MB':>7} {'ceiling':>8} {'stock':>13} {'fused':>13} {'speedup':>8} {'% ceiling':>9} {'cos vs stock':>12}")
    for m, t, cos in rows:
        sent = 7 * m * 12 * 260 / 1e6
        hbm = (W_BYTES + m * 96 * 1024 + m * 3072 + 2 * 8 * m * 12 * 260) / 1e6
        c = ceiling_us(m)
        print(f"{m:>4} {sent:>8.2f} {hbm:>7.1f} {c:>8.1f} {t['stock'][0]:>7.1f}/{t['stock'][1]:<5.1f} "
              f"{t['ours'][0]:>7.1f}/{t['ours'][1]:<5.1f} {t['stock'][0] / t['ours'][0]:>7.2f}x {100 * c / t['ours'][1]:>8.0f}% {cos:>12.6f}")


if __name__ == "__main__":
    main()
