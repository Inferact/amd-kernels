# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four-GPU timing of the TP4 decode block (``index_decode.nvfp4_index_sparse_decode``: index top-16 + sparse decode
attention, the served layer call) on each route, to place ``decode_top16.SPLIT_MIN_REQS``.

One process per GPU (gloo group); rank r = card r of ``HIP_VISIBLE_DEVICES``, index head r, KV head r, as served:
``IndexDecodeTop16`` with the four-rank group on the replicated index cache, fp16 compute, the MXFP8 o_proj input from
``layer.QUANT_OUT_MIN_ROWS`` rows. Sides (the same inputs, buffers and cards):

- ``one_gpu``: ``split=False``, the one-GPU route (``fused``: scorer + the attention's in-kernel top-16);
- ``split4``: the four-GPU scorer + top-16, then the attention (``SPLIT_MIN_REQS`` set to 1 at capture).

Inputs: ``--reqs N,..``: N requests of the contexts ``F[str(N)]`` of ``--lens-json F``, a JSON file {"N": [context
length of request i of N, ...]} with an entry for every N of ``--reqs`` (for the route: the contexts requests have while
they decode, e.g. the (i + 0.5) / N quantiles of the contexts of the requests decoding at N in a workload, sampled over
time); without it, ``synthetic_lens``: request i of N at the (i + 0.5) / N quantile of a log-uniform 32K-192K
distribution. Row j of a request of context C sees C - ql + 1 + j tokens. Every rank holds the same pool of NVFP4
index pages (Gaussian keys quantized as the model does) and KV pages; each call's requests sit on distinct random pages
of their own region of the pool, and a graph holds >= 100 calls and enough that a region comes back after >= 384 MB of
other pages (cold pages, graphs captured at max_model_len 1M).

Timing: 2 warm-up replays, then ``--rounds`` rounds of ``--replays`` replays, the sides alternated (A B B A ...),
each graph after a host barrier and a small four-GPU call (device barrier); a sample = the max over the four ranks of
the per-call time. Checks: ids and attention output bytes of the sides equal on the first and last input (rows that
differ, max over ranks).

    HIP_VISIBLE_DEVICES=0,1,2,3 python3 \
        benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_index_decode_ranks4.py \
        --ql 4 --lens-json decode_contexts.json --reqs 1,2,4,8
"""

import argparse
import json
import os
import socket
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

BLK, D, H, TOPK = 128, 128, 16, 16
PAGE_BYTES = BLK * 72
KV_PAGE_BYTES = 2 * BLK * 72
INIT, LOCAL = 0, 1  # MiniMax-M3.1 sparse_init_block / sparse_local_block
MAX_MODEL_LEN = 1 << 20
MIN_POOL_BYTES = 768 << 20
REUSE_BYTES = 384 << 20
WORLD = 4
OPROJ_N, OPROJ_K = 6144, 2048  # the o_proj GEMM the MXFP8 output feeds (TP4)
SYNTHETIC_RANGE = (32768, 196608)  # the log-uniform context range without --lens-json


def synthetic_lens(n, lo=SYNTHETIC_RANGE[0], hi=SYNTHETIC_RANGE[1]):
    """Request i of n at the (i + 0.5) / n quantile of a log-uniform [lo, hi] distribution."""
    return [round(lo * (hi / lo) ** ((i + 0.5) / n)) for i in range(n)]


def cdiv(a, b):
    return -(-a // b)


def make_tables(lens, region, span, pool_slots, rng, cols):
    """[reqs, cols] int32: each request's blocks on distinct random slots of this region of the pool."""
    nb = [cdiv(s, BLK) for s in lens]
    perm = (region * span + rng.permutation(sum(nb))) % pool_slots
    t = np.zeros((len(lens), cols), dtype=np.int32)
    p = 0
    for r, n in enumerate(nb):
        t[r, :n] = perm[p : p + n]
        p += n
    return torch.from_numpy(t).cuda()


def body(rank, port, args):
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    group = dist.group.WORLD
    dev = torch.device("cuda", rank)
    from amd_kernels.attention.nvfp4_sparse_layer_gqa16_d128_p128_top16 import layer
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import decode_top16
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import nvfp4_index_pages, score_width
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4.host import IndexSplit4RankOwned

    ql = args.ql
    reqs = [int(n) for n in args.reqs.split(",")]
    if args.lens_json:
        table = json.load(open(args.lens_json))
        lens_by_n = {n: table[str(n)] for n in reqs}
    else:
        lens_by_n = {n: synthetic_lens(n) for n in reqs}
        if rank == 0:
            print(f"# synthetic lengths (no --lens-json): log-uniform {SYNTHETIC_RANGE[0]}-{SYNTHETIC_RANGE[1]} "
                  "quantiles", flush=True)
    max_reqs = max(64, max(reqs))
    t0 = time.time()
    index = decode_top16.IndexDecodeTop16(dev, query_len=ql, max_reqs=max_reqs, max_seq_len=MAX_MODEL_LEN, rank=rank,
                                          group=group)
    assert index.split is not None, index.split_off
    # a second four-GPU object: the device barrier between timed graphs
    bar_cp = IndexSplit4RankOwned(rank, group, dev, ql * 4, 64, ql)
    biggest = max(sum(cdiv(s, BLK) for s in v) for v in lens_by_n.values())
    slots = max(cdiv(MIN_POOL_BYTES, PAGE_BYTES), 3 * biggest + 8)
    icache = torch.empty(slots, BLK, 72, dtype=torch.uint8, device=dev)
    g = torch.Generator(device="cuda")
    for c0 in range(0, slots, 512):  # the same pages on every rank (the replicated cache)
        n = min(512, slots - c0)
        g.manual_seed(1_000_003 + c0 // 512)
        icache[c0 : c0 + n] = nvfp4_index_pages((torch.randn(n * BLK, D, device=dev, generator=g) * 2).bfloat16())
    g.manual_seed(17)
    kv = torch.randint(0, 256, (slots, 2, BLK, 72), dtype=torch.uint8, device=dev, generator=g) & 0x77
    torch.cuda.synchronize()
    if rank == 0:
        print(f"# ql {ql}; pool {slots} slots ({slots * (PAGE_BYTES + KV_PAGE_BYTES) / 2**30:.2f} GB per rank); "
              f"setup + warm-up {time.time() - t0:.1f} s", flush=True)
    rng_b = np.random.default_rng(7)
    b_iq = torch.from_numpy(rng_b.standard_normal((ql, D), dtype=np.float32)).cuda().to(torch.float8_e4m3fn)
    b_tab = torch.arange(4, dtype=torch.int32, device=dev).view(1, 4)
    b_seq = torch.tensor([4 * BLK], dtype=torch.int32, device=dev)

    def barrier():
        dist.barrier()
        bar_cp.score_push(b_iq, icache, b_tab, b_seq, INIT, LOCAL, 64, sharded=False)

    one = torch.ones((), dtype=torch.float32, device=dev)
    sm = D**-0.5

    def call(inp, o, side):
        saved = dict(decode_top16.SPLIT_MIN_REQS)
        if side == "split4":
            decode_top16.SPLIT_MIN_REQS[ql] = 1
        try:
            r = layer.index_sparse_decode(
                index, inp["iq"], icache, inp["table"], inp["seq"], INIT, LOCAL, inp["q"], kv, o["ids"], inp["table"],
                inp["seq"], sm, o["out"], k_scale=one, v_scale=one, decode_query_len=ql, max_seq_len=MAX_MODEL_LEN,
                quant_out=o["quant"], split=side != "one_gpu",
            )
        finally:
            decode_top16.SPLIT_MIN_REQS.clear()
            decode_top16.SPLIT_MIN_REQS.update(saved)
        want = "split4" if side == "split4" else ("fused", "chain")
        assert r in want, (side, r)

    sides = ["one_gpu", "split4"]
    for n in reqs:
        t0 = time.time()
        lens = lens_by_n[n]
        rows = n * ql
        span = sum(cdiv(s, BLK) for s in lens)
        regions = max(1, slots // max(span, 1))
        assert regions >= 3, ("pool too small", slots, span)
        per_call = span * PAGE_BYTES
        calls = max(100, 1 + cdiv(REUSE_BYTES, per_call))
        if calls >= regions:
            calls = cdiv(calls, regions) * regions
        inputs = []
        for i in range(calls):
            rng = np.random.default_rng(1000 * n + i)
            q4 = rng.standard_normal((rows, WORLD, D), dtype=np.float32) * 0.5
            iq = torch.from_numpy(np.ascontiguousarray(q4[:, rank : rank + 1])).cuda().to(torch.float8_e4m3fn)
            q = (torch.from_numpy(rng.standard_normal((rows, H, D), dtype=np.float32)).cuda()).bfloat16()
            inputs.append(dict(iq=iq, q=q, table=make_tables(lens, i % regions, span, slots, rng, MAX_MODEL_LEN // BLK),
                               seq=torch.tensor(lens, dtype=torch.int32, device=dev)))
        outs = {}
        for s in sides:
            outs[s] = dict(ids=torch.empty(1, rows, TOPK, dtype=torch.int32, device=dev),
                           out=torch.empty(rows, H, D, dtype=torch.bfloat16, device=dev),
                           quant=layer.quant_out_buffers(rows, OPROJ_N, OPROJ_K, dev))
        runs = {}
        for s in sides:
            for inp in inputs[:2]:
                call(inp, outs[s], s)
            torch.cuda.synchronize()
            barrier()
            graphs = []
            chunks = max(1, calls // 100)
            for k in range(chunks):
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    for inp in inputs[k * calls // chunks : (k + 1) * calls // chunks]:
                        call(inp, outs[s], s)
                graphs.append(gr)
            torch.cuda.synchronize()
            runs[s] = graphs
        check = {}
        for i in (0, calls - 1):
            got = []
            for s in sides:
                o = outs[s]
                o["ids"].fill_(-7)
                o["out"].fill_(7.0)
                if o["quant"] is not None:  # the GEMM tile's padding rows stay as filled
                    o["quant"][0].fill_(0)
                    o["quant"][1].fill_(0)
                barrier()
                call(inputs[i], o, s)
                torch.cuda.synchronize()
                got.append((o["ids"].clone(), o["out"].clone(), None if o["quant"] is None else o["quant"][0].clone()))
            (ia, oa, qa), (ib, ob, qb) = got
            bad = (ia != ib).any(-1).sum().item()
            bad_out = (oa.view(torch.int16) != ob.view(torch.int16)).flatten(1).any(-1).sum().item()
            if qa is not None:
                bad_out += int(not torch.equal(qa, qb))
            t = torch.tensor([bad, bad_out], dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            check[i] = [int(x) for x in t.tolist()]
        for s in sides:
            barrier()
            for _ in range(2):
                for gr in runs[s]:
                    gr.replay()
                torch.cuda.synchronize()
        samples = {s: [] for s in sides}
        for k in range(args.rounds):
            order = sides if k % 2 == 0 else sides[::-1]
            for s in order:
                el = 0.0
                for _ in range(args.replays):
                    for gr in runs[s]:
                        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                        barrier()
                        st.record()
                        gr.replay()
                        en.record()
                        torch.cuda.synchronize()
                        el += st.elapsed_time(en)
                us = el * 1e3 / (args.replays * calls)
                t = torch.tensor([us], dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                samples[s].append(t.item())
        if rank == 0:
            med = {s: statistics.median(v) for s, v in samples.items()}
            rec = dict(ql=ql, reqs=n, rows=rows, lens_json=os.path.basename(args.lens_json or "synthetic"),
                       mean_ctx=int(np.mean(lens)),
                       calls=calls, samples=samples, check=check, secs=round(time.time() - t0, 1))
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            rng_s = " ".join(f"{s}={med[s]:.2f} [{min(v):.2f}-{max(v):.2f}]" for s, v in samples.items())
            ratio = med["split4"] / med["one_gpu"]
            print(f"ql={ql} N={n:3d} ctx~{int(np.mean(lens)):7d} {rng_s} split4/one={ratio:.3f} diff(ids,out)={check}",
                  flush=True)
        del inputs, runs
        torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


def worker(rank, port, args, errfile):
    try:
        with torch.inference_mode():
            body(rank, port, args)
    except BaseException:
        with open(f"{errfile}.{rank}", "w") as f:
            f.write(traceback.format_exc())
        raise


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ql", type=int, default=4, help="decode rows per request (1 + the draft length)")
    ap.add_argument("--lens-json", default=None, help='JSON {"N": [context per request]}; default: synthetic_lens')
    ap.add_argument("--reqs", required=True, help="request counts N to time")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--out", default=None, help="append JSON lines here")
    args = ap.parse_args()
    assert torch.cuda.device_count() >= WORLD, "needs four GPUs"
    import torch.multiprocessing as mp

    errfile = f"/tmp/bench_index_decode_ranks4_err_{os.getpid()}"
    try:
        mp.spawn(worker, args=(free_port(), args, errfile), nprocs=WORLD, join=True)
    except Exception:
        for r in range(WORLD):
            if os.path.exists(f"{errfile}.{r}"):
                print(open(f"{errfile}.{r}").read(), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
