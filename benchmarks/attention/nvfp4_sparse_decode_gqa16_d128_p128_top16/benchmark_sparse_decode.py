# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of the NVFP4 sparse decode on gfx950, K / V cold.

Per case one HIP graph holds ``--calls`` (100) serving launches (automatic schedule,
the device's XCD count), each with its own queries, block table and output; the
tables point at the call's own copy of the KV pages, so no call finds its pages in L2
or the 256 MB MALL. Reported: the median over 5 rounds of 10 replays of the time per
call, the host's key (splits, XCD grid, tail grid) and the K / V bytes per call (every
distinct (request, KV head, block) the rows select, 18 KB each: a request's rows share
their blocks in L2). The first call is checked against ``reference.py`` before timing.

Shapes: the served configurations of the tests' ``plan.py``; contexts 60K-100K tokens; the
top-k with the measured serving overlap: every 4 consecutive rows of a request and KV
head select 25 distinct blocks besides their own, and successive 4-row groups share two
thirds of the older ones.

    python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_sparse_decode.py \
        [--configs kv1_f16_ql8] [--requests 1 8]
"""

import argparse
import statistics

import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import nvfp4_sparse_decode
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
    BLOCK,
    GROUP,
    HEAD_DIM,
    QUANT_REL,
    REL_L2,
    TOPK,
    aiter_scale_offsets,
    mxfp8_dequantize,
    quantize_cache,
    rel_l2,
    sparse_decode_reference,
    tile_code_offsets,
)

from tests.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.plan import SERVED, plan

REQUESTS = {
    "kv1_bf16_ql1": [1, 16, 64, 128, 256],
    "kv1_bf16_ql4": [1, 8, 16, 32, 64, 128],
    "kv1_bf16_ql8": [1, 2, 4, 16, 32, 64, 128, 256],
    "kv1_f16_ql1": [1, 16, 64, 128, 256],
    "kv1_f16_ql4": [1, 8, 16, 32, 64, 128],
    "kv1_f16_ql8": [1, 2, 4, 16, 32, 64, 128, 256],
}
UNION_BLOCKS = 25  # measured: distinct blocks per 4 query rows
PAGE_HEAD_BYTES = 2 * 9216  # one KV head's K and V of one page
SM_SCALE = HEAD_DIM**-0.5


def _topk(reqs, ql, hkv, seq_lens, gen):
    """[hkv, reqs * ql, TOPK] logical blocks with the measured overlap."""
    topk = torch.full((hkv, reqs * ql, TOPK), -1, dtype=torch.int32)
    older_n = UNION_BLOCKS - 1
    shared_n = older_n * 2 // 3
    for r in range(reqs):
        seq = int(seq_lens[r])
        first_self = (seq - ql) // BLOCK
        for h in range(hkv):
            cand = torch.randperm(first_self, generator=gen)
            shared, fresh = cand[:shared_n], cand[shared_n:].tolist()
            for g in range((ql + 3) // 4):
                pool = torch.cat(
                    [shared, torch.tensor([fresh.pop() for _ in range(older_n - shared_n)])]
                )
                toks = range(g * 4, min(ql, g * 4 + 4))
                chunks = torch.randperm(older_n, generator=gen).chunk(len(toks))
                for t, must in zip(toks, chunks):
                    must = must[: TOPK - 1]
                    rest = torch.tensor([i for i in range(older_n) if i not in set(must.tolist())])
                    extra = rest[torch.randperm(rest.numel(), generator=gen)]
                    sel = pool[torch.cat([must, extra[: TOPK - 1 - must.numel()]])]
                    row = r * ql + t
                    topk[h, row, 0] = (seq - ql + t) // BLOCK
                    topk[h, row, 1:] = sel
    return topk


def make_calls(config, reqs, calls, seed):
    hkv, compute, ql, _ = SERVED[config]
    gen = torch.Generator().manual_seed(seed)
    seq_lens = torch.randint(60000, 100001, (reqs,), generator=gen)
    topk = _topk(reqs, ql, hkv, seq_lens, gen)
    used = [
        sorted(set(topk[:, r * ql : (r + 1) * ql].flatten().tolist())) for r in range(reqs)
    ]
    pool = 1 + sum(len(u) for u in used)
    perm = torch.randperm(pool - 1, generator=gen) + 1
    table = torch.zeros(reqs, int(seq_lens.max()) // BLOCK + 1, dtype=torch.int32)
    i = 0
    for r, blocks in enumerate(used):
        for b in blocks:
            table[r, b] = perm[i]
            i += 1
    chunks = []
    for start in range(0, pool, 256):
        shape = (min(256, pool - start), BLOCK, hkv, HEAD_DIM)
        k = torch.randn(shape, generator=gen).bfloat16().cuda()
        v = torch.randn(shape, generator=gen).bfloat16().cuda()
        chunks.append(quantize_cache(k, v, 1.0, 1.0))
    base = torch.cat(chunks)
    cache = base.repeat(calls, 1, 1, 1)
    distinct = sum(
        len({b for b in topk[h, r * ql : (r + 1) * ql].flatten().tolist() if b >= 0})
        for r in range(reqs)
        for h in range(hkv)
    )
    rows = reqs * ql
    width = hkv * GROUP * HEAD_DIM
    shared = {
        "kv_cache": cache,
        "topk_idx": topk.cuda(),
        "seq_lens": seq_lens.to(torch.int32).cuda(),
        "k_scale": torch.ones((), dtype=torch.float32, device="cuda"),
        "v_scale": torch.ones((), dtype=torch.float32, device="cuda"),
        "compute": compute,
        "ql": ql,
        "rows": rows,
        "width": width,
        "kv_bytes": distinct * PAGE_HEAD_BYTES,
        "pool": pool,  # pages per copy; call c reads copy c
    }
    per_call = []
    for c in range(calls):
        q = torch.randn(rows, hkv * GROUP, HEAD_DIM, generator=gen).bfloat16().cuda()
        u8 = dict(dtype=torch.uint8, device="cuda")
        per_call.append(
            {
                "q": q,
                "block_table": (table + c * pool).cuda(),
                "out": torch.empty_like(q),
                "codes": torch.empty(-(-rows // 16) * 16, width, **u8),
                "scales": torch.empty(-(-rows // 32) * 32, width // 32, **u8),
            }
        )
    return shared, per_call


def launch(shared, call, quant, splits=None):
    nvfp4_sparse_decode(
        call["q"],
        shared["kv_cache"],
        shared["topk_idx"],
        call["block_table"],
        shared["seq_lens"],
        SM_SCALE,
        call["out"],
        k_scale=shared["k_scale"],
        v_scale=shared["v_scale"],
        decode_query_len=shared["ql"],
        quant_out=(call["codes"], call["scales"], 16) if quant else None,
        compute_type=shared["compute"],
        splits=splits,
    )


def check(shared, call, quant):
    ref = sparse_decode_reference(
        call["q"],
        shared["kv_cache"][: shared["pool"]],  # call 0's copy
        shared["topk_idx"],
        call["block_table"],
        shared["seq_lens"],
        SM_SCALE,
        1.0,
        1.0,
        shared["ql"],
        round_kv_to_e4m3=shared["compute"] == torch.bfloat16,
    )
    if not quant:
        rel = rel_l2(call["out"], ref)
        assert rel < REL_L2, rel
        return
    rows, width = shared["rows"], shared["width"]
    codes = call["codes"].view(-1)[tile_code_offsets(rows, width, ref.device)]
    scales = call["scales"].view(-1)[aiter_scale_offsets(rows, width, ref.device)]
    rel = rel_l2(mxfp8_dequantize(codes, scales).view_as(ref), ref)
    assert rel < QUANT_REL, rel


def timeit_graph(shared, per_call, quant, splits=None, replays=10, rounds=5, warmup=5):
    """Median over rounds of the mean time per call (us) of one graph of every call."""
    for call in per_call:
        launch(shared, call, quant, splits)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in per_call:
            launch(shared, call, quant, splits)
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for start, end in events:
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(
        s.elapsed_time(e) * 1e3 / (replays * len(per_call)) for s, e in events
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=list(SERVED))
    parser.add_argument("--requests", type=int, nargs="*")
    parser.add_argument("--outputs", nargs="*", default=["mxfp8", "bf16"])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument(
        "--splits", type=int, nargs="*", help="time these explicit split counts too "
        "(the padded grid; the automatic schedule is the first row of each case)"
    )
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    if "gfx950" not in props.gcnArchName:
        raise RuntimeError("requires gfx950")
    print(f"{props.name} ({props.gcnArchName}), {props.multi_processor_count} CUs")
    print("| config | requests | rows | output | splits / XCD grid / tail | us | K/V MB | GB/s |")
    print("|---|---:|---:|---|---|---:|---:|---:|")
    with torch.inference_mode():
        for config in args.configs:
            hkv, compute, ql, _ = SERVED[config]
            for reqs in args.requests or REQUESTS[config]:
                shared, per_call = make_calls(config, reqs, args.calls, seed=reqs)
                for output in args.outputs:
                    quant = output == "mxfp8"
                    launch(shared, per_call[0], quant)
                    torch.cuda.synchronize()
                    check(shared, per_call[0], quant)
                    us = timeit_graph(shared, per_call, quant)
                    splits, _, xcds, _, _, tail = plan(reqs, ql, hkv, compute, quant)
                    mb = shared["kv_bytes"] / 1e6
                    rows = [(f"{splits} / {xcds or '-'} / {'yes' if tail else 'no'}", us)]
                    for forced in args.splits or []:
                        launch(shared, per_call[0], quant, forced)
                        torch.cuda.synchronize()
                        check(shared, per_call[0], quant)
                        t = timeit_graph(shared, per_call, quant, forced)
                        rows.append((f"{forced} (forced)", t))
                    for key, t in rows:
                        print(
                            f"| {config} | {reqs} | {reqs * ql} | {output} | {key} | {t:.1f} "
                            f"| {mb:.1f} | {shared['kv_bytes'] / t / 1e3:.0f} |",
                            flush=True,
                        )
                del shared, per_call
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
