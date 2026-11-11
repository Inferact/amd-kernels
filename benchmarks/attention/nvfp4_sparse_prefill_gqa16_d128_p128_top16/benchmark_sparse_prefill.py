# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of the NVFP4 sparse prefill kernels on gfx950, inputs HBM-cold.

Timing (``timeit_graph``): one HIP graph holds 100 calls, each with its own input -- a
request of T new tokens after a 16384-token prefix, with its own Q, its own indexer-like top-k picks (``tests.cases.shared_picks``),
its own NVFP4 pages and its own outputs (fewer inputs where 100 exceed 64 GB) -- so the
operands rotate through far more than the 256 MB MALL and every call reads its KV
pages and Q from HBM. The graph's time per
call is the median of 5 rounds of ``replays`` replays. The variants (bf16, fp16; fp8
with ``--fp8``) are timed on the same GPU, alternated, for two rounds; the first and
last input are checked against ``reference.py`` before timing.

Output modes: ``plain`` (bf16 ``out``), ``quant`` (the o_proj's MXFP8 codes + AITER
scales, the served prefill-only step) and ``pack`` (the sequence-parallel all-to-all
rows, TP4 only). ``--fp8`` adds fp8 with a prepacked Q (``query.pack_query`` outside the
timed kernel; plain only); ``--fp8-bf16q`` adds ``fp8q``: fp8 with the bf16 Q a caller
passes (quantized inside the kernel) in every mode; ``--fp16-lazy`` adds ``fp16lazy``: fp16 with the
opt-in lazy rescale (``lazy_rescale=True``).
"""

import argparse
import statistics

import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import nvfp4_sparse_prefill, reference
from tests.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.cases import BLOCK, GROUP, HEAD_DIM, SM_SCALE, TP4_HKV, shared_picks

PREFIX = 16384
N_INPUTS = 100
# Q + outputs of all inputs of a case; the largest steps get fewer than 100 inputs
# (each of their calls already moves more than 256 MB)
BUDGET_BYTES = 64 * 1024**3
# (KV heads, prefill tokens per step): TP4 up to 65536 tokens
CASES = [
    (TP4_HKV, 1024),
    (TP4_HKV, 4096),
    (TP4_HKV, 16384),
    (TP4_HKV, 65536),
]


def timeit_graph(calls, replays=10, rounds=5, warmup=3):
    """Median over rounds of the mean time per call (ms) of one graph of ``calls``."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
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
    return statistics.median(s.elapsed_time(e) / (replays * len(calls)) for s, e in events)


def make_inputs(hkv, t, n_inputs, seed=0):
    """``n_inputs`` requests of ``t`` tokens after ``PREFIX``, each on its own pages."""
    ctx = PREFIX + t
    nb = (ctx + BLOCK - 1) // BLOCK
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    chunks = []
    for i in range(n_inputs):  # each input's pages, NVFP4-encoded as the insert does
        kv = []
        for _ in range(2):
            x = torch.randn(nb, hkv, BLOCK, HEAD_DIM, generator=gen, device="cuda") * 2
            kv += list(reference.nvfp4_quantize(x.bfloat16().float()))
        chunks.append(reference.pack_kv_pages(*kv))
    cache = torch.cat(chunks)
    del chunks
    pos = torch.arange(PREFIX, ctx, dtype=torch.int32)
    req = torch.zeros(t, dtype=torch.int32)
    inputs = []
    for i in range(n_inputs):
        table = (torch.randperm(nb, generator=cpu).int() + i * nb)[None].cuda()
        topk = shared_picks(pos, req, hkv, cpu).contiguous().cuda()
        q = (torch.randn(t, hkv * GROUP, HEAD_DIM, generator=gen, device="cuda") * 2).bfloat16()
        inputs.append(dict(q=q, topk=topk, table=table))
    return cache, pos.cuda(), req.cuda(), inputs


def n_inputs(hkv, t, n_variants, limit):
    q_bytes = t * hkv * GROUP * HEAD_DIM * 2
    return max(2, min(limit, BUDGET_BYTES // (q_bytes * (1 + n_variants))))


def outputs(hkv, t, mode, shared_out):
    """Per-call outputs; ``out`` is only written in the plain mode."""
    h = hkv * GROUP
    if mode != "plain":
        out = shared_out
    else:
        out = torch.empty(t, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    if mode == "quant":
        padded = (t + 31) // 32 * 32
        codes = torch.empty(padded, h * HEAD_DIM, dtype=torch.uint8, device="cuda")
        scales = torch.empty(padded, h * 4, dtype=torch.uint8, device="cuda")
        return out, {"quant_out": (codes, scales)}
    if mode == "pack":
        return out, {"pack_out": torch.empty(t, h * 132, dtype=torch.uint8, device="cuda")}
    return out, {}


def check(cache, pos, req, inp, hkv, variant, out):
    t = inp["q"].shape[0]
    rows = sorted({0, 1, 2, 3, t // 2, t - 4, t - 3, t - 2, t - 1})
    ref = reference.sparse_prefill_reference(
        inp["q"], cache, inp["topk"], inp["table"], req, pos, hkv, SM_SCALE, rows=rows,
        fp8=variant == "fp8",
    )
    err = reference.rel_l2(out[rows], ref)
    assert err < reference.REL_L2_TOL[variant], (variant, hkv, t, err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", action="store_true", help="also time the fp8 kernel (plain, prepacked Q)")
    ap.add_argument("--fp8-bf16q", action="store_true", help="also time fp8 with a bf16 Q in every mode (fp8q)")
    ap.add_argument("--fp16-lazy", action="store_true",
                    help="also time fp16 with the opt-in lazy rescale (fp16lazy, another numeric rule)")
    ap.add_argument("--inputs", type=int, default=N_INPUTS)
    ap.add_argument("--rounds", type=int, default=2, help="alternated A/B rounds")
    ap.add_argument("--only", default="", help="hkv:T,... subset of the cases")
    args = ap.parse_args()
    cases = CASES
    if args.only:
        cases = [tuple(int(x) for x in c.split(":")) for c in args.only.split(",")]
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} ({props.gcnArchName}), {props.multi_processor_count} CUs")
    extra = (("fp8",) if args.fp8 else ()) + (("fp8q",) if args.fp8_bf16q else ()) + (
        ("fp16lazy",) if args.fp16_lazy else ())
    print("hkv | q heads | tokens | inputs | mode | " + " | ".join(
        f"{v} us" for v in ("bf16", "fp16") + extra
    ) + " | fp16 / bf16")
    for hkv, t in cases:
        n = n_inputs(hkv, t, 2 + len(extra), args.inputs)
        cache, pos, req, inputs = make_inputs(hkv, t, n)
        shared_out = torch.empty_like(inputs[0]["q"])
        modes = ["plain", "quant"] + (["pack"] if hkv == TP4_HKV else [])
        for mode in modes:
            variants = ["bf16", "fp16"] + (["fp8"] if args.fp8 and mode == "plain" else []) + (
                ["fp8q"] if args.fp8_bf16q else []) + (["fp16lazy"] if args.fp16_lazy else [])
            calls = {v: [] for v in variants}
            for v in variants:
                for i, inp in enumerate(inputs):
                    out, kw = outputs(hkv, t, mode, shared_out)
                    cd = {"fp8q": "fp8", "fp16lazy": "fp16"}.get(v, v)
                    lazy = dict(lazy_rescale=True) if v == "fp16lazy" else {}
                    if mode == "plain" and i in (0, len(inputs) - 1):
                        nvfp4_sparse_prefill(
                            inp["q"], cache, inp["topk"], inp["table"], req, pos, hkv,
                            SM_SCALE, out, compute_dtype=cd, **lazy,
                        )
                        check(cache, pos, req, inp, hkv, cd, out)
                    q = inp["q"]
                    if v == "fp8":  # input preparation outside the timed kernel
                        from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.query import (
                            pack_query,
                        )

                        q = pack_query(q)
                    calls[v].append(
                        lambda q=q, inp=inp, out=out, kw=kw, cd=cd, lazy=lazy: nvfp4_sparse_prefill(
                            q, cache, inp["topk"], inp["table"], req, pos, hkv, SM_SCALE,
                            out, compute_dtype=cd, **kw, **lazy,
                        )
                    )
            times = {v: [] for v in variants}
            for r in range(args.rounds):
                for v in (variants if r % 2 == 0 else variants[::-1]):
                    times[v].append(timeit_graph(calls[v]) * 1e3)
            med = {v: statistics.median(times[v]) for v in variants}
            spread = " ".join(f"{v}={'/'.join(f'{x:.1f}' for x in times[v])}" for v in variants)
            cols = " | ".join(f"{med[v]:.1f}" for v in variants)
            print(
                f"{hkv} | {hkv * GROUP} | {t} | {n} | {mode} | {cols} | "
                f"{med['fp16'] / med['bf16']:.3f}   (rounds: {spread})",
                flush=True,
            )
            del calls
        del cache, inputs, shared_out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
