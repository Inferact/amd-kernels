# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``sparse_mla_decode`` against AITER's Triton ``sparse_mla_fwd`` (the gfx950 fp8 kernel with
the inverse-RoPE MXFP8 epilogue), K / V cold.

One HIP graph holds ``--copies`` (100) launches, each with its own queries and key lists; every list names rows of
pools of 1 GiB (the 256 MB MALL cannot hold a launch's rows twice). A step has B requests of 6 verify rows (the last
request of a row count that is no multiple of 6 takes the rest): the window is the 128 slots before the row's
position, the top-512 list shares 80% of its rows with the request's other rows (the measured overlap of the
indexer's selection) and the rest is fresh. Per row count the two implementations are timed A B B A (baseline,
this kernel, this kernel, baseline) in two passes; every graph is replayed 10 times per round, the time of a pass is
the minimum over 15 rounds, in us per launch. The first and the last launch are checked against the baseline's
dequantized output and the first against ``reference.py`` (16 rows) before timing.

    python3 benchmarks/attention/sparse_mla_decode_fp8kv_h64_d512_rot64/benchmark_decode.py [--rows 16 192 768]
"""

import argparse
import functools

import torch
from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd
from amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64 import reference as R
from amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64 import sparse_mla_decode

H, D = 64, 512
POOL_ROWS = 1 << 21  # 1 GiB of 512-byte rows
SWA_BLOCK, COMP_BLOCK = 32, 64
WINDOW, TOPK, SHARE = 128, 512, 0.8
SOFTMAX_SCALE = D**-0.5


def e4m3(shape, std, gen):
    out = torch.empty(shape, dtype=torch.uint8, device="cuda")
    flat = out.view(-1)
    for i in range(0, flat.numel(), 1 << 26):
        n = min(1 << 26, flat.numel() - i)
        flat[i : i + n] = (torch.randn(n, generator=gen, device="cuda") * std).to(torch.float8_e4m3fn).view(torch.uint8)
    return out


def make_launch(rows, gen, swa_rows, comp_rows, ctx=(8192, 131072)):
    """The arguments of one launch: ``rows`` query rows in requests of 6."""
    lens = [6] * (rows // 6) + ([rows % 6] if rows % 6 else [])
    B = len(lens)
    cpu = torch.Generator().manual_seed(int(torch.randint(0, 1 << 30, (1,), generator=gen, device="cuda")))
    ctxs = torch.randint(ctx[0], ctx[1], (B,), generator=cpu)
    req = torch.repeat_interleave(torch.arange(B), torch.tensor(lens))
    j = torch.cat([torch.arange(n) for n in lens])
    pos = (ctxs[req] + j).cuda()
    # window: slots of the 128 positions up to the row's, in a per-request region of the window pool
    region = torch.randint(0, swa_rows // 256, (B,), generator=cpu)[req].cuda() * 256
    k = torch.arange(WINDOW, device="cuda")
    p_k = pos[:, None] - (WINDOW - 1) + k[None, :]
    keep = p_k >= 0
    main = (region[:, None] + p_k % 256)[keep]
    main_n = keep.sum(1)
    # top-k: the request's base set, 20% of every row replaced
    ncomp = (ctxs // 2).clamp_min(TOPK)
    base = torch.stack([torch.randint(0, int(n), (TOPK,), generator=cpu) for n in ncomp]).cuda()[req.cuda()]
    fresh = (torch.rand((rows, TOPK), generator=gen, device="cuda") * (pos[:, None] // 2)).long()
    sel = torch.where(torch.rand((rows, TOPK), generator=gen, device="cuda") < SHARE, base, fresh)
    cbase = torch.randint(0, comp_rows, (B,), generator=cpu)[req].cuda()
    extra = ((cbase[:, None] + sel) % comp_rows).flatten()
    ip = lambda n: torch.cat([torch.zeros(1, dtype=torch.int64, device="cuda"), n.cumsum(0)]).int()
    ang = torch.rand((int(pos.max()) + 8, 32), generator=gen, device="cuda") * 6.2832
    return dict(
        q=e4m3((rows, H, D), 1.0, gen), main_indptr=ip(main_n), main_indices=main.int(),
        extra_indptr=(torch.arange(rows + 1, device="cuda") * TOPK).int(), extra_indices=extra.int(),
        positions=pos.int(), cos_sin=torch.cat([ang.cos(), ang.sin()], dim=1).float(),
    )  # fmt: skip


def run_ours(L, pools, out, p_terms=2):
    swa, comp, sink, qs, ks = pools
    sparse_mla_decode(L["q"], swa, L["main_indptr"], L["main_indices"], comp, L["extra_indptr"], L["extra_indices"],
                      sink, SOFTMAX_SCALE, qs, ks, L["positions"], L["cos_sin"], out[0], out[1], p_terms=p_terms)  # fmt: skip


def run_base(L, pools, out):
    swa, comp, sink, qs, ks = pools
    f8 = torch.float8_e4m3fn
    sparse_mla_fwd(
        L["q"].view(f8), swa.view(f8), L["main_indptr"], L["main_indices"], SOFTMAX_SCALE, kv_scale=ks, q_scale=qs,
        kv_lora_rank=D, qk_rope_head_dim=0, dot_precision="fp8", attn_sink=sink, extra_kv=comp.view(f8),
        extra_indptr=L["extra_indptr"], extra_indices=L["extra_indices"], has_invalid=False,
        inv_rope_positions=L["positions"], inv_rope_cos_sin_cache=L["cos_sin"],
        out_mxfp8=(out[0].view(f8), out[1]),
    )  # fmt: skip


def graph_of(fn, launches, pools, out):
    for L in launches[:2]:
        fn(L, pools, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for L in launches:
            fn(L, pools, out)
    torch.cuda.synchronize()
    return graph


def replay_us(graph, copies, replays=10, rounds=15):
    for _ in range(3):
        graph.replay()
    best = float("inf")
    for _ in range(rounds):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / (replays * copies))
    return best


def check(rows, launches, pools, out_a, out_b, run_ours):
    swa, comp, sink, qs, ks = pools
    for i in (0, len(launches) - 1):
        run_ours(launches[i], pools, out_a)
        run_base(launches[i], pools, out_b)
        torch.cuda.synchronize()
        a, b = R.dequant(*out_a).double(), R.dequant(*out_b).double()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1).min().item()
        assert cos >= 0.998, ("vs the baseline", rows, i, cos)
    if rows <= 32:
        L = launches[0]
        run_ours(L, pools, out_a)
        att = R.attention_reference(L["q"], 1.0, swa, L["main_indptr"], L["main_indices"], comp, L["extra_indptr"],
                                    L["extra_indices"], 1.0, sink, SOFTMAX_SCALE)  # fmt: skip
        ref = R.inverse_rope(att, L["positions"], L["cos_sin"]).flatten(1).double()
        cos = torch.nn.functional.cosine_similarity(R.dequant(*out_a).double(), ref, dim=1).min().item()
        assert cos >= 0.9990, ("vs the reference", rows, cos)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="*", type=int, default=[16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768])
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument("--p-terms", type=int, default=2, help="1: one e4m3 P (the baseline's numerics), 2: hi + residual")
    args = parser.parse_args()
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("requires gfx950")
    gen = torch.Generator(device="cuda").manual_seed(0)
    swa = e4m3((POOL_ROWS // SWA_BLOCK, SWA_BLOCK, D), 1.0, gen)
    comp = e4m3((POOL_ROWS // COMP_BLOCK, COMP_BLOCK, D), 1.0, gen)
    sink = torch.randn(H, generator=gen, device="cuda")
    one = torch.ones(1, device="cuda")
    pools = (swa, comp, sink, one, one)
    with torch.inference_mode():
        for rows in args.rows:
            launches = [make_launch(rows, gen, POOL_ROWS, POOL_ROWS) for _ in range(args.copies)]
            out_a = (torch.empty(rows, H * D, dtype=torch.uint8, device="cuda"), torch.empty(rows, H * D // 32, dtype=torch.uint8, device="cuda"))
            out_b = (torch.empty_like(out_a[0]), torch.empty_like(out_a[1]))
            ours = functools.partial(run_ours, p_terms=args.p_terms)
            check(rows, launches, pools, out_a, out_b, ours)
            graphs = {"baseline": graph_of(run_base, launches, pools, out_b), "ours": graph_of(ours, launches, pools, out_a)}
            times = {"baseline": [], "ours": []}
            for name in ("baseline", "ours", "ours", "baseline", "baseline", "ours", "ours", "baseline"):
                times[name].append(replay_us(graphs[name], args.copies))
            b, o = times["baseline"], times["ours"]
            print(
                f"rows={rows:4d}: baseline {sum(b) / len(b):7.1f} us ({min(b):.1f}-{max(b):.1f}), ours "
                f"{sum(o) / len(o):7.1f} us ({min(o):.1f}-{max(o):.1f}), x{sum(b) / sum(o) * len(o) / len(b):.2f}"
            )
            del launches, graphs


if __name__ == "__main__":
    main()
