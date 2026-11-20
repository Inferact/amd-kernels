# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The whole TP2 DSpark draft forward per step on 2 ranks: the input fc (5 taps) + the embedding + 5 layers (qkv,
q/k norm + RoPE + fp8 insert, SWA decode over a full 4096 window, o_proj, MLP, the all-reduce + norm tails) + the
final norm, 7 rows per request, against the TP2 stock path and the roofline.

- ``block``: ``DenseDecoderRanks2`` (fc eager through ``ConcatLinearComm``; ``forward`` in one CUDA graph) around the
  draft attention on its codes: qkv ``mxfp8_gemm_prequant`` (4608 x 6144), ``qkv_norm_rope_insert`` (32 q / 2 KV
  heads), ``swa_decode_fp8kv_oproj_input`` (MXFP8 epilogue for the o_proj) and o_proj (6144 x 4096).
- ``stock``: what vLLM runs at TP2 without the block, in one CUDA graph: the replicated fc (``torch.cat`` +
  ``F.linear``), ``VocabParallelEmbedding`` (mask, gather, AITER custom all-reduce), vLLM's Triton Gemma norms,
  Triton ``tl.dot_scaled`` qkv / o_proj / gate_up (row-major weights, fused activation quant), the same draft
  attention with bf16 output, the uncompiled SwiGLU-OAI aten chain, the amd_kernels down GEMM (6144 x 6144 is in
  its table), AITER custom all-reduce after o_proj and down.

Weights 5 x ~172 MB MXFP8 per path (> the 256 MB MALL), KV pools of disjoint pages per layer (cold), every request
one random token + six mask ids, context 8K-64K. Alternated rounds after a barrier, max over ranks, median us per
step. Roofline per rank: per op max(HBM bytes / 6.5 TB/s, flops / peak (MXFP8 5 PF, bf16 2.5 PF)), plus the link
bytes of the fc exchange and the ten all-reduce tails (owner form) at 56 GB/s per direction, serial.

    python3 benchmarks/decoder/dense_mxfp8_h6144_i6144_ranks2/benchmark_draft_forward.py [--reqs 1 8 32 52] [--parts]
"""
import argparse
import json
import statistics

import torch
import torch.distributed as dist

H, V2, I, QL, LAYERS = 6144, 100032, 6144, 7, 5
NQ, NKV, D, PAGE, W = 32, 2, 128, 128, 4096
QKV_N, O_K = (NQ + 2 * NKV) * D, NQ * D
EPS, SWIGLU = 1e-6, (1.702, 1.0, 7.0)
MASK_ID = 200058
HBM, MXFP8_PF, BF16_PF, LINK = 6.5e12, 5e15, 2.5e15, 56e9


def roofline_us(reqs):
    """(total, of which link) us per rank of one draft step at ``reqs`` requests (module doc)."""
    m = QL * reqs
    w8 = 1 + 1 / 32  # MXFP8 codes + e8m0 scales
    fc = max((3072 * 5 * H * 2 + m * 5 * H * 2 + m * H * 2) / HBM, 2 * m * 3072 * 5 * H / BF16_PF)

    def gemm(n, k):
        return max((n * k * w8 + m * k + m * n * 2) / HBM, 2 * m * n * k / MXFP8_PF)

    kv = reqs * NKV * (W - 1 + QL) * 256
    attn = max((kv + m * QKV_N * 2 + m * O_K) / HBM, 4 * m * NQ * D * (W - 1 + QL) / BF16_PF)
    mlp = max((2 * I * H * w8 + H * I * w8 + m * 2 * I * 2 * 2 + m * I * 2) / HBM, 2 * m * 3 * I * H / MXFP8_PF)
    layer = gemm(QKV_N, H) + attn + gemm(H, O_K) + mlp
    # per direction of the one link: the fc's 3072-column shard; nine MXFP8 tails and one bf16 tail (owner form from
    # 24 rows: half the partial rows to their owner, the owner's codes + scales / bf16 rows back; else all rows)
    quant_tail = m * (12288 + 6336) // 2 if m >= 24 else m * 12288
    link = (m * 3072 * 2 + 9 * quant_tail + m * 12288) / LINK
    local = fc + LAYERS * layer + m * H * 2 * 3 / HBM
    return (local + link) * 1e6, link * 1e6


def _weights(n, k, g, row_major):
    from amd_kernels.gemm.mxfp8_common import gemm as G
    from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8
    codes, exp = quantize_mxfp8((torch.randn((n, k), device=g.device, generator=g) * 0.02).bfloat16())
    pre = G.preshuffle_mxfp8_weight(codes.clone(), exp.clone())
    return pre, ((codes, exp) if row_major else None)


def _kv(reqs, g, cpu):
    """A pool of disjoint pages per layer, the block tables, lengths and draft slots/positions (full windows)."""
    per_req = (W + QL + PAGE - 1) // PAGE + 1
    pools, bts = [], []
    ctx = torch.randint(8192, 65536, (reqs,), generator=cpu)
    seq = ctx + QL
    nblk = int((seq.max() + PAGE - 1) // PAGE)
    for _ in range(LAYERS):
        n_pages = reqs * per_req + 2
        pool = (torch.randn((n_pages, NKV, PAGE, 256), device=g.device, generator=g) * 1.2).to(
            torch.float8_e4m3fn).view(torch.uint8)
        order = torch.randperm(n_pages, generator=cpu).int()
        bt = torch.zeros((reqs, nblk), dtype=torch.int32)
        cur = 0
        for r, (c, s_r) in enumerate(zip(ctx.tolist(), seq.tolist())):
            nb = (s_r + PAGE - 1) // PAGE
            lo = max((c - W + 1) // PAGE, 0)
            bt[r, lo:nb] = order[cur:cur + nb - lo]
            cur += nb - lo
        pools.append(pool)
        bts.append(bt.to(g.device))
    pos = (seq[:, None] - QL + torch.arange(QL)[None]).flatten()
    slots = [(bt.cpu()[torch.arange(reqs).repeat_interleave(QL), pos // PAGE].long() * PAGE + pos % PAGE).to(g.device)
             for bt in bts]
    qsl = torch.arange(0, (reqs + 1) * QL, QL, dtype=torch.int32, device=g.device)
    return pools, bts, seq.int().to(g.device), qsl, pos.to(g.device), slots


def _timed(rank, world, arms, rounds, reps):
    """{name: median over rounds of the max over ranks, us per call} of ``arms`` {name: callable(reps)}."""
    ts = {k: [] for k in arms}
    for r in range(rounds):
        for k in (list(arms) if r % 2 == 0 else list(reversed(list(arms)))):
            dist.barrier()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            arms[k](reps)
            b.record()
            b.synchronize()
            ts[k].append(a.elapsed_time(b) * 1e3 / reps)
    every = [None] * world
    dist.all_gather_object(every, ts)
    return {k: statistics.median(max(e[k][j] for e in every) for j in range(rounds)) for k in ts}


def worker(rank, errors, reqs_list, rounds, check, parts):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import _mxfp8_dot_scaled_linear
    from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import gemma_fused_add_rmsnorm, gemma_rmsnorm

    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.qkv_norm_rope_insert import qkv_norm_rope_insert
    from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import swa_decode_fp8kv_oproj_input
    from amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2 import DenseDecoderRanks2, DenseLayer, mxfp8_swiglu_oai_mlp
    from amd_kernels.gemm.mxfp8_common import gemm as G
    from amd_kernels.gemm.mxfp8_common.reference import swiglu_bf16_reference

    torch.set_num_threads(8)
    dev = torch.device("cuda", rank)
    world = dist.get_world_size()
    g = torch.Generator(device=dev).manual_seed(5)  # replicated tensors
    gr = torch.Generator(device=dev).manual_seed(50 + rank)  # this rank's shards
    fc_w = (torch.randn(H, 5 * H, device=dev, generator=g) * 0.01).bfloat16()
    emb = (torch.randn(V2, H, device=dev, generator=gr) * 0.5).bfloat16()
    blk = DenseDecoderRanks2(rank, dist.group.WORLD, dev, fc_weight=fc_w, embedding=emb)
    ca = CustomAllreduce(group=dist.group.WORLD, device=dev, max_size=64 * 1024 * 1024)

    def vec(scale=0.1):
        return (torch.randn(H, device=dev, generator=g) * scale).bfloat16()

    qkv = [_weights(QKV_N, H, gr, True) for _ in range(LAYERS)]
    o = [_weights(H, O_K, gr, True) for _ in range(LAYERS)]
    gu = [_weights(2 * I, H, gr, True) for _ in range(LAYERS)]
    dn = [_weights(H, I, gr, False) for _ in range(LAYERS)]
    layers = [DenseLayer(vec(), (QKV_N, H), vec(), *gu[i][0], *dn[i][0]) for i in range(LAYERS)]
    final = vec()
    qn = [(torch.randn(D, device=dev, generator=g) * 0.1).bfloat16() for _ in range(LAYERS)]
    kn = [(torch.randn(D, device=dev, generator=g) * 0.1).bfloat16() for _ in range(LAYERS)]
    cos_sin = (torch.rand(65536 + 64, 64, device=dev, generator=g) * 2 - 1).bfloat16()
    one = torch.ones((), dtype=torch.float32, device=dev)
    sm = D ** -0.5
    v_start = rank * V2

    def stock_embed(ids):
        local = ids - v_start
        valid = (local >= 0) & (local < V2)
        x = torch.nn.functional.embedding(torch.where(valid, local, 0), emb)
        x.masked_fill_(~valid[:, None], 0)
        return ca.custom_all_reduce(x)

    for reqs in reqs_list:
        m = QL * reqs
        cpu = torch.Generator().manual_seed(1000 + reqs)
        pools, bts, seq, qsl, pos, slots = _kv(reqs, g, cpu)
        ids = torch.full((reqs, QL), MASK_ID, dtype=torch.int64)
        ids[:, 0] = torch.randint(0, 2 * V2, (reqs,), generator=cpu)
        ids = ids.flatten().to(dev)
        taps = [torch.randn(m, H, device=dev, generator=g).bfloat16() for _ in range(5)]
        fc_out = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
        qbuf = torch.empty(m, NQ * D, dtype=torch.bfloat16, device=dev)

        def attention(i, x_qkv, mxfp8):
            qkv_norm_rope_insert(x_qkv, pos, qn[i], kn[i], cos_sin, EPS, NQ, NKV, qbuf, pools[i], slots[i], one, one)
            out = swa_decode_fp8kv_oproj_input(qbuf, pools[i], bts[i], seq, qsl, QL, sm, W, False, one, one,
                                               oproj_n=H if mxfp8 else None)
            if not mxfp8:
                return out
            if isinstance(out, torch.Tensor):
                return G.mxfp8_linear(out, *o[i][0], torch.bfloat16)
            codes, scales, tile = out
            c8 = codes.view(torch.float8_e4m3fn)
            return G.mxfp8_gemm_prequant(c8 if tile else c8[:m], scales, *o[i][0], torch.bfloat16, m).reshape(m, H)

        def block_attn(i, q, s):
            x_qkv = G.mxfp8_gemm_prequant(q, s, *qkv[i][0], torch.bfloat16, m).reshape(m, QKV_N)
            return attention(i, x_qkv, True)

        def block_stack():
            return blk.forward(ids, layers, block_attn, final, EPS, SWIGLU)

        def stock_fc():
            return torch.nn.functional.linear(torch.cat(taps, 1), fc_w)

        def stock_stack():
            x = stock_embed(ids)
            residual, h = x, gemma_rmsnorm(x, layers[0].input_norm, EPS)
            for i in range(LAYERS):
                if i:
                    h, residual = gemma_fused_add_rmsnorm(h, residual, layers[i].input_norm, EPS)
                x_qkv = _mxfp8_dot_scaled_linear(h, *qkv[i][1])
                a = attention(i, x_qkv, False)
                h = ca.custom_all_reduce(_mxfp8_dot_scaled_linear(a, *o[i][1]))
                h, residual = gemma_fused_add_rmsnorm(h, residual, layers[i].post_norm, EPS)
                act = swiglu_bf16_reference(_mxfp8_dot_scaled_linear(h, *gu[i][1]), *SWIGLU, "bf16_chain")
                h = ca.custom_all_reduce(G.mxfp8_linear(act, *dn[i][0], torch.bfloat16))
            return gemma_fused_add_rmsnorm(h, residual, final, EPS)[0]

        got, want = block_stack(), stock_stack()
        blk.fc(taps, m, fc_out)
        ref_fc = stock_fc()
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(got.float().flatten(), want.float().flatten(), dim=0)
        cos_fc = torch.nn.functional.cosine_similarity(fc_out.float().flatten(), ref_fc.float().flatten(), dim=0)
        if check and (cos < 0.99 or cos_fc < 0.9999):
            errors.append(f"rank {rank} R {reqs}: block vs stock cosine {float(cos):.5f}, fc {float(cos_fc):.6f}")
        cases = [("block", block_stack, blk.capture), ("stock", stock_stack, ca.capture),
                 ("stock_fc", stock_fc, ca.capture)]
        if parts:  # one launch per layer's weights (cold), divided by the count below
            q0, s0, res = blk.embed_norm_quant(ids, layers[0].input_norm, EPS, (QKV_N, H))
            part = (torch.randn(m, H, device=dev, generator=gr) * 0.1).bfloat16()
            qg, sg, res = blk.ar_norm_quant(part, res, layers[0].post_norm, EPS, (2 * I, H))
            cases += [
                ("embed", lambda: blk.embed_norm_quant(ids, layers[0].input_norm, EPS, (QKV_N, H)), blk.capture),
                ("attn", lambda: [block_attn(i, q0, s0) for i in range(LAYERS)], blk.capture),
                ("tail_q", lambda: [blk.ar_norm_quant(part, res, layers[i].post_norm, EPS, (2 * I, H))
                                    for i in range(LAYERS)], blk.capture),
                ("mlp", lambda: [mxfp8_swiglu_oai_mlp(qg, sg, m, *gu[i][0], *dn[i][0], *SWIGLU) for i in range(LAYERS)],
                 blk.capture),
                ("tail_bf16", lambda: blk.ar_norm(part, res, final, EPS), blk.capture)]
        graphs = {}
        for name, fn, ctx in cases:
            gph = torch.cuda.CUDAGraph()
            with ctx(), torch.cuda.graph(gph):
                fn()
            graphs[name] = gph
        for gph in graphs.values():
            gph.replay()
        torch.cuda.synchronize()

        def replays(name):
            def run(n):
                for _ in range(n):
                    graphs[name].replay()
            return run

        def block_fc(n):
            for _ in range(n):
                blk.fc(taps, m, fc_out)

        arms = {name: replays(name) for name in graphs}
        arms["block_fc"] = block_fc
        t = _timed(rank, world, arms, rounds, 10)
        for name in ("attn", "tail_q", "mlp"):
            if name in t:
                t[name] /= LAYERS
        if rank == 0:
            roof, link = roofline_us(reqs)
            new, old = t["block"] + t["block_fc"], t["stock"] + t["stock_fc"]
            print(json.dumps(dict(reqs=reqs, rows=m, block_stack=round(t["block"], 1), block_fc=round(t["block_fc"], 1),
                                  block=round(new, 1), stock_stack=round(t["stock"], 1),
                                  stock_fc=round(t["stock_fc"], 1), stock=round(old, 1), speedup=round(old / new, 2),
                                  roofline=round(roof, 1), roofline_link=round(link, 1),
                                  frac=round(roof / new, 3), cos=round(float(cos), 5),
                                  **{k: round(t[k], 1) for k in ("embed", "attn", "tail_q", "mlp", "tail_bf16")
                                     if k in t})), flush=True)
        del graphs, pools
        torch.cuda.empty_cache()
    blk.close()
    ca.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reqs", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 32, 40, 48, 52, 64, 96, 128])
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--parts", action="store_true", help="also time the block's parts (per layer: attn, tails, MLP)")
    a = p.parse_args()
    from tests.multi_gpu import run_on_2_gpus
    run_on_2_gpus(worker, a.reqs, a.rounds, not a.no_check, a.parts, timeout=3600)
