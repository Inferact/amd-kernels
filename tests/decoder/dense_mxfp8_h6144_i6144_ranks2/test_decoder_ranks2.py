# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks: ``DenseDecoderRanks2``'s calls equal the per-kernel calls they wrap with the code layout of the
consumer GEMM (bit for bit), the fc in 2048-row pieces equals the torch reference (cosine) and the per-piece calls;
``forward`` (eager and graph replay) and ``store_context_kv`` equal the per-call loop and chain bit for bit at
1-2048 rows; after the block's construction and the entries' first calls nothing compiles."""
import os

import pytest
import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_2_gpus

WORLD, H, V2, I = 2, 6144, 100032, 6144
QKV, CTX_N, EPS, SWIGLU = (4608, H), 2560, 1e-6, (1.702, 1.0, 7.0)
KV_HEADS = 2


def _real(q, s, m, tile):
    """The written bytes of codes / scales of m rows (padding rows stay unwritten)."""
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
        aiter_scale_offsets,
        tile_code_offsets,
    )
    c = q.view(torch.uint8)
    c = c.reshape(-1)[tile_code_offsets(m, H, q.device)] if tile else c[:m]
    return c, s.reshape(-1)[aiter_scale_offsets(m, H, s.device)]


def _check(errors, ok, what):
    if not ok:
        errors.append(what)


def _calls_worker(rank, errors):
    from amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2 import DenseDecoderRanks2
    from amd_kernels.embedding.vocab_gather_gemma_mxfp8_h6144 import vocab_gather_gemma_mxfp8
    from amd_kernels.gemm.mxfp8_common import gemm as G
    dev = torch.device("cuda", rank)
    g = torch.Generator(device=dev).manual_seed(11)  # the same replicated tensors on both ranks
    fc_w = (torch.randn(H, 5 * H, device=dev, generator=g) * 0.01).bfloat16()
    emb = (torch.randn(V2, H, device=dev, generator=torch.Generator(device=dev).manual_seed(20 + rank))).bfloat16()
    blk = DenseDecoderRanks2(rank, dist.group.WORLD, dev, fc_weight=fc_w, embedding=emb)
    nw = (torch.randn(H, device=dev, generator=g) * 0.1).bfloat16()
    for m in (1, 7, 8, 21, 56, 224, 364, 448, 1792, 2048, 2049, 4100):
        taps = [torch.randn(m + 3, H, device=dev, generator=g).bfloat16() for _ in range(5)]
        out = torch.empty(m + 3, H, dtype=torch.bfloat16, device=dev)
        blk.fc(taps, m, out)
        want = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
        for i in range(0, m, 2048):
            j = min(i + 2048, m)
            blk.fc_comm([t[i:j] for t in taps], want[i:j])
        torch.cuda.synchronize()
        _check(errors, torch.equal(out[:m].view(torch.int16), want.view(torch.int16)), f"rank {rank} m {m}: fc pieces")
        ref = (torch.cat([t[:m] for t in taps], 1).double() @ fc_w.double().T)
        cos = torch.nn.functional.cosine_similarity(out[:m].double().flatten(), ref.flatten(), dim=0)
        _check(errors, cos > 0.9999, f"rank {rank} m {m}: fc cosine {float(cos)}")
        if m > 2048:
            continue
        _check(errors, torch.equal(blk.fc_cat(torch.cat([t[:m] for t in taps], 1)).view(torch.int16),
                                   out[:m].view(torch.int16)), f"rank {rank} m {m}: fc_cat")
        # embedding + norm + quant, all-reduce + norm (+ quant) for a consumer of (N, K) = qkv / gate_up shapes
        ids = torch.randint(0, WORLD * V2, (m,), device=dev, generator=g)
        for shape in (QKV, (2 * I, H)):
            tile = G.prequant_code_tile_rows(m, *shape, torch.bfloat16)
            got = blk.embed_norm_quant(ids, nw, EPS, shape)
            want = vocab_gather_gemma_mxfp8(ids, blk.vocab, nw, EPS, tile)
            x = torch.randn(m, H, device=dev, generator=torch.Generator(device=dev).manual_seed(m + rank)).bfloat16()
            res = torch.randn(m, H, device=dev, generator=g).bfloat16()
            got2 = blk.ar_norm_quant(x, res, nw, EPS, shape)
            want2 = blk.ar.ar_add_rmsnorm_mxfp8(x, res, nw, EPS, tile, owned_residual=True)
            torch.cuda.synchronize()
            lo, hi = blk.ar.owned_rows(m) if m >= 24 else (0, m)  # the owner form writes its residual rows only
            for name, g_, w_, rows in (("embed", got, want, slice(0, m)), ("ar", got2, want2, slice(lo, hi))):
                same = g_[0].shape == w_[0].shape and g_[1].shape == w_[1].shape
                same = same and all(torch.equal(a, b) for a, b in zip(_real(g_[0], g_[1], m, tile),
                                                                      _real(w_[0], w_[1], m, tile)))
                same = same and torch.equal(g_[2][rows].view(torch.int16), w_[2][rows].view(torch.int16))
                _check(errors, same, f"rank {rank} m {m} {shape}: {name}")
        got3 = blk.ar_norm(x, res, nw, EPS)
        want3 = blk.ar.ar_add_rmsnorm(x, res, nw, EPS, owned_residual=True)
        torch.cuda.synchronize()
        _check(errors, torch.equal(got3[0].view(torch.int16), want3[0].view(torch.int16))
               and torch.equal(got3[1][lo:hi].view(torch.int16), want3[1][lo:hi].view(torch.int16)),
               f"rank {rank} m {m}: ar_norm")
    blk.close()


ROWS = (1, 7, 8, 9, 16, 17, 56, 224, 280, 364, 448, 896, 1792, 2048)


def _mxfp8_weight(n, k, g):
    from amd_kernels.gemm.mxfp8_common import gemm as G
    codes = (torch.randn(n, k, device=g.device, generator=g) * 0.6).to(torch.float8_e4m3fn)
    exp = torch.randint(117, 122, (n, k // 32), device=g.device, generator=g).to(torch.uint8)
    return G.preshuffle_mxfp8_weight(codes, exp)


def _loop(blk, ids, layers, attn, final, form):
    """The per-call sequence ``forward`` replaces."""
    from amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2 import mxfp8_swiglu_oai_mlp
    m = ids.numel()
    q, s, res = blk.embed_norm_quant(ids, layers[0].input_norm, EPS, layers[0].qkv_shape)
    for i, layer in enumerate(layers):
        o = attn(i, q, s)
        q, s, res = blk.ar_norm_quant(o, res, layer.post_norm, EPS, layer.gate_up_w.shape)
        d = mxfp8_swiglu_oai_mlp(q, s, m, layer.gate_up_w, layer.gate_up_ws, layer.down_w, layer.down_ws, *SWIGLU,
                                 form=form)
        if i + 1 < len(layers):
            q, s, res = blk.ar_norm_quant(d, res, layers[i + 1].input_norm, EPS, layers[i + 1].qkv_shape)
    return blk.ar_norm(d, res, final, EPS)[0]


def _setup(rank, dev):
    from amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2 import DenseDecoderRanks2, DenseLayer
    g = torch.Generator(device=dev).manual_seed(11)
    fc_w = (torch.randn(H, 5 * H, device=dev, generator=g) * 0.01).bfloat16()
    gr = torch.Generator(device=dev).manual_seed(30 + rank)  # this rank's shards
    emb = torch.randn(V2, H, device=dev, generator=gr).bfloat16()
    blk = DenseDecoderRanks2(rank, dist.group.WORLD, dev, fc_weight=fc_w, embedding=emb)

    def norm():
        return (torch.randn(H, device=dev, generator=g) * 0.1).bfloat16()

    layers = [DenseLayer(norm(), QKV, norm(), *_mxfp8_weight(2 * I, H, gr), *_mxfp8_weight(H, I, gr))
              for _ in range(3)]
    return blk, layers, norm, g, gr


def _forward_worker(rank, errors):
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import (
        kv_norm_rope_insert_multilayer,
    )
    from amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2 import gemma_rmsnorm_mxfp8_linear
    from amd_kernels.gemm.mxfp8_common import gemm as G
    dev = torch.device("cuda", rank)
    blk, layers, norm, g, gr = _setup(rank, dev)
    taps = [torch.randn(16, H, device=dev, generator=g).bfloat16() for _ in range(5)]
    out = torch.empty(16, H, dtype=torch.bfloat16, device=dev)
    for bad in (taps[:4], taps[:4] + [torch.randn(16, H + 1, device=dev).bfloat16()[:, :H]]):
        try:
            blk.fc(bad, 16, out)
            errors.append(f"rank {rank}: fc accepted bad taps")
        except AssertionError:
            pass
    final = norm()
    ctx_w, ctx_ws = _mxfp8_weight(CTX_N, H, g)
    k_norm = (torch.randn(5, 128, device=dev, generator=g) * 0.1).bfloat16()
    cos_sin = (torch.rand(4096, 64, device=dev, generator=g) * 2 - 1).bfloat16()
    ks = [torch.tensor([0.5 + 0.1 * i], device=dev) for i in range(5)]
    vs = [torch.tensor([0.4 + 0.1 * i], device=dev) for i in range(5)]
    for m in ROWS:
        ids = torch.randint(0, WORLD * V2, (m,), device=dev, generator=g)
        outs = [(torch.randn(m, H, device=dev, generator=gr) * 0.1).bfloat16() for _ in layers]
        tile = G.prequant_code_tile_rows(m, *QKV, torch.bfloat16)
        seen = {"forward": [], "loop": [], "graph": []}

        def attn(key):
            return lambda i, q, s: (seen[key].append(_real(q, s, m, tile)), outs[i])[1]

        got = blk.forward(ids, layers, attn("forward"), final, EPS, SWIGLU)
        want = _loop(blk, ids, layers, attn("loop"), final, "bf16_chain")
        torch.cuda.synchronize()
        _check(errors, torch.equal(got.view(torch.int16), want.view(torch.int16)), f"rank {rank} m {m}: forward")
        for a, b in zip(seen["forward"], seen["loop"], strict=True):
            _check(errors, all(torch.equal(x, y) for x, y in zip(a, b)), f"rank {rank} m {m}: qkv codes")
        if m in (7, 56, 364, 2048):
            graph = torch.cuda.CUDAGraph()
            with blk.capture(), torch.cuda.graph(graph):
                captured = blk.forward(ids, layers, lambda i, q, s: outs[i], final, EPS, SWIGLU)
            graph.replay()
            torch.cuda.synchronize()
            _check(errors, torch.equal(captured.view(torch.int16), want.view(torch.int16)),
                   f"rank {rank} m {m}: graph replay")
            del graph
        # the context K / V chain: norm + GEMM, then the multi-layer insert into the 2-KV-head caches
        states = torch.randn(m, H, device=dev, generator=g).bfloat16()
        hn = norm()
        pos = torch.randint(0, 4096, (m,), device=dev, generator=g)
        slots = torch.randperm(24 * 128, device=dev, generator=g)[:m]
        slots[1::5] = -1
        caches = [torch.zeros(24, KV_HEADS, 128, 256, dtype=torch.uint8, device=dev) for _ in range(5)]
        ref = [c.clone() for c in caches]
        blk.store_context_kv(states, hn, EPS, ctx_w, ctx_ws, pos, k_norm, EPS, cos_sin, caches, slots, ks, vs)
        kv = gemma_rmsnorm_mxfp8_linear(states, hn, EPS, ctx_w, ctx_ws)
        kv_norm_rope_insert_multilayer(kv, pos, k_norm, cos_sin, ref, [slots] * 5, ks, vs, EPS)
        torch.cuda.synchronize()
        _check(errors, all(torch.equal(a, b) for a, b in zip(caches, ref)), f"rank {rank} m {m}: context K/V")
        _check(errors, all(bool(c[:, h].any()) for c in caches for h in range(KV_HEADS)),
               f"rank {rank} m {m}: a KV head not written")
    blk.close()


def _compile_worker(rank, errors):
    """The block built (its comms warm up), the entries' first calls made: forward and store_context_kv compile
    nothing at the draft's row counts (fresh FlyDSL cache, so a variant missed reaches the compiler)."""
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels import warm_up
    from amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2 import gemma_rmsnorm_mxfp8_linear, mxfp8_swiglu_oai_mlp
    from amd_kernels.gemm.mxfp8_common import gemm as G
    warm_up.set_auto(True)
    try:
        dev = torch.device("cuda", rank)
        blk, layers, norm, g, gr = _setup(rank, dev)
        ctx_w, ctx_ws = _mxfp8_weight(CTX_N, H, g)
        k_norm = (torch.randn(5, 128, device=dev, generator=g) * 0.1).bfloat16()
        cos_sin = (torch.rand(4096, 64, device=dev, generator=g) * 2 - 1).bfloat16()
        ks = [torch.tensor([1.0], device=dev) for _ in range(5)]
        caches = [torch.zeros(24, KV_HEADS, 128, 256, dtype=torch.uint8, device=dev) for _ in range(5)]
        # the caller's warm-ups: the GEMM weights' (vLLM's linear integration) and the entries' first calls
        for w, ws in ((layers[0].gate_up_w, layers[0].gate_up_ws), (layers[0].down_w, layers[0].down_ws),
                      (ctx_w, ctx_ws)):
            G.warmup_mxfp8_gemm_weight(w, ws, 2048, torch.bfloat16)
        x = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
        q, s, _ = blk.embed_norm_quant(torch.zeros(1, dtype=torch.int64, device=dev), layers[0].post_norm, EPS,
                                       layers[0].gate_up_w.shape)
        mxfp8_swiglu_oai_mlp(q, s, 1, layers[0].gate_up_w, layers[0].gate_up_ws, layers[0].down_w,
                             layers[0].down_ws, *SWIGLU, form="bf16_chain")
        gemma_rmsnorm_mxfp8_linear(x, norm(), EPS, ctx_w, ctx_ws)
        blk.store_context_kv(x, norm(), EPS, ctx_w, ctx_ws, torch.zeros(1, dtype=torch.int64, device=dev), k_norm,
                             EPS, cos_sin, caches, torch.full((1,), -1, dtype=torch.int64, device=dev), ks, ks)
        torch.cuda.synchronize()
        late, original = [], MlirCompiler.compile

        def spy(*a, **k):
            if not warm_up._in_warm_up.get():
                late.append(m)
            return original(*a, **k)

        MlirCompiler.compile = spy
        try:
            final = norm()
            for m in (1, 2, 3, 7, 8, 14, 15, 16, 17, 21, 28, 35, 49, 56, 63, 70, 112, 140, 175, 224, 231, 280, 287,
                      336, 343, 364, 371, 448, 512, 700, 896, 1024, 1400, 1792, 2047, 2048):
                ids = torch.zeros(m, dtype=torch.int64, device=dev)
                o = torch.zeros(m, H, dtype=torch.bfloat16, device=dev)
                blk.forward(ids, layers, lambda i, q, s: o, final, EPS, SWIGLU)
                blk.store_context_kv(o, final, EPS, ctx_w, ctx_ws, torch.zeros(m, dtype=torch.int64, device=dev),
                                     k_norm, EPS, cos_sin, caches, torch.full((m,), -1, dtype=torch.int64, device=dev),
                                     ks, ks)
            torch.cuda.synchronize()
        finally:
            MlirCompiler.compile = original
        _check(errors, not late, f"rank {rank}: compiles after the warm-ups at rows {sorted(set(late))}")
        blk.close()
    finally:
        warm_up.set_auto(False)


def test_decoder_ranks2_calls():
    run_on_2_gpus(_calls_worker, timeout=1800)


def test_decoder_ranks2_forward_context_kv():
    run_on_2_gpus(_forward_worker, timeout=1800)


def test_decoder_ranks2_no_compile_after_warm_up(tmp_path):
    saved = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(tmp_path / "flydsl")  # new rank processes with a fresh cache
    try:
        run_on_2_gpus(_compile_worker, timeout=3600)
    finally:
        if saved is None:
            os.environ.pop("FLYDSL_RUNTIME_CACHE_DIR")
        else:
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = saved
