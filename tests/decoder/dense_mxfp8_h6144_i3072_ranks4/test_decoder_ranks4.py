# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four spawned ranks: ``DenseDecoderRanks4``'s calls equal the per-kernel calls they wrap with the code layout of the
consumer GEMM (bit for bit), the fc in 2048-row pieces equals the torch reference (cosine) and the per-piece calls;
``forward`` and ``store_context_kv`` equal the per-call loop and chain bit for bit at 1-2048 rows."""
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, H, V4 = 4, 6144, 50016


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _real(q, s, m, tile):
    """The written bytes of codes / scales of m rows (padding rows stay unwritten)."""
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
        aiter_scale_offsets,
        tile_code_offsets,
    )
    c = q.view(torch.uint8)
    c = c.reshape(-1)[tile_code_offsets(m, H, q.device)] if tile else c[:m]
    return c, s.reshape(-1)[aiter_scale_offsets(m, H, s.device)]


def _worker(rank, port):
    torch.set_num_threads(8)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10), device_id=dev)
    group = dist.new_group(backend="gloo")
    from amd_kernels import warm_up
    warm_up.set_auto(False)
    from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4 import DenseDecoderRanks4
    from amd_kernels.embedding.vocab_gather_gemma_mxfp8_h6144 import vocab_gather_gemma_mxfp8
    from amd_kernels.gemm.mxfp8_common import gemm as G
    g = torch.Generator(device=dev).manual_seed(11)  # the same replicated tensors on every rank
    fc_w = (torch.randn(H, 5 * H, device=dev, generator=g) * 0.01).bfloat16()
    emb = (torch.randn(V4, H, device=dev, generator=torch.Generator(device=dev).manual_seed(20 + rank))).bfloat16()
    blk = DenseDecoderRanks4(rank, group, dev, fc_weight=fc_w, embedding=emb)
    nw = (torch.randn(H, device=dev, generator=g) * 0.1).bfloat16()
    for m in (1, 7, 8, 21, 56, 224, 448, 1792, 2048, 2049, 4100):
        taps = [torch.randn(m + 3, H, device=dev, generator=g).bfloat16() for _ in range(5)]
        out = torch.empty(m + 3, H, dtype=torch.bfloat16, device=dev)
        blk.fc(taps, m, out)
        want = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
        for i in range(0, m, 2048):
            j = min(i + 2048, m)
            blk.fc_comm([t[i:j] for t in taps], want[i:j])
        torch.cuda.synchronize()
        assert torch.equal(out[:m].view(torch.int16), want.view(torch.int16)), (rank, m)
        ref = (torch.cat([t[:m] for t in taps], 1).double() @ fc_w.double().T)
        cos = torch.nn.functional.cosine_similarity(out[:m].double().flatten(), ref.flatten(), dim=0)
        assert cos > 0.9999, (rank, m, float(cos))
        if m <= 2048:
            assert torch.equal(blk.fc_cat(torch.cat([t[:m] for t in taps], 1)).view(torch.int16),
                               out[:m].view(torch.int16))
        if m > 2048:
            continue
        # embedding + norm + quant, all-reduce + norm (+ quant) for a consumer of (N, K) = qkv / gate_up shapes
        ids = torch.randint(0, 4 * V4, (m,), device=dev, generator=g)
        for shape in ((2304, H), (2 * 3072, H)):
            tile = G.prequant_code_tile_rows(m, *shape, torch.bfloat16)
            got = blk.embed_norm_quant(ids, nw, 1e-6, shape)
            want = vocab_gather_gemma_mxfp8(ids, blk.vocab, nw, 1e-6, tile)
            x = torch.randn(m, H, device=dev, generator=torch.Generator(device=dev).manual_seed(m + rank)).bfloat16()
            res = torch.randn(m, H, device=dev, generator=g).bfloat16()
            got2 = blk.ar_norm_quant(x, res, nw, 1e-6, shape)
            want2 = blk.ar.ar_add_rmsnorm_mxfp8(x, res, nw, 1e-6, tile, owned_residual=True)
            torch.cuda.synchronize()
            lo, hi = blk.ar.owned_rows(m) if m >= 32 else (0, m)  # the owner form writes its residual rows only
            for g_, w_, rows in ((got, want, slice(0, m)), (got2, want2, slice(lo, hi))):
                assert g_[0].shape == w_[0].shape and g_[1].shape == w_[1].shape, (rank, m, shape)
                for a, b in zip(_real(g_[0], g_[1], m, tile), _real(w_[0], w_[1], m, tile)):
                    assert torch.equal(a, b), (rank, m, shape)
                assert torch.equal(g_[2][rows].view(torch.int16), w_[2][rows].view(torch.int16)), (rank, m, shape)
        got3 = blk.ar_norm(x, res, nw, 1e-6)
        want3 = blk.ar.ar_add_rmsnorm(x, res, nw, 1e-6, owned_residual=True)
        torch.cuda.synchronize()
        assert torch.equal(got3[0].view(torch.int16), want3[0].view(torch.int16)), (rank, m)
        assert torch.equal(got3[1][lo:hi].view(torch.int16), want3[1][lo:hi].view(torch.int16)), (rank, m)
    blk.close()
    dist.barrier()
    dist.destroy_process_group()


ROWS = (1, 7, 8, 9, 56, 224, 448, 1792, 2048)
QKV, CTX_N, EPS, SWIGLU = (2304, H), 1280, 1e-6, (1.702, 1.0, 7.0)


def _mxfp8_weight(n, k, g):
    from amd_kernels.gemm.mxfp8_common import gemm as G
    codes = (torch.randn(n, k, device=g.device, generator=g) * 0.6).to(torch.float8_e4m3fn)
    exp = torch.randint(117, 122, (n, k // 32), device=g.device, generator=g).to(torch.uint8)
    return G.preshuffle_mxfp8_weight(codes, exp)


def _loop(blk, ids, layers, attn, final, form):
    """The per-call sequence ``forward`` replaces."""
    from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4 import mxfp8_swiglu_oai_mlp
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


def _forward_worker(rank, port):
    torch.set_num_threads(8)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10), device_id=dev)
    group = dist.new_group(backend="gloo")
    from amd_kernels import warm_up
    warm_up.set_auto(False)
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import (
        kv_norm_rope_insert_multilayer,
    )
    from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4 import (DenseDecoderRanks4, DenseLayer,
                                                                    gemma_rmsnorm_mxfp8_linear)
    from amd_kernels.gemm.mxfp8_common import gemm as G
    g = torch.Generator(device=dev).manual_seed(11)
    fc_w = (torch.randn(H, 5 * H, device=dev, generator=g) * 0.01).bfloat16()
    gr = torch.Generator(device=dev).manual_seed(30 + rank)  # this rank's shards
    emb = torch.randn(V4, H, device=dev, generator=gr).bfloat16()
    blk = DenseDecoderRanks4(rank, group, dev, fc_weight=fc_w, embedding=emb)
    taps = [torch.randn(16, H, device=dev, generator=g).bfloat16() for _ in range(5)]
    out = torch.empty(16, H, dtype=torch.bfloat16, device=dev)
    for bad in (taps[:4], taps[:4] + [torch.randn(16, H + 1, device=dev).bfloat16()[:, :H]]):
        with pytest.raises(AssertionError):
            blk.fc(bad, 16, out)

    def norm():
        return (torch.randn(H, device=dev, generator=g) * 0.1).bfloat16()

    layers = [DenseLayer(norm(), QKV, norm(), *_mxfp8_weight(2 * 3072, H, gr), *_mxfp8_weight(H, 3072, gr))
              for _ in range(3)]
    final = norm()
    ctx_w, ctx_ws = _mxfp8_weight(CTX_N, H, g)
    k_norm = (torch.randn(5, 128, device=dev, generator=g) * 0.1).bfloat16()
    cos_sin = (torch.rand(4096, 64, device=dev, generator=g) * 2 - 1).bfloat16()
    ks = [torch.tensor([0.5 + 0.1 * i], device=dev) for i in range(5)]
    vs = [torch.tensor([0.4 + 0.1 * i], device=dev) for i in range(5)]
    for m in ROWS:
        ids = torch.randint(0, 4 * V4, (m,), device=dev, generator=g)
        outs = [(torch.randn(m, H, device=dev, generator=gr) * 0.1).bfloat16() for _ in layers]
        tile = G.prequant_code_tile_rows(m, *QKV, torch.bfloat16)
        seen = {"forward": [], "loop": []}

        def attn(key):
            return lambda i, q, s: (seen[key].append(_real(q, s, m, tile)), outs[i])[1]

        got = blk.forward(ids, layers, attn("forward"), final, EPS, SWIGLU)
        want = _loop(blk, ids, layers, attn("loop"), final, "bf16_chain")
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (rank, m)
        for a, b in zip(seen["forward"], seen["loop"], strict=True):
            assert all(torch.equal(x, y) for x, y in zip(a, b)), (rank, m)
        # the context K / V chain: norm + GEMM, then the multi-layer insert
        states = torch.randn(m, H, device=dev, generator=g).bfloat16()
        hn = norm()
        pos = torch.randint(0, 4096, (m,), device=dev, generator=g)
        slots = torch.randperm(24 * 128, device=dev, generator=g)[:m]
        slots[1::5] = -1
        caches = [torch.zeros(24, 1, 128, 256, dtype=torch.uint8, device=dev) for _ in range(5)]
        ref = [c.clone() for c in caches]
        blk.store_context_kv(states, hn, EPS, ctx_w, ctx_ws, pos, k_norm, EPS, cos_sin, caches, slots, ks, vs)
        kv = gemma_rmsnorm_mxfp8_linear(states, hn, EPS, ctx_w, ctx_ws)
        kv_norm_rope_insert_multilayer(kv, pos, k_norm, cos_sin, ref, [slots] * 5, ks, vs, EPS)
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(caches, ref)), (rank, m)
        assert any(bool(c.any()) for c in caches), (rank, m)
    blk.close()
    dist.barrier()
    dist.destroy_process_group()


def test_decoder_ranks4_forward_context_kv():
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires four gfx950 GPUs")
    mp.spawn(_forward_worker, args=(_port(),), nprocs=WORLD, join=True)


def test_decoder_ranks4_calls():
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires four gfx950 GPUs")
    mp.spawn(_worker, args=(_port(),), nprocs=WORLD, join=True)
