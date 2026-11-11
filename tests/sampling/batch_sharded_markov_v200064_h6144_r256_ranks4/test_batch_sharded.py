# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four spawned ranks: the block (``BatchShardedSampling``) against the per-kernel entries it sequences, bit for bit;
its target logits against the torch reference; graph replay; the first-call warm-up compiles every variant."""
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, V, H, RANK, N = 4, 200064, 6144, 256, 50016
MAX_REQS, SLOTS = 64, 96


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init(rank, port):
    torch.set_num_threads(8)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10), device_id=dev)
    return dev, dist.new_group(backend="gloo")


def _shard(rank, dev):
    from amd_kernels.gemm.bf16_n50016_k6144 import shuffle_weight
    g = torch.Generator(device=dev).manual_seed(500 + rank)
    w = (torch.randn(N, H, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    return w, shuffle_weight(w)


def _draft_inputs(steps, dev):
    """The same on every rank (one seed): the speculator's buffers for MAX_REQS requests."""
    g = torch.Generator(device=dev).manual_seed(1234 + steps)
    q = steps + 1
    t = MAX_REQS * q
    hidden = torch.randn(t, H, device=dev, generator=g).to(torch.bfloat16)
    req = torch.arange(MAX_REQS, device=dev).repeat_interleave(steps)
    step = torch.arange(steps, device=dev).repeat(MAX_REQS)
    sample_indices = (req * q + step + 1).to(torch.int64)
    slot_of = torch.randperm(SLOTS, device=dev, generator=g)[:MAX_REQS]
    sample_idx_mapping = slot_of[req].to(torch.int32)
    sample_pos = torch.randint(1, 1 << 20, (MAX_REQS,), device=dev, generator=g)[req] + step
    input_ids = torch.randint(0, V, (t,), device=dev, generator=g).to(torch.int32)
    anchor = torch.arange(MAX_REQS, device=dev) * q
    temperature = torch.tensor([0.0, 1.0, 0.7, 1.3], device=dev)[torch.randint(0, 4, (SLOTS,), device=dev,
                                                                                generator=g)]
    seeds = torch.randint(0, 1 << 62, (SLOTS,), device=dev, generator=g)
    w1 = (torch.randn(V, RANK, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(V, RANK, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    return dict(hidden=hidden, sample_indices=sample_indices, sample_idx_mapping=sample_idx_mapping,
                sample_pos=sample_pos, input_ids=input_ids, anchor=anchor, temperature=temperature, seeds=seeds,
                w1=w1, w2=w2)


def _per_kernel_markov(comm_a2a, comm_gather, tables, steps, num_reqs, x, lm, d, cache, draft_tokens):
    """The chain as separate per-kernel entry calls (the sequence the block replaces)."""
    from amd_kernels.gemm.bf16_n50016_k6144.host import bf16_gemm_n50016_k6144
    from amd_kernels.gemm.bf16_n200064_k256.host import bf16_gemm_packed_rows, pack_bf16_rows, padded_rows
    from amd_kernels.sampling.gumbel_combine_markov_embed_v200064_r256.host import gumbel_combine_markov_embed_packed
    from amd_kernels.sampling.gumbel_partials_bias_bf16_v200064.host import gumbel_partials_bias
    t = tables[num_reqs]
    cap, dev = t.cap, x.device
    rows = d["sample_indices"][t.send_rows]
    base = torch.empty((cap * steps, V), dtype=torch.bfloat16, device=dev)
    bf16_gemm_n50016_k6144(x, lm, shuffled=True, row_indices=rows, publish=comm_a2a.slot())
    comm_a2a.pull(rows.numel(), t.counts, base)
    base = base.view(cap, steps, V)
    slots = torch.where(t.valid, d["sample_idx_mapping"][t.slot_rows], -1).to(torch.int32)
    pos = d["sample_pos"][t.slot_rows]
    prev = d["input_ids"][t.anchor_rows].to(torch.int64)
    rows_ar = torch.arange(cap, dtype=torch.int64, device=dev)
    packed = torch.empty((padded_rows(cap), RANK), dtype=torch.bfloat16, device=dev)
    pack_bf16_rows(d["w1"], prev, packed)
    bias = torch.empty((cap, V), dtype=torch.bfloat16, device=dev)
    lmax = torch.empty((cap, 196), dtype=torch.float32, device=dev)
    larg = torch.empty((cap, 196), dtype=torch.int64, device=dev)
    tokens = torch.empty((cap, 7), dtype=torch.int64, device=dev)
    prev_out = torch.empty(cap, dtype=torch.int64, device=dev)
    for i in range(steps):
        bf16_gemm_packed_rows(packed, d["w2"], cap, bias)
        gumbel_partials_bias(base[:, i], slots[i], d["temperature"], d["seeds"], pos[i], cache, i, bias, rows_ar,
                             lmax, larg, pos_offset=-1)
        gumbel_combine_markov_embed_packed(lmax, larg, rows_ar, d["w1"], tokens, i, prev_out, packed)
    full = torch.empty((num_reqs, 7), dtype=torch.int64, device=dev)
    comm_gather(tokens, t.gathered_src, cap, full)
    draft_tokens[:num_reqs].copy_(full[:, :steps])


def _worker(rank, port):
    dev, group = _init(rank, port)
    from amd_kernels import warm_up
    warm_up.set_auto(False)  # single sizes here; the warm-up test runs it (a spawned child may skip tests/__init__)
    from amd_kernels.collectives.all_gather_i64_c7_ranks4 import I64C7AllGather
    from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4 import RowAllToAll
    from amd_kernels.gemm.bf16_n50016_k6144.reference import bf16_gemm_n50016_k6144_reference
    from amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4 import BatchShardedSampling
    from amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4.reference import owner_order_reference
    w, lm = _shard(rank, dev)
    # target logits: the block vs the reference of every owner's rows (cosine) and vs the per-kernel pair (bits)
    s = BatchShardedSampling(rank, group, dev, max_reqs=MAX_REQS, target_rows=MAX_REQS * 8)
    old_a2a = RowAllToAll(rank, group, dev, max_rows=MAX_REQS * 8)
    g = torch.Generator(device=dev).manual_seed(7)
    hidden = torch.randn(MAX_REQS * 8, H, device=dev, generator=g).to(torch.bfloat16)
    for m in (1, 7, 64, 129, 300, 512):
        rows = torch.randperm(hidden.shape[0], device=dev, generator=g)[:m].to(torch.int64)
        counts = [m // WORLD + int(r < m % WORLD) for r in range(WORLD)]
        out = s.target_logits(hidden, lm, rows, counts, counts[rank])
        from amd_kernels.gemm.bf16_n50016_k6144.host import bf16_gemm_n50016_k6144
        bf16_gemm_n50016_k6144(hidden, lm, shuffled=True, row_indices=rows, publish=old_a2a.slot())
        ref_out = torch.empty_like(out)
        old_a2a.pull(m, torch.tensor(counts, dtype=torch.int32, device=dev), ref_out)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16)), (rank, m)
        # owner r gets owner-sorted rows [sum(counts[:r]), sum(counts[:r + 1])); shard s at columns s * 50016
        lo = sum(counts[:rank])
        mine = hidden[rows[lo:lo + counts[rank]]]
        part = bf16_gemm_n50016_k6144_reference(mine, w)  # this rank's shard of its own rows
        if counts[rank]:
            cos = torch.nn.functional.cosine_similarity(out[:, rank * N:(rank + 1) * N].float().flatten(),
                                                        part.float().flatten(), dim=0)
            assert cos > 0.9999, (rank, m, float(cos))
    none = s.target_logits(hidden, lm, torch.empty(0, dtype=torch.int64, device=dev), [0] * WORLD, 0)
    assert none.shape == (0, V) and none.dtype == torch.bfloat16
    # sampler outputs: a known pattern in batch order
    num_reqs = 13
    n_loc = len(owner_order_reference(num_reqs)[rank])
    tok = torch.arange(n_loc * 8, device=dev).view(n_loc, 8) + 1000 * rank
    ns = torch.full((n_loc,), 3 + rank, dtype=torch.int64, device=dev)
    nr = torch.full((n_loc,), rank, dtype=torch.int64, device=dev)
    nn = torch.full((n_loc,), 2 * rank, dtype=torch.int64, device=dev)
    cu = torch.arange(n_loc + 1, dtype=torch.int32, device=dev)  # one NaN-count row per request
    cap = -(-num_reqs // WORLD)
    req = torch.arange(num_reqs, device=dev)
    src = ((req % WORLD) * cap + req // WORLD).to(torch.int64)
    out_t, out_s, out_r, out_n = s.gather_sampler_output(tok, ns, nr, nn, cu, src, cap, num_reqs, 4, True)
    torch.cuda.synchronize()
    for i in range(num_reqs):
        owner, local = i % WORLD, i // WORLD
        assert out_t[i].tolist() == [1000 * owner + local * 8 + j for j in range(4)], (rank, i)
        assert (int(out_s[i]), int(out_r[i]), int(out_n[i])) == (3 + owner, owner, 2 * owner), (rank, i)
    # the Markov draft: the block vs the per-kernel chain, every request count of a sample, DSpark 3 and 7
    for steps in (3, 7):
        d = _draft_inputs(steps, dev)
        s.setup_markov(steps, MAX_REQS, d["anchor"], d["w1"], d["w2"])
        old_a2a_d = RowAllToAll(rank, group, dev, max_rows=WORLD * (MAX_REQS // WORLD) * steps)
        old_gather = I64C7AllGather(rank, group, dev)
        for num_reqs in (1, 2, 4, 5, 13, 31, 32, 48, 63, 64):
            cache_a = torch.zeros(SLOTS, steps, V, dtype=torch.bfloat16, device=dev)
            cache_b = torch.zeros_like(cache_a)
            ids_a = torch.full((MAX_REQS, steps), -1, dtype=torch.int64, device=dev)
            ids_b = ids_a.clone()
            s.markov_draft(num_reqs, d["hidden"], lm, d["sample_indices"], d["sample_idx_mapping"], d["sample_pos"],
                           d["input_ids"], d["temperature"], d["seeds"], cache_a, ids_a)
            _per_kernel_markov(old_a2a_d, old_gather, s.markov.tables, steps, num_reqs, d["hidden"], lm, d, cache_b,
                               ids_b)
            torch.cuda.synchronize()
            assert torch.equal(ids_a, ids_b), (rank, steps, num_reqs)
            diff = (cache_a.view(torch.int16) != cache_b.view(torch.int16))
            if diff.any():
                where = diff.any(dim=2).nonzero().tolist()
                na = (cache_a.view(torch.int16) != 0).any(dim=2).nonzero().tolist()
                nb = (cache_b.view(torch.int16) != 0).any(dim=2).nonzero().tolist()
                mine = owner_order_reference(num_reqs)[rank]
                exp = sorted(set(d["sample_idx_mapping"].view(MAX_REQS, steps)[mine, 0].tolist()))
                raise AssertionError((rank, steps, num_reqs, "diff", where[:20], "a", na[:20], "b", nb[:20], "mine",
                                      exp, int(diff.sum())))
            assert (ids_a[:num_reqs] >= 0).all() and (ids_a[:num_reqs] < V).all()
        # graph replay with changed inputs (the tables and the request count are fixed per graph)
        num_reqs = 13
        cache_g = torch.zeros(SLOTS, steps, V, dtype=torch.bfloat16, device=dev)
        ids_g = torch.full((MAX_REQS, steps), -1, dtype=torch.int64, device=dev)
        args = (d["hidden"], lm, d["sample_indices"], d["sample_idx_mapping"], d["sample_pos"], d["input_ids"],
                d["temperature"], d["seeds"])
        s.markov_draft(num_reqs, *args, cache_g, ids_g)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s.markov_draft(num_reqs, *args, cache_g, ids_g)
        for k in range(3):
            d["hidden"].copy_(torch.randn(d["hidden"].shape, device=dev, generator=torch.Generator(device=dev)
                                          .manual_seed(k)).to(torch.bfloat16))
            d["seeds"].add_(1)
            graph.replay()
            cache_e = torch.zeros_like(cache_g)
            ids_e = torch.full_like(ids_g, -1)
            s.markov_draft(num_reqs, *args, cache_e, ids_e)
            torch.cuda.synchronize()
            assert torch.equal(ids_g, ids_e), (rank, steps, "graph", k)
            mine = owner_order_reference(num_reqs)[rank]
            slots = d["sample_idx_mapping"].view(MAX_REQS, steps)[mine, 0].long()
            assert torch.equal(cache_g[slots].view(torch.int16), cache_e[slots].view(torch.int16)), (rank, k)
        del graph
        old_a2a_d.close()
        old_gather.close()
    old_a2a.close()
    s.close()
    dist.barrier()
    dist.destroy_process_group()


def _warm_worker(rank, port):
    """Each mode's first call warms it up; later calls at other sizes compile nothing."""
    dev, group = _init(rank, port)
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4 import BatchShardedSampling
    warm_up.set_auto(True)
    warm_up.set_workers(4)
    _, lm = _shard(rank, dev)
    steps, max_reqs = 7, 48
    d = _draft_inputs(steps, dev)
    s = BatchShardedSampling(rank, group, dev, max_reqs=max_reqs, target_rows=max_reqs * 8)
    s.setup_markov(steps, max_reqs, d["anchor"], d["w1"], d["w2"])
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    cache = torch.zeros(SLOTS, steps, V, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(MAX_REQS, steps, dtype=torch.int64, device=dev)
    args = (d["hidden"], lm, d["sample_indices"], d["sample_idx_mapping"], d["sample_pos"], d["input_ids"],
            d["temperature"], d["seeds"], cache, ids)
    for num_reqs in (1, 2, 3, 5, 6, 9, 10, 11, 17, 21, 33, 40, 47, 48):
        s.markov_draft(num_reqs, *args)
        m = num_reqs * 8
        rows = torch.arange(m, dtype=torch.int64, device=dev)
        counts = [m // WORLD + int(r < m % WORLD) for r in range(WORLD)]
        s.target_logits(d["hidden"], lm, rows, counts, counts[rank])
        n_loc = len(range(rank, num_reqs, WORLD))
        t = torch.zeros(n_loc, 8, dtype=torch.int64, device=dev)
        c = torch.ones(n_loc, dtype=torch.int32, device=dev)
        cu = torch.arange(n_loc + 1, dtype=torch.int32, device=dev)
        cap = -(-num_reqs // WORLD)
        req = torch.arange(num_reqs, device=dev)
        src = ((req % WORLD) * cap + req // WORLD).to(torch.int64)
        for nans in (False, True):
            s.gather_sampler_output(t, c, c, c, cu, src, cap, num_reqs, 8, nans)
    torch.cuda.synchronize()
    MlirCompiler.compile = original
    assert late == [] and warm_up.late_compiles() == 0, (rank, len(late), warm_up.late_compiles())
    assert len([k for k in warm_up.warmed() if k[0].startswith("sampling/batch_sharded")]) == 2, warm_up.warmed()
    s.close()
    dist.barrier()
    dist.destroy_process_group()


def _need_gpus():
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires four gfx950 GPUs")


def test_block_matches_per_kernel_entries():
    _need_gpus()
    mp.spawn(_worker, args=(_port(),), nprocs=WORLD, join=True)


def test_first_call_warm_up_compiles_every_variant(tmp_path, monkeypatch):
    _need_gpus()
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "flydsl"))
    mp.spawn(_warm_worker, args=(_port(),), nprocs=WORLD, join=True)


def test_sampling_blocker():
    from amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4 import sampling_blocker

    for max_reqs, per_req in ((4, 1), (256, 8), (256, 1), (32, 8)):
        assert sampling_blocker(WORLD, max_reqs, per_req) is None, (max_reqs, per_req)
    for world, max_reqs, per_req in ((2, 64, 4), (8, 64, 4), (4, 3, 1), (4, 257, 1), (4, 64, 0), (4, 64, 9),
                                     (4, 512, 8)):
        assert sampling_blocker(world, max_reqs, per_req) is not None, (world, max_reqs, per_req)


def test_markov_blocker():
    from amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4 import markov_blocker

    w = torch.empty(V, RANK, dtype=torch.bfloat16)
    cache = torch.empty(16, 3, V, dtype=torch.bfloat16)
    assert markov_blocker(3, 64, w, w) is None
    assert markov_blocker(3, 64, w, w, cache=cache) is None
    for steps, reqs, w1 in ((0, 64, w), (8, 64, w), (3, 257, w), (3, 64, w.float()), (3, 64, w[:-1])):
        assert markov_blocker(steps, reqs, w1, w) is not None, (steps, reqs)
    for bad in (cache.float(), cache[:, :2], torch.empty(16, 3, V - 1, dtype=torch.bfloat16)):
        assert markov_blocker(3, 64, w, w, cache=bad) is not None
    for kw in ("fp64_noise", "confidences", "logit_transform", "reduced_vocab"):
        assert markov_blocker(3, 64, w, w, **{kw: True}) is not None, kw
    assert markov_blocker(3, 64, w, w, lm_weight_shuffled=False) is not None
