# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The phases of SeqParallelHandoffs on 2 ranks, bit for bit against their kernels run on one GPU per shard (every rank
rebuilds every rank's seeded inputs): the input quant + gather, the attention all-to-all + o_proj + norm (+ MoE
preparation, the routed payload's gather, the full shared expert of the own rows) + gather of a MoE and a dense layer,
the FFN reduce-scatter + next norm / quant + gather of both (MoE: owner-form and every-row shared terms), the final
reduce-scatter and the bf16 gather; at the served token counts (tails off the 32-row grid included). A MoE layer's
chain (hand-off, routed experts, owner-form tail) against the one-GPU composition, with the shared expert's side
stream delayed; with its join skipped the comparison must fail. Layers enqueued far ahead of the GPU must not grow the
allocator pool by the side stream's buffers (recorded on the side stream instead, they must). The MoE FFN's norm + quant has no one-GPU kernel: its
codes and scales meet ``norm/gemma_add_rmsnorm_h6144/reference.py``'s tolerances, its residual bit for bit."""

import contextlib

import pytest
import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_2_gpus

from .workload import H, Q, SERVED_TOKENS, WORLD, moe_weights, o_proj_weight, rand

EPS = 1e-6


def _expect(errors, what, got, want):
    if got.shape != want.shape:
        errors.append(f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}")
    elif not torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)):
        bad = (got.contiguous().view(torch.uint8) != want.contiguous().view(torch.uint8)).reshape(got.shape[0], -1)
        errors.append(f"{what}: rows {bad.any(dim=1).nonzero().flatten()[:6].tolist()} differ")


def _expect_mxfp8(errors, what, codes, scales, y64):
    """Row-major codes and AITER-shuffled scales of a shard against the fp64 norm (the reference's tolerances)."""
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import reference as R

    st = R.compare_mxfp8(codes.contiguous().view(torch.uint8), R.unshuffle_scales(scales, codes.shape[0]), y64)
    if (st["scale_max_diff"] > 1 or st["scale_mismatch"] > R.SCALE_MISMATCH or st["code_mismatch"] > R.CODE_MISMATCH
            or st["code_far"] or st["cos"] < R.MXFP8_COS):  # fmt: skip
        errors.append(f"{what}: {st}")


def _unpacked(attn, t, rows, p, dev):
    """Rank p's o_proj input from every rank's attention output: codes [rows, 2 Q], scales in the GEMM's layout."""
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import mxfp8_pack_q4096
    from amd_kernels.gemm.mxfp8_common.gemm import _preshuffle_activation_scale

    codes, scales = [], []
    for a in attn:
        packed = torch.zeros((WORLD * rows, Q + Q // 32), dtype=torch.uint8, device=dev)
        mxfp8_pack_q4096(a, out=packed[:t])
        block = packed[p * rows : (p + 1) * rows]
        codes.append(block[:, :Q])
        scales.append(block[:, Q:])
    return torch.cat(codes, 1).contiguous().view(torch.float8_e4m3fn), _preshuffle_activation_scale(
        torch.cat(scales, 1).contiguous()
    )


class _Delayed:
    """The SDMA phases with a ``between`` that delays this rank's stream by a rank-dependent time, as the other
    microbatch's compute would: the pushes and waits must order the bytes, not the timing."""

    def __init__(self, ph, rank):
        self.ph, self.cycles = ph, 50_000 * (1 + rank)

    def __getattr__(self, name):
        fn = getattr(self.ph, name)
        if name not in ("input_gather", "attention_moe", "attention_dense", "ffn_norm_gather", "ffn_reduce_scatter",
                        "gather"):  # fmt: skip
            return fn
        return lambda *a, **k: fn(*a, between=lambda: torch.cuda._sleep(self.cycles), **k)


def _worker(rank, errors, token_counts, sdma=False, full=True):
    from amd_kernels.collectives.pull_collectives_ranks2.reference import (
        reduce_scatter_reference,
        slots4_reduce_scatter_reference,
    )
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import SeqParallelHandoffs, shard_rows
    from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_e4m3_quantize_flydsl, mxfp8_gemm
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import prepare_mxfp4_moe, shared_expert_full
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm_mxfp8_row_major
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.host import gemma_add_rmsnorm
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import add_rmsnorm_reference

    dev = torch.device("cuda", rank)
    if sdma:
        from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import SdmaSeqParallelHandoffs

        ph = _Delayed(SdmaSeqParallelHandoffs(rank, dist.group.WORLD, dev, max(token_counts), compact=not full), rank)
    else:
        ph = SeqParallelHandoffs(rank, dist.group.WORLD, dev, max(token_counts))
    o_w, o_s = o_proj_weight(dev)
    mw = moe_weights(dev, rank, full)
    nw = rand((H,), 3, dev, 0.2)

    def o_proj(xq, xs):
        return mxfp8_gemm(xq, xs, o_w, o_s, torch.bfloat16, activation_scale_preshuffled=True)

    for t in token_counts:
        rows = shard_rows(t)
        padded = WORLD * rows
        tag = f"rank {rank} t {t}"
        seed = t * 100
        # 1. the first layer's input
        xs = [rand((rows, H), seed + p, dev) for p in range(WORLD)]
        codes, scales = ph.input_gather(xs[rank])
        want = [mxfp8_e4m3_quantize_flydsl(x) for x in xs]
        torch.cuda.synchronize()
        _expect(errors, f"{tag} input codes", codes, torch.cat([w[0] for w in want]))
        _expect(errors, f"{tag} input scales", scales, torch.cat([w[1] for w in want]))
        # 2. the attention output of a MoE layer, then of a dense layer
        attn = [rand((t, Q), seed + 10 + p, dev, 0.5) for p in range(WORLD)]
        res = [rand((rows, H), seed + 20 + p, dev) for p in range(WORLD)]
        moe_in, r, pending = ph.attention_moe(attn[rank], o_w, o_s, res[rank], nw, EPS, mw)
        if full:
            pending.wait()
        local, rs = [], []
        for p in range(WORLD):
            y, rp = gemma_add_rmsnorm(o_proj(*_unpacked(attn, t, rows, p, dev)), res[p], nw, EPS)
            local.append(prepare_mxfp4_moe(y, mw))
            rs.append(rp)
        _expect(errors, f"{tag} moe residual", r, rs[rank])
        if full:
            own = shared_expert_full(local[rank].shared_codes, local[rank].shared_scales, mw.shared_weights)
            torch.cuda.synchronize()
            if not moe_in.routed_only:
                errors.append(f"{tag}: the gathered MoE input is not the routed-only payload")
            for k, field in enumerate(moe_in.tensors()[:4]):
                _expect(errors, f"{tag} moe input field {k}", field, torch.cat([c.tensors()[k] for c in local]))
            _expect(errors, f"{tag} shared expert of the own rows", pending.out, own)
            del own
        else:
            compact = [c.compact() for c in local]
            torch.cuda.synchronize()
            if pending is not None:
                errors.append(f"{tag}: a shard copy returned a pending shared expert")
            for k, field in enumerate(moe_in.tensors()):
                _expect(errors, f"{tag} compact moe input field {k}", field, torch.cat([c.tensors()[k] for c in compact]))
            del compact
        del moe_in, pending, local
        codes, scales, r = ph.attention_dense(attn[rank], o_w, o_s, res[rank], nw, EPS)
        want = [gemma_add_rmsnorm_mxfp8_row_major(o_proj(*_unpacked(attn, t, rows, p, dev)), res[p], nw, EPS)
                for p in range(WORLD)]  # fmt: skip
        torch.cuda.synchronize()
        _expect(errors, f"{tag} dense codes", codes, torch.cat([w[0] for w in want]))
        _expect(errors, f"{tag} dense scales", scales, torch.cat([w[1] for w in want]))
        _expect(errors, f"{tag} dense residual", r, want[rank][2])
        # 3. the FFN output: MoE partials, then a dense partial; the last layer's reduce-scatter; the gather
        slots = [rand((padded, 4, H), seed + 30 + p, dev, 0.25) for p in range(WORLD)]
        shared = [rand((padded, H), seed + 40 + p, dev, 0.25) for p in range(WORLD)]
        codes, scales, r = ph.ffn_norm_gather((slots[rank], shared[rank]), res[rank], nw, EPS)
        sums = [slots4_reduce_scatter_reference(slots, shared, p) for p in range(WORLD)]
        want = [add_rmsnorm_reference(s, res[p], nw, EPS) for p, s in enumerate(sums)]
        torch.cuda.synchronize()
        for p, (y64, _) in enumerate(want):
            blk = slice(p * rows, (p + 1) * rows)
            _expect_mxfp8(errors, f"{tag} slots codes / scales of rank {p}", codes[blk], scales[blk], y64)
        _expect(errors, f"{tag} slots residual", r, want[rank][1])
        h = ph.ffn_reduce_scatter((slots[rank], shared[rank]))
        torch.cuda.synchronize()
        _expect(errors, f"{tag} slots reduce-scatter", h, sums[rank])
        # the owner form: each rank's shared term of its own rows only
        owned = [rand((rows, H), seed + 60 + p, dev, 0.25) for p in range(WORLD)]
        codes, scales, r = ph.ffn_norm_gather((slots[rank], owned[rank]), res[rank], nw, EPS)
        sums = [slots4_reduce_scatter_reference(slots, owned, p) for p in range(WORLD)]
        want = [add_rmsnorm_reference(s, res[p], nw, EPS) for p, s in enumerate(sums)]
        torch.cuda.synchronize()
        for p, (y64, _) in enumerate(want):
            blk = slice(p * rows, (p + 1) * rows)
            _expect_mxfp8(errors, f"{tag} owner slots codes / scales of rank {p}", codes[blk], scales[blk], y64)
        _expect(errors, f"{tag} owner slots residual", r, want[rank][1])
        h = ph.ffn_reduce_scatter((slots[rank], owned[rank]))
        torch.cuda.synchronize()
        _expect(errors, f"{tag} owner slots reduce-scatter", h, sums[rank])
        del slots, shared, owned
        dense = [rand((padded, H), seed + 50 + p, dev, 0.25) for p in range(WORLD)]
        codes, scales, r = ph.ffn_norm_gather(dense[rank], res[rank], nw, EPS)
        sums = [reduce_scatter_reference(dense, p) for p in range(WORLD)]
        want = [gemma_add_rmsnorm_mxfp8_row_major(s, res[p], nw, EPS) for p, s in enumerate(sums)]
        torch.cuda.synchronize()
        _expect(errors, f"{tag} dense ffn codes", codes, torch.cat([w[0] for w in want]))
        _expect(errors, f"{tag} dense ffn scales", scales, torch.cat([w[1] for w in want]))
        _expect(errors, f"{tag} dense ffn residual", r, want[rank][2])
        h = ph.ffn_reduce_scatter(dense[rank])
        out = ph.gather(res[rank])
        torch.cuda.synchronize()
        _expect(errors, f"{tag} dense reduce-scatter", h, sums[rank])
        _expect(errors, f"{tag} gather", out, torch.cat(res))
        del dense
        torch.cuda.empty_cache()


def _chain_worker(rank, errors, token_counts, sdma=False, delay=0, skip_join=False):
    """A MoE layer through the hand-offs (attention_moe, the routed experts on the gathered payload, the owner-form
    tail and the final reduce-scatter) against the one-GPU composition: every rank's preparation, the routed partials
    of the concatenated payload, each owner's ``shared_expert_full``, the reference sum. ``delay``: cycles the side
    stream sleeps before the shared expert; ``skip_join``: ``SharedPending.wait`` does nothing (negative control)."""
    from unittest import mock

    from amd_kernels.collectives.pull_collectives_ranks2.reference import slots4_reduce_scatter_reference
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import (SdmaSeqParallelHandoffs,
                                                                               SeqParallelHandoffs, handoffs,
                                                                               shard_rows)
    from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_gemm
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import MoeInput, mxfp4_moe, prepare_mxfp4_moe, shared_expert_full
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.host import gemma_add_rmsnorm
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import add_rmsnorm_reference

    dev = torch.device("cuda", rank)
    cls = SdmaSeqParallelHandoffs if sdma else SeqParallelHandoffs
    ph = cls(rank, dist.group.WORLD, dev, max(token_counts))
    o_w, o_s = o_proj_weight(dev)
    mw = moe_weights(dev, rank)
    nw = rand((H,), 3, dev, 0.2)
    fork = handoffs._fork_shared

    def delayed_fork(side, *a):
        with torch.cuda.stream(side):
            torch.cuda._sleep(delay)
        pending = fork(side, *a)
        if skip_join:  # the output's block may still hold an earlier call's identical bits: poison it first
            pending.out.fill_(float("nan"))
        return pending

    patches = [mock.patch.object(handoffs, "_fork_shared", delayed_fork)] if delay else []
    if sdma:
        from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import handoffs_sdma

        patches = [mock.patch.object(handoffs_sdma, "_fork_shared", delayed_fork)] if delay else []
    if skip_join:
        patches.append(mock.patch.object(handoffs.SharedPending, "wait", lambda self: None))
    for t in token_counts:
        rows = shard_rows(t)
        tag = f"rank {rank} t {t} delay {delay} skip_join {skip_join}"
        seed = t * 100 + 7
        attn = [rand((t, Q), seed + 10 + p, dev, 0.5) for p in range(WORLD)]
        res = [rand((rows, H), seed + 20 + p, dev) for p in range(WORLD)]
        with contextlib.ExitStack() as stack:
            for p_ in patches:
                stack.enter_context(p_)
            moe_in, r, pending = ph.attention_moe(attn[rank], o_w, o_s, res[rank], nw, EPS, mw)
            slots, none = mxfp4_moe(moe_in, mw, return_partials=True)
            codes, scales, r2 = ph.ffn_norm_gather((slots, pending), r, nw, EPS)
            moe_in, r3, pending = ph.attention_moe(attn[rank], o_w, o_s, res[rank], nw, EPS, mw)
            slots, _ = mxfp4_moe(moe_in, mw, return_partials=True)
            h = ph.ffn_reduce_scatter((slots, pending))
        local, rs = [], []
        for p in range(WORLD):
            xq, xs = _unpacked(attn, t, rows, p, dev)
            y, rp = gemma_add_rmsnorm(mxfp8_gemm(xq, xs, o_w, o_s, torch.bfloat16, activation_scale_preshuffled=True),
                                      res[p], nw, EPS)
            local.append(prepare_mxfp4_moe(y, mw))
            rs.append(rp)
        gathered = MoeInput(*[torch.cat([c.tensors()[k] for c in local]) for k in range(4)])
        ref_slots, _ = mxfp4_moe(gathered, mw, return_partials=True)
        owned = [shared_expert_full(c.shared_codes, c.shared_scales, mw.shared_weights) for c in local]
        sums = [slots4_reduce_scatter_reference([ref_slots, ref_slots], owned, p) for p in range(WORLD)]
        y64, res_ref = add_rmsnorm_reference(sums[rank], rs[rank], nw, EPS)
        torch.cuda.synchronize()
        if none is not None:
            errors.append(f"{tag}: the routed-only partials returned a shared output")
        _expect(errors, f"{tag} routed slots", slots, ref_slots)
        _expect(errors, f"{tag} residual after the MoE tail", r2, res_ref)
        _expect_mxfp8(errors, f"{tag} next codes / scales", codes[rank * rows:(rank + 1) * rows],
                      scales[rank * rows:(rank + 1) * rows], y64)
        _expect(errors, f"{tag} final reduce-scatter", h, sums[rank])
        _expect(errors, f"{tag} second residual", r3, rs[rank])
        del moe_in, pending, slots, local, gathered, ref_slots, owned
        torch.cuda.empty_cache()
    if sdma:
        ph.drain()


def _negative_worker(rank, errors, token_counts, sdma=False):
    """The chain with the side stream delayed and the join skipped: it must report a mismatch."""
    found = []
    _chain_worker(rank, found, token_counts, sdma, 200_000_000, True)
    if not found:
        errors.append(f"rank {rank}: the skipped join went unnoticed")


@pytest.mark.parametrize("sdma", [False, True])
def test_moe_chain_delayed_side_stream(sdma):
    run_on_2_gpus(_chain_worker, [2048, 4097, 15745], sdma, 2_000_000, timeout=1500)


@pytest.mark.parametrize("sdma", [False, True])
def test_moe_chain_skipped_join_fails(sdma):
    run_on_2_gpus(_negative_worker, [4097], sdma, timeout=900)


@pytest.mark.parametrize("tokens", [SERVED_TOKENS])
def test_handoffs_bitwise(tokens):
    run_on_2_gpus(_worker, tokens, timeout=1500)


@pytest.mark.parametrize("tokens", [SERVED_TOKENS])
def test_handoffs_sdma_bitwise(tokens):
    run_on_2_gpus(_worker, tokens, True, timeout=1500)


@pytest.mark.parametrize("sdma", [False, True])
def test_handoffs_shard_copy_compact(sdma):
    """A shard copy of the shared expert keeps the compact payload (no pending shared expert)."""
    run_on_2_gpus(_worker, (2048, 8885), sdma, False, timeout=1500)


def _pool_worker(rank, errors, t, layers, sdma=False, recorded=False):
    """``layers`` MoE layers enqueued behind a long sleep (the host far ahead of the GPU): the pool may grow by about
    one layer's working set, not by a layer's side-stream buffers per layer. ``recorded``: the side stream's buffers
    are recorded on it instead of held by the ``SharedPending`` (negative control)."""
    from unittest import mock

    import amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 as P
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import (SdmaSeqParallelHandoffs,
                                                                               SeqParallelHandoffs, shard_rows)

    dev = torch.device("cuda", rank)
    cls = SdmaSeqParallelHandoffs if sdma else SeqParallelHandoffs
    ph = cls(rank, dist.group.WORLD, dev, t)
    o_w, o_s = o_proj_weight(dev)
    mw = moe_weights(dev, rank)
    nw = rand((H,), 3, dev, 0.2)
    x = rand((shard_rows(t), H), t + rank, dev)
    attn = rand((t, Q), t + 10 + rank, dev, 0.5)
    full = P.shared_expert_full
    patch = mock.patch.object(P, "shared_expert_full", lambda *a, keep=None, **k: full(*a, **k))

    def run(n):
        r = x
        for _ in range(n):
            moe_in, r, shared = ph.attention_moe(attn, o_w, o_s, r, nw, EPS, mw)
            slots, _ = P.mxfp4_moe(moe_in, mw, return_partials=True)
            del moe_in
            _, _, r = ph.ffn_norm_gather((slots, shared), r, nw, EPS)
            del slots, shared

    with patch if recorded else contextlib.nullcontext():
        run(1)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        r0, a0 = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
        torch.cuda._sleep(4_000_000_000)
        run(layers)
        torch.cuda.synchronize()
    grown, used = torch.cuda.max_memory_reserved() - r0, torch.cuda.max_memory_allocated() - a0
    if grown > used + 2**29:
        errors.append(f"rank {rank}: the pool grew by {grown / 2**30:.2f} GiB, {used / 2**30:.2f} GiB allocated")
    if sdma:
        ph.drain()


def _pool_negative_worker(rank, errors, t, layers, sdma):
    found = []
    _pool_worker(rank, found, t, layers, sdma, True)
    if not found:
        errors.append(f"rank {rank}: recorded side-stream buffers did not grow the pool")


@pytest.mark.parametrize("sdma", [False, True])
def test_side_stream_pool_bounded(sdma):
    run_on_2_gpus(_pool_worker, 16384, 24, sdma, timeout=900)


@pytest.mark.parametrize("sdma", [False, True])
def test_side_stream_pool_recorded_grows(sdma):
    run_on_2_gpus(_pool_negative_worker, 16384, 24, sdma, timeout=900)
