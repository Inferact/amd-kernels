# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Batch-sharded sampling on two ranks: each rank verifies, and drafts for, the requests it owns (batch position i
on rank i % 2).

One object per process (``BatchShardedSampling``, built collectively, before any CUDA graph capture) owns the
communicators, the owner tables of every request count, and (``lm_weight`` given) a copy of the peer's lm_head shard.
The communicators compile and run their warm-up collectively when they are built; the two GEMM chains are entries
(``amd_kernels.warm_up``): the first call of each mode compiles every variant the mode can reach. Their warm-up calls
never touch a communicator or a caller's buffer: a variant compiled earlier in the process launches even in FlyDSL's
compile-only mode, so they publish into a private slot and skip the exchanges.

Logits of the owned rows, two ways with the same bits (the GEMM's sums do not depend on its row tiling):
- exchange (up to ``LOCAL_MIN_ROWS`` rows, or without the peer copy): the lm_head GEMM over all owner-sorted rows
  publishes into the exchange's send slot, the pull writes this rank's rows in vocabulary order. One xGMI link
  carries (M / 2) x 100032 x 2 B per direction, 4x the per-link bytes of four ranks.
- local (above): the GEMM over the owned rows only, once on this rank's shard and once on the peer's copy, straight
  into the two vocabulary halves of the output; no exchange.

- ``target_logits``: the target's verify logits of this rank's rows.
- ``gather_sampler_output``: tokens / counts / NaN counts back in batch order (``SamplerAllGather``).
- ``markov_draft``: the Markov draft of every owned request: the draft lm_head, then per step the bias projection,
  the Gumbel partials (+ draft-cache row) and the combine (+ draft id, next packed ``markov_w1`` row; two launches
  above 64 owned rows), then the id gather into batch order (``I64C7AllGather``).

The arithmetic is the per-kernel entries' (the tests compare bits with them); the chains only sequence them.
"""

import ctypes
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from amd_kernels.common.hip_runtime import load_hip_runtime
from amd_kernels.warm_up import _in_warm_up, entry

V, H, RANK = 200064, 6144, 256  # vocabulary, hidden size (the lm_head's K), Markov rank
N = V // 2  # the lm_head's vocabulary shard
WORLD = 2
MAX_ROWS = 2048  # the exchanges' and the lm_head GEMM's row bound
MAX_REQS = 256  # the small gathers' request bound
MAX_OWNED = 128  # requests per rank of the Markov bias GEMM (bf16_gemm_packed_rows)
COMBINE_ROWS = 64  # rows per launch of gumbel_combine_markov_embed_packed
MAX_STEPS = 7  # draft steps (I64C7AllGather's columns)
MAX_LOGITS_PER_REQ = 8  # SamplerAllGather's token columns
LOCAL_MIN_ROWS = 96  # above (total rows): the owned rows on both shards locally; crossover of benchmark_block.py


def sampling_blocker(world: int, max_reqs: int, logits_per_req: int) -> str | None:
    """Why ``BatchShardedSampling`` cannot serve this batch shape (None: it can): ``world`` ranks, up to ``max_reqs``
    requests of up to ``logits_per_req`` logits rows each."""
    if world != WORLD:
        return f"{world} ranks (two-rank kernels)"
    if not WORLD <= max_reqs <= MAX_REQS:
        return f"{max_reqs} requests outside [{WORLD}, {MAX_REQS}]"
    if not 1 <= logits_per_req <= MAX_LOGITS_PER_REQ:
        return f"{logits_per_req} logits rows per request (the gather has {MAX_LOGITS_PER_REQ} columns)"
    if max_reqs * logits_per_req > MAX_ROWS:
        return f"more than {MAX_ROWS} logits rows per step"
    return None


def markov_blocker(steps: int, max_reqs: int, markov_w1: torch.Tensor, markov_w2: torch.Tensor, *,
                   cache: torch.Tensor | None = None, lm_weight_shuffled: bool = True, fp64_noise: bool = False,
                   confidences: bool = False, logit_transform: bool = False,
                   reduced_vocab: bool = False) -> str | None:
    """Why ``setup_markov`` cannot serve this draft (None: it can). The keywords describe the caller's draft: its
    draft-logits cache, whether its lm_head weight holds ``shuffle_weight``'s layout, and the features the chain does
    not have (fp64 Gumbel noise, draft confidences, an output logit transform, a reduced draft vocabulary)."""
    if not 1 <= steps <= MAX_STEPS:
        return f"{steps} draft steps (the id gather has {MAX_STEPS} columns)"
    if -(-max_reqs // WORLD) > MAX_OWNED:
        return f"more than {MAX_OWNED} requests per rank"
    for w in (markov_w1, markov_w2):
        if tuple(w.shape) != (V, RANK) or w.dtype != torch.bfloat16 or not w.is_contiguous():
            return f"the Markov head is not bf16 [{V}, {RANK}]"
    if cache is not None and not (cache.dtype == torch.bfloat16 and cache.dim() == 3 and cache.shape[1:] == (steps, V)
                                  and cache.is_contiguous()):
        return f"the draft-logits cache {tuple(cache.shape)} {cache.dtype} (contiguous bf16 [slots, {steps}, {V}])"
    for flag, why in ((not lm_weight_shuffled, "the lm_head weight is not in shuffle_weight's layout"),
                      (fp64_noise, "fp64 Gumbel noise (the chain draws fp32 noise)"),
                      (confidences, "draft confidences (the chain writes ids and draft logits only)"),
                      (logit_transform, "an output logit transform"),
                      (reduced_vocab, f"a reduced draft vocabulary (the chain samples all {V} ids)")):
        if flag:
            return why
    return None


@dataclass
class OwnerTables:
    """Device index tables of one request count (``owner_tables``)."""

    cap: int  # requests per rank, cdiv(num_reqs, 2)
    send_rows: torch.Tensor  # int64 [2 * cap * steps]: owner-sorted (request, step) rows, padding -> request 0
    own_rows: torch.Tensor  # int64 [cap * steps]: this rank's part of send_rows
    slot_rows: torch.Tensor  # int64 [steps, cap]: this rank's (request, step) rows
    valid: torch.Tensor  # bool [1, cap]: the local request exists
    anchor_rows: torch.Tensor  # int64 [cap]: the anchor token's row in input_ids
    gathered_src: torch.Tensor  # int64 [num_reqs]: batch request -> (owner, local row) in the id gather
    counts: torch.Tensor  # int32 [2]: rows per rank in the exchange


def owner_tables(num_reqs: int, steps: int, rank: int, anchor_idx: np.ndarray, device) -> OwnerTables:
    """Batch position i is owned by rank i % 2 (local row i // 2); a rank's padding rows read request 0."""
    cap = -(-num_reqs // WORLD)
    send = np.arange(cap)[None, :] * WORLD + np.arange(WORLD)[:, None]  # [2, cap] batch positions
    valid = send < num_reqs
    send_c = np.where(valid, send, 0)
    step = np.arange(steps)
    send_rows = (send_c[..., None] * steps + step).reshape(-1)
    local = send_c[rank]
    req = np.arange(num_reqs)
    own = send_rows[rank * cap * steps:(rank + 1) * cap * steps]
    return OwnerTables(
        cap=cap,
        send_rows=torch.from_numpy(send_rows.astype(np.int64)).to(device),
        own_rows=torch.from_numpy(own.astype(np.int64)).to(device),
        slot_rows=torch.from_numpy((local[None, :] * steps + step[:, None]).astype(np.int64)).to(device),
        valid=torch.from_numpy(valid[rank][None, :].copy()).to(device),
        anchor_rows=torch.from_numpy(anchor_idx[local].astype(np.int64)).to(device),
        gathered_src=torch.from_numpy(((req % WORLD) * cap + req // WORLD).astype(np.int64)).to(device),
        counts=torch.full((WORLD,), cap * steps, dtype=torch.int32, device=device),
    )


_CAPTURED_HOST: list[torch.Tensor] = []


def _device_counts(counts, device: torch.device) -> torch.Tensor:
    """counts[2] (host ints) -> device int32 through a pinned staging tensor (non-blocking; the caching host
    allocator keeps the block until its copy has run). Under graph capture (a timing harness) the staging tensor is
    kept, since every replay reads it."""
    host = torch.empty(WORLD, dtype=torch.int32, pin_memory=True)
    host.numpy()[:] = counts
    if torch.cuda.is_current_stream_capturing():
        _CAPTURED_HOST.append(host)
    return host.to(device, non_blocking=True)


def _publish_slot(comm, rows: int, device):
    """The exchange's send slot, or in a warm-up a private slot pair, counter and flag (no peer, no caller buffer)."""
    if not _in_warm_up.get():
        return comm.slot(), None
    slot_bytes = max(rows, 1) * N * 2
    keep = (torch.empty(2 * slot_bytes, dtype=torch.uint8, device=device),
            torch.zeros(1, dtype=torch.int32, device=device), torch.zeros(16, dtype=torch.int32, device=device))
    return (keep[0].data_ptr(), slot_bytes, keep[1], keep[2].data_ptr()), keep


class _PeerShard:
    """This rank's copy of the peer's lm_head shard (same layout), filled once over IPC: each rank writes its own
    shard into the peer's copy (collective)."""

    def __init__(self, rank, group, device, lm_weight):
        from amd_kernels.collectives.all_to_all_bf16_n100032_ranks2.host import (_DevicePointer, _alloc_pool,
                                                                                  _exchange, _open_pool)

        assert lm_weight.shape == (N, H) and lm_weight.dtype == torch.bfloat16 and lm_weight.is_contiguous()
        nbytes = N * H * 2
        self.ptr, handle, pool = _alloc_pool(nbytes, device)
        self.weight = pool.view(torch.bfloat16).view(N, H)
        self.own = lm_weight.data_ptr()
        handles = _exchange(handle, rank, WORLD, group)
        peer = _open_pool(handles[1 - rank], device)
        dst = torch.as_tensor(_DevicePointer(peer, nbytes), device=device)
        dst.copy_(lm_weight.view(-1).view(torch.uint8))
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        lib = load_hip_runtime()
        lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        assert lib.hipIpcCloseMemHandle(ctypes.c_void_p(peer)) == 0
        dist.barrier(group=group)

    def close(self):
        lib = load_hip_runtime()
        lib.hipFree.argtypes = [ctypes.c_void_p]
        torch.cuda.synchronize()
        assert lib.hipFree(ctypes.c_void_p(self.ptr)) == 0
        self.weight = None


@dataclass
class _Markov:
    steps: int
    max_reqs: int
    tables: dict
    arange: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor


class BatchShardedSampling:
    """The communicators and tables of batch-sharded sampling on two ranks (module doc). Build it collectively
    (every rank, the same order), before any CUDA graph capture; ``setup_markov`` adds the draft's exchange.
    ``lm_weight``: this rank's lm_head shard in ``shuffle_weight``'s layout, the tensor every call passes; given, the
    rank keeps a copy of the peer's shard (1.23 GB) and computes logits locally above ``LOCAL_MIN_ROWS`` rows."""

    def __init__(self, rank: int, group, device, *, max_reqs: int, target_rows: int,
                 lm_weight: torch.Tensor | None = None):
        from amd_kernels.collectives.all_gather_sampler_i64_c8_ranks2 import SamplerAllGather
        from amd_kernels.collectives.all_to_all_bf16_n100032_ranks2 import RowAllToAll

        assert WORLD <= max_reqs <= MAX_REQS and 1 <= target_rows <= MAX_ROWS
        self.rank, self.group, self.device = rank, group, torch.device(device)
        self.max_reqs, self.target_rows = max_reqs, target_rows
        self.peer = None if lm_weight is None else _PeerShard(rank, group, self.device, lm_weight)
        self.target_a2a = RowAllToAll(rank, group, self.device, max_rows=self._exchange_rows(target_rows))
        self.target_gather = SamplerAllGather(rank, group, self.device)
        self.target_a2a.warmup()  # collective: every rank, here
        self.target_gather.warmup()
        # the lm_head's publish (and local: rows) tiles at load (the entry's first call then finds them compiled)
        from amd_kernels.gemm.bf16_n100032_k6144.host import warmup_bf16_gemm_n100032_k6144

        warmup_bf16_gemm_n100032_k6144(self.device, shuffled=True, mode="publish")
        if self.peer is not None:
            warmup_bf16_gemm_n100032_k6144(self.device, shuffled=True, mode="rows")
        torch.cuda.empty_cache()
        self.markov: _Markov | None = None
        self.draft_a2a = self.draft_gather = None

    def _exchange_rows(self, rows: int) -> int:
        return rows if self.peer is None else min(rows, LOCAL_MIN_ROWS)

    def _local(self, m: int, lm_weight) -> bool:
        if self.peer is None or m <= LOCAL_MIN_ROWS:
            return False
        assert lm_weight.data_ptr() == self.peer.own, "lm_weight is not the shard given at construction"
        return True

    # ------------------------------------------------------------ the target's verify logits
    def target_logits(self, hidden, lm_weight, sorted_rows, counts, num_local: int) -> torch.Tensor:
        """[num_local, 200064] bf16: the logits of this rank's rows, bit for bit what the lm_head GEMM of every rank
        over ``hidden[sorted_rows]`` + an all-to-all of the vocabulary shards give. ``hidden`` bf16 [T, 6144]
        contiguous (the same on every rank); ``lm_weight`` this rank's [100032, 6144] shard in ``shuffle_weight``'s
        layout; ``sorted_rows`` int64 [M] (the owner-sorted rows, 0 <= M <= target_rows, the same M on every rank);
        ``counts`` host ints [2] (rows per owner; ``num_local == counts[rank]``)."""
        if sorted_rows.shape[0] == 0:  # the same on every rank: no exchange
            return torch.empty((0, V), dtype=torch.bfloat16, device=hidden.device)
        return _target_logits(self, hidden, lm_weight, sorted_rows, list(counts), num_local)

    # ------------------------------------------------------------ the sampler outputs in batch order
    def gather_sampler_output(self, tokens, num_sampled, num_rejected, num_nans, cu, src, src_stride: int,
                              num_reqs: int, width: int, with_nans: bool):
        """(tokens int64 [num_reqs, width], num_sampled int32, num_rejected int32, num_nans int32 or None) of every
        request in batch order, from this rank's ``tokens`` [n, >= width] int64 / counts [n] / ``cu`` [n + 1]
        (``tokens`` None: no local request) and the batch -> (owner, local row) map ``src`` int64 [num_reqs]
        (row = owner * src_stride + local). Any integer dtype for the counts; ``num_nans`` is read with
        ``with_nans`` only. Every rank calls, the same ``num_reqs``, ``width`` and ``with_nans``."""
        return _gather_sampler_output(self, tokens, num_sampled, num_rejected, num_nans, cu, src, src_stride,
                                      num_reqs, width, with_nans)

    # ------------------------------------------------------------ the Markov draft
    def setup_markov(self, steps: int, max_reqs: int, anchor_idx, markov_w1, markov_w2) -> None:
        """The draft's exchange, id gather and owner tables of 1..max_reqs requests (collective, before capture).
        ``anchor_idx`` [max_reqs] (host or device ints): request i's anchor token row in ``input_ids``."""
        from amd_kernels.collectives.all_gather_i64_c7_ranks2 import I64C7AllGather
        from amd_kernels.collectives.all_to_all_bf16_n100032_ranks2 import RowAllToAll

        why = markov_blocker(steps, max_reqs, markov_w1, markov_w2)
        assert why is None, why
        anchor = anchor_idx.cpu().numpy() if isinstance(anchor_idx, torch.Tensor) else np.asarray(anchor_idx)
        cap = -(-max_reqs // WORLD)
        rows = self._exchange_rows(min(MAX_ROWS, WORLD * cap * steps))
        self.draft_a2a = RowAllToAll(self.rank, self.group, self.device, max_rows=rows)
        self.draft_gather = I64C7AllGather(self.rank, self.group, self.device)
        self.draft_a2a.warmup()  # collective: every rank, here
        self.draft_gather.warmup()
        self.markov = _Markov(
            steps=steps,
            max_reqs=max_reqs,
            tables={n: owner_tables(n, steps, self.rank, anchor, self.device) for n in range(1, max_reqs + 1)},
            arange=torch.arange(cap, dtype=torch.int64, device=self.device),
            w1=markov_w1,
            w2=markov_w2,
        )

    def markov_draft(self, num_reqs: int, head_hidden, lm_weight, sample_indices, sample_idx_mapping, sample_pos,
                     input_ids, temperature, seeds, cache, draft_tokens) -> None:
        """The Markov draft of ``num_reqs`` requests (1..max_reqs of ``setup_markov``), every rank the same call:
        ``draft_tokens[:num_reqs, :steps]`` = every request's draft ids in batch order, and this rank's requests'
        draft-cache rows ``cache[slot, step]``.

        ``head_hidden`` bf16 [T, 6144] contiguous (the draft head's output rows, the same on every rank),
        ``lm_weight`` the shuffled lm_head shard, ``sample_indices`` int64 [>= num_reqs * steps] ((request, step) ->
        row of head_hidden), ``sample_idx_mapping`` [>= num_reqs * steps] (-> request slot), ``sample_pos`` int64
        (-> draw position + 1), ``input_ids`` (the anchor tokens at ``anchor_idx``), ``temperature`` fp32 / ``seeds``
        int64 [slots], ``cache`` bf16 [slots, steps, 200064], ``draft_tokens`` int64 [>= num_reqs, steps]. The draws
        are upstream vLLM's ``_sample_sequential_sharded`` (``gumbel_partials_bias`` at ``sample_pos - 1``)."""
        _markov_draft(self, num_reqs, head_hidden, lm_weight, sample_indices, sample_idx_mapping, sample_pos,
                      input_ids, temperature, seeds, cache, draft_tokens)

    def close(self) -> None:
        """Collective teardown after every graph that uses the communicators is retired."""
        for c in (self.target_a2a, self.target_gather, self.draft_a2a, self.draft_gather):
            if c is not None:
                c.close()
        if self.peer is not None:
            self.peer.close()
            self.peer = None


def _local_logits(s: BatchShardedSampling, x, lm_weight, own_rows, out):
    """out [n, 200064] = the logits of x[own_rows]: this rank's shard and the peer copy into their vocabulary halves."""
    from amd_kernels.gemm.bf16_n100032_k6144.host import bf16_gemm_n100032_k6144

    for r, w in ((s.rank, lm_weight), (1 - s.rank, s.peer.weight)):
        bf16_gemm_n100032_k6144(x, w, out[:, r * N:(r + 1) * N], shuffled=True, row_indices=own_rows)


# ---------------------------------------------------------------- the target logits
def _target_logits(s: BatchShardedSampling, hidden, lm_weight, sorted_rows, counts, num_local):
    from amd_kernels.gemm.bf16_n100032_k6144.host import bf16_gemm_n100032_k6144

    m = sorted_rows.shape[0]
    assert 1 <= m <= s.target_rows and len(counts) == WORLD and sum(counts) == m and counts[s.rank] == num_local
    dev = hidden.device
    out = torch.empty((num_local, V), dtype=torch.bfloat16, device=dev)
    if s._local(m, lm_weight):
        if num_local:
            lo = sum(counts[:s.rank])
            _local_logits(s, hidden, lm_weight, sorted_rows[lo:lo + num_local], out)
        return out
    cs = _device_counts(counts, dev)
    slot, keep = _publish_slot(s.target_a2a, m, dev)
    bf16_gemm_n100032_k6144(hidden, lm_weight, shuffled=True, row_indices=sorted_rows, publish=slot)
    if keep is None:  # the pull's grids compiled in the communicator's warm-up
        s.target_a2a.pull(m, cs, out)
    return out


def _target_mode(s, hidden, lm_weight, sorted_rows, counts, num_local):
    return ("target", s.target_rows, s.peer is not None)


def _target_sizes(mode):
    from amd_kernels.gemm.bf16_n100032_k6144.host import WARMUP_ROWS

    rows = mode[1]
    # every lm_head tile variant of each path (the pull's grids compile with the communicator); a local call runs
    # owned-row counts, about half of the total
    sizes = {min(m, rows) for m in WARMUP_ROWS}
    if mode[2]:
        sizes |= {min(WORLD * m, rows) for m in WARMUP_ROWS if WORLD * m > LOCAL_MIN_ROWS}
    return sorted(sizes)


def _target_inputs(mode, n, s, hidden, lm_weight, sorted_rows, counts, num_local):
    x = hidden if hidden.shape[0] >= n else torch.zeros((n, H), dtype=torch.bfloat16, device=hidden.device)
    c = [n // WORLD + int(r < n % WORLD) for r in range(WORLD)]
    rows = torch.arange(n, dtype=torch.int64, device=hidden.device)
    return (s, x, lm_weight, rows, c, c[s.rank]), {}


_target_logits = entry("sampling/batch_sharded_markov_v200064_h6144_r256_ranks2:target", mode=_target_mode,
                       sizes=_target_sizes, inputs=_target_inputs)(_target_logits)


# ---------------------------------------------------------------- the sampler outputs
def _gather_sampler_output(s, tokens, num_sampled, num_rejected, num_nans, cu, src, src_stride, num_reqs, width,
                           with_nans):
    i32, i64, dev = torch.int32, torch.int64, s.device
    assert 1 <= width <= MAX_LOGITS_PER_REQ and num_reqs == src.shape[0] <= MAX_REQS
    if tokens is None:  # no local requests
        tokens = torch.zeros((0, 1), dtype=i64, device=dev)
        num_sampled = num_rejected = torch.zeros(0, dtype=i32, device=dev)
        num_nans = torch.zeros(0, dtype=i32, device=dev) if with_nans else None
        cu = torch.zeros(1, dtype=i32, device=dev)
    else:
        if tokens.stride(-1) != 1 or tokens.shape[1] > MAX_LOGITS_PER_REQ:
            tokens = tokens[:, :MAX_LOGITS_PER_REQ].contiguous()
        num_sampled = num_sampled.to(i32).contiguous()
        num_rejected = num_rejected.to(i32).contiguous()
        num_nans = num_nans.to(i32).contiguous() if with_nans else None
        cu = cu.to(i32).contiguous()
    out_tokens = torch.empty((num_reqs, MAX_LOGITS_PER_REQ), dtype=i64, device=dev)
    out_sampled = torch.empty(num_reqs, dtype=i32, device=dev)
    out_rejected = torch.empty(num_reqs, dtype=i32, device=dev)
    out_nans = torch.empty(num_reqs, dtype=i32, device=dev) if with_nans else None
    s.target_gather(tokens, num_sampled, num_rejected, num_nans, cu, src, src_stride, out_tokens, out_sampled,
                    out_rejected, out_nans)
    sampled = out_tokens if width == MAX_LOGITS_PER_REQ else out_tokens[:, :width].contiguous()
    return sampled, out_sampled, out_rejected, out_nans


# ---------------------------------------------------------------- the Markov draft
def _markov_draft(s, num_reqs, head_hidden, lm_weight, sample_indices, sample_idx_mapping, sample_pos, input_ids,
                  temperature, seeds, cache, draft_tokens):
    from amd_kernels.gemm.bf16_n100032_k6144.host import bf16_gemm_n100032_k6144
    from amd_kernels.gemm.bf16_n200064_k256.host import bf16_gemm_packed_rows, pack_bf16_rows, padded_rows
    from amd_kernels.sampling.gumbel_combine_markov_embed_v200064_r256.host import (
        gumbel_combine_markov_embed_packed,
    )
    from amd_kernels.sampling.gumbel_partials_bias_bf16_v200064.host import gumbel_partials_bias
    from amd_kernels.sampling.gumbel_partials_bias_bf16_v200064.reference import NB

    d = s.markov
    assert d is not None and 1 <= num_reqs <= d.max_reqs
    assert head_hidden.dtype == torch.bfloat16 and head_hidden.dim() == 2 and head_hidden.is_contiguous()
    t = d.tables[num_reqs]
    cap, steps, dev = t.cap, d.steps, head_hidden.device
    bf16 = torch.bfloat16
    # 1. the draft lm_head of this rank's requests' rows: over every owner's rows into the exchange, or locally
    base = torch.empty((cap * steps, V), dtype=bf16, device=dev)
    keep = None
    if s._local(WORLD * cap * steps, lm_weight):
        _local_logits(s, head_hidden, lm_weight, sample_indices[t.own_rows], base)
    else:
        rows = sample_indices[t.send_rows]
        slot, keep = _publish_slot(s.draft_a2a, rows.numel(), dev)
        bf16_gemm_n100032_k6144(head_hidden, lm_weight, shuffled=True, row_indices=rows, publish=slot)
        if keep is None:  # the pull's grids compiled in the communicator's warm-up
            s.draft_a2a.pull(rows.numel(), t.counts, base)
    base = base.view(cap, steps, V)
    # 2. the local rows' request slots (-1: no request), draw positions and anchor tokens
    slots = torch.where(t.valid, sample_idx_mapping[t.slot_rows], -1).to(torch.int32)
    pos = sample_pos[t.slot_rows]
    prev = input_ids[t.anchor_rows].to(torch.int64)
    # 3. the Markov steps: bias GEMM, Gumbel partials (+ cache row), combine (+ id, next packed embedding row)
    rows_ar = d.arange[:cap]
    packed = torch.empty((padded_rows(cap), RANK), dtype=bf16, device=dev)
    pack_bf16_rows(d.w1, prev, packed)
    bias = torch.empty((cap, V), dtype=bf16, device=dev)
    lmax = torch.empty((cap, NB), dtype=torch.float32, device=dev)
    larg = torch.empty((cap, NB), dtype=torch.int64, device=dev)
    tokens = torch.empty((cap, MAX_STEPS), dtype=torch.int64, device=dev)
    prev_out = torch.empty(cap, dtype=torch.int64, device=dev)
    for i in range(steps):
        bf16_gemm_packed_rows(packed, d.w2, cap, bias)
        gumbel_partials_bias(base[:, i], slots[i], temperature, seeds, pos[i], cache, i, bias, rows_ar, lmax, larg,
                             pos_offset=-1)  # upstream draws at sample_pos - 1
        for lo in range(0, cap, COMBINE_ROWS):  # packed rows [lo, lo + 64) are their own 64-row packed block
            hi = min(cap, lo + COMBINE_ROWS)
            gumbel_combine_markov_embed_packed(lmax[lo:hi], larg[lo:hi], rows_ar[lo:hi], d.w1, tokens[lo:hi], i,
                                               prev_out, packed[lo:lo + padded_rows(hi - lo)])
    # 4. every rank gets every request's drafts, in batch order (the gather compiled in the communicator's warm-up)
    if keep is not None:
        return
    out = draft_tokens[:num_reqs]
    if steps == MAX_STEPS:
        s.draft_gather(tokens, t.gathered_src, cap, out)
    else:
        full = torch.empty((num_reqs, MAX_STEPS), dtype=torch.int64, device=dev)
        s.draft_gather(tokens, t.gathered_src, cap, full)
        out.copy_(full[:, :steps])


def _markov_mode(s, num_reqs, *args):
    return ("markov", s.markov.steps, s.markov.max_reqs, s.peer is not None)


def _markov_inputs(mode, n, s, num_reqs, head_hidden, lm_weight, sample_indices, sample_idx_mapping, sample_pos,
                   input_ids, temperature, seeds, cache, draft_tokens):
    # the caller's read-only buffers, no request slot (slot -1: the Gumbel kernel writes no cache row), a private
    # output; a kernel compiled before the warm-up launches, so nothing it writes may be the caller's
    no_slot = torch.full_like(sample_idx_mapping, -1)
    out = torch.empty((n, mode[1]), dtype=torch.int64, device=s.device)
    return (s, n, head_hidden, lm_weight, sample_indices, no_slot, sample_pos, input_ids, temperature, seeds, cache,
            out), {}


_markov_draft = entry("sampling/batch_sharded_markov_v200064_h6144_r256_ranks2:markov", mode=_markov_mode,
                      sizes=lambda mode: range(1, mode[2] + 1), inputs=_markov_inputs)(_markov_draft)
