# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Batch-sharded sampling on a TP group (the target's verify-logits exchange, the sampler-output gather, the Gumbel
Markov draft chain): routes the communicator by (vocab, world), the model's vocabulary and the group's size, to the
shape package (``_ROUTES``, imported on first use). The ops are its methods (``target_logits``,
``gather_sampler_output``, ``setup_markov``, ``markov_draft``, ``close``)."""

import inspect

import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare

_ROUTES = Routes("batch_sharded_sampling", {
    (200064, 4): "amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4",
    (200064, 2): "amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks2",
}, ("vocab", "world"))
_OP = _ROUTES.op


def batch_sharded_sampling_shapes() -> frozenset[tuple[int, int]]:
    """The (vocab, world) with communicators."""
    return _ROUTES.shapes()


def batch_sharded_sampling_config_blocker(vocab: int, world: int, max_reqs: int, logits_per_req: int) -> str | None:
    """None when the communicators serve this batch shape (``world`` ranks, up to ``max_reqs`` requests of up to
    ``logits_per_req`` logits rows): a load-time check before the group exists."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = (vocab, world)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    reason = _ROUTES.package(key).sampling_blocker(world, max_reqs, logits_per_req)
    return None if reason is None else f"{_OP}: {reason}"


def _takes_lm_weight(pkg) -> bool:
    return "lm_weight" in inspect.signature(pkg.BatchShardedSampling.__init__).parameters


def batch_sharded_sampling(rank: int, group, device, *, vocab: int, max_reqs: int, target_rows: int, lm_weight=None):
    """The routed package's ``BatchShardedSampling`` (collective on the group, warmed up); ``lm_weight``: this rank's
    shuffled lm_head shard, for a package that keeps the peer's copy."""
    pkg = _ROUTES.package((vocab, dist.get_world_size(group)))
    kw = {} if lm_weight is None else {"lm_weight": lm_weight}
    return pkg.BatchShardedSampling(rank, group, device, max_reqs=max_reqs, target_rows=target_rows, **kw)


def batch_sharded_sampling_blocker(rank: int, group, device, *, vocab: int, max_reqs: int, target_rows: int,
                                   lm_weight=None) -> str | None:
    """None when ``batch_sharded_sampling`` with these arguments builds the communicators, else why not."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    world = dist.get_world_size(group)
    key = (vocab, world)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    pkg = _ROUTES.package(key)
    if not (world <= max_reqs <= pkg.MAX_REQS and 1 <= target_rows <= pkg.MAX_ROWS):
        return f"{_OP}: {max_reqs} requests, {target_rows} target rows ({world}..{pkg.MAX_REQS}, 1..{pkg.MAX_ROWS})"
    if lm_weight is not None and not _takes_lm_weight(pkg):
        return f"{_OP}: lm_weight (the {world}-rank communicators keep no peer shard)"
    if dist.get_rank(group) != rank:
        return f"{_OP}: rank {rank} is not this process's rank {dist.get_rank(group)} in the group"
    return None


def prepare_batch_sharded_sampling(rank: int, group, device, *, vocab: int, max_reqs: int, target_rows: int,
                                   lm_weight=None):
    """``batch_sharded_sampling(...)``, or the reason it cannot be built."""
    kw = dict(vocab=vocab, max_reqs=max_reqs, target_rows=target_rows, lm_weight=lm_weight)
    return prepare(batch_sharded_sampling_blocker(rank, group, device, **kw),
                   lambda: batch_sharded_sampling(rank, group, device, **kw))


def batch_sharded_markov_blocker(sampling, steps: int, max_reqs: int, markov_w1, markov_w2, *, cache=None,
                                 lm_weight_shuffled: bool = True, fp64_noise: bool = False, confidences: bool = False,
                                 logit_transform: bool = False, reduced_vocab: bool = False) -> str | None:
    """None when ``sampling.setup_markov`` serves this draft (its steps, requests, Markov head, draft-logits cache and
    features), else why not."""
    op = "batch_sharded_markov"
    key = _ROUTES.owner(type(sampling))
    if key is None:
        return f"{op}: {type(sampling).__qualname__} not built by batch_sharded_sampling"
    reason = _ROUTES.package(key).markov_blocker(
        steps, max_reqs, markov_w1, markov_w2, cache=cache, lm_weight_shuffled=lm_weight_shuffled,
        fp64_noise=fp64_noise, confidences=confidences, logit_transform=logit_transform, reduced_vocab=reduced_vocab)
    return None if reason is None else f"{op}: {reason}"
