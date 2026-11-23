# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The hand-offs of a sequence-parallel decoder layer on a TP group (the attention output's pack + all-to-all +
o_proj + post-attention norm + gather, the FFN output's reduce-scatter + next norm + gather), on pull collectives or
(``sdma``) SDMA pushes: routes by (hidden, attention output columns per rank, world), the layer's sizes and the
group's, to the shape package (``_ROUTES``, imported on first use). The ops are the communicator's methods
(``input_gather``, ``attention_moe``, ``attention_dense``, ``ffn_norm_gather``, ``ffn_reduce_scatter``, ``gather``,
``supports``, ``warm_up``)."""

import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare

_ROUTES = Routes("seq_parallel_handoffs", {
    (6144, 2048, 4): "amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4",
    (6144, 4096, 2): "amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2",
}, ("hidden", "q_per_rank", "world"))
_OP = _ROUTES.op


def seq_parallel_handoffs_shapes() -> frozenset[tuple[int, int, int]]:
    """The (hidden, attention output columns per rank, world) with hand-offs."""
    return _ROUTES.shapes()


def seq_parallel_handoffs_layer_blocker(world_size: int, hidden: int, q_per_rank: int) -> str | None:
    """None when a layer of these sizes on ``world_size`` ranks has hand-offs (a load-time check)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    return _ROUTES.blocker((hidden, q_per_rank, world_size))


def seq_parallel_min_tokens(world_size: int, hidden: int, q_per_rank: int) -> int:
    """The token count from which the sequence-parallel layer pays (below it, the caller's own path)."""
    return _ROUTES.package((hidden, q_per_rank, world_size)).MIN_TOKENS


def _key(cpu_group, hidden, q_per_rank):
    return (hidden, q_per_rank, dist.get_world_size(cpu_group))


def seq_parallel_handoffs(rank: int, cpu_group, device, *, hidden: int, q_per_rank: int,
                          max_tokens: int | None = None, sdma: bool = False):
    """The routed package's ``SeqParallelHandoffs`` (``sdma``: ``SdmaSeqParallelHandoffs``) for up to ``max_tokens``
    tokens per call (default the package's bound); collective on the CPU group. Call its ``warm_up`` before serving."""
    pkg = _ROUTES.package(_key(cpu_group, hidden, q_per_rank))
    cls = pkg.SdmaSeqParallelHandoffs if sdma else pkg.SeqParallelHandoffs
    return cls(rank, cpu_group, device, pkg.MAX_TOKENS if max_tokens is None else max_tokens)


def seq_parallel_handoffs_blocker(rank: int, cpu_group, device, *, hidden: int, q_per_rank: int,
                                  max_tokens: int | None = None, sdma: bool = False) -> str | None:
    """None when ``seq_parallel_handoffs`` with these arguments builds the communicator, else why not."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(cpu_group, hidden, q_per_rank)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    pkg = _ROUTES.package(key)
    if max_tokens is not None and not 0 < max_tokens <= pkg.MAX_TOKENS:
        return f"{_OP}: max_tokens {max_tokens} (1..{pkg.MAX_TOKENS})"
    if dist.get_rank(cpu_group) != rank:
        return f"{_OP}: rank {rank} is not this process's rank {dist.get_rank(cpu_group)} in the group"
    return None


def prepare_seq_parallel_handoffs(rank: int, cpu_group, device, *, hidden: int, q_per_rank: int,
                                  max_tokens: int | None = None, sdma: bool = False):
    """``seq_parallel_handoffs(...)``, or the reason it cannot be built."""
    kw = dict(hidden=hidden, q_per_rank=q_per_rank, max_tokens=max_tokens, sdma=sdma)
    return prepare(seq_parallel_handoffs_blocker(rank, cpu_group, device, **kw),
                   lambda: seq_parallel_handoffs(rank, cpu_group, device, **kw))


def seq_parallel_shard(handoffs, x, rank: int):
    """Rank ``rank``'s rows of every token's ``x`` [t, ...] in ``handoffs``' sharding, zero padded past t."""
    key = _ROUTES.owner(type(handoffs))
    if key is None:
        raise ValueError(f"{_OP}: {type(handoffs).__qualname__} not built by seq_parallel_handoffs")
    return _ROUTES.package(key).shard(x, rank)
