# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The one-launch Markov draft chain: routes a call by (vocab, Markov rank, draft steps) of ``w1`` and ``idx_map`` to
the shape package (``_ROUTES``)."""

from amd_kernels.routes import Routes, Rows, arity, check, first, previous_form

_ROUTES = Routes("markov_chain", {(129280, 256, 5): "amd_kernels.sampling.markov_chain_gumbel_bf16_v129280_r256_s5"},
                 ("vocab", "rank", "steps"))
# requests at which the one launch beat the sequential draft steps on MI355X (vLLM DeepSeek-V4.1 DEP8 routes)
_REQUESTS = Rows((1, 160))


def markov_chain_shapes() -> frozenset:
    """(vocab, Markov rank, draft steps) keys."""
    return _ROUTES.shapes()


def _key(w1, idx_map):
    return (w1.shape[0], w1.shape[1], idx_map.shape[1])


def _supported(w1, idx_map) -> str | None:
    if w1.dim() != 2 or idx_map.dim() != 2:
        return "markov_chain: w1 must be [vocab, rank], idx_map [rows, steps]"
    return _ROUTES.blocker(_key(w1, idx_map))


@previous_form(_supported, arity(2))
def markov_chain_blocker(base, w1, w2, input_ids, anchor_idx, idx_map, *args) -> str | None:
    """None when a package serves the Markov draft chain over ``w1`` [vocab, rank] for ``idx_map`` [requests, steps]
    and beats the sequential steps at that many requests."""
    return first(_supported(w1, idx_map), _REQUESTS.blocker("markov_chain", idx_map.shape[0]))


def markov_chain(base, w1, w2, input_ids, anchor_idx, idx_map, *args):
    """Every draft step of every request in one launch; arguments and outputs as the shape package's
    ``markov_chain``."""
    check(_supported(w1, idx_map))
    return _ROUTES.package(_key(w1, idx_map)).markov_chain(base, w1, w2, input_ids, anchor_idx, idx_map, *args)


def warmup_markov_chain(device, vocab: int, rank: int, steps: int, idx_dtype=None, ids_dtype=None) -> None:
    import torch

    _ROUTES.package((vocab, rank, steps)).warmup_markov_chain(device, idx_dtype or torch.int64,
                                                               ids_dtype or torch.int32)
