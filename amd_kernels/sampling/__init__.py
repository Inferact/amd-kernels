# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sampling-side kernels: logits checks, top-k / top-p, Gumbel draws, block verification.

Op entries (``amd_kernels.routes``): the one-launch Markov draft chain and standard rejection sampling."""

from amd_kernels.routes import Routes, check, first

_MARKOV = Routes("markov_chain", {(129280, 256, 5): "amd_kernels.sampling.markov_chain_gumbel_bf16_v129280_r256_s5"})
_REJECTION = Routes("rejection_sample", {(129280, 5): "amd_kernels.sampling.rejection_standard_v129280_s5"})


def markov_chain_shapes() -> frozenset:
    """(vocab, Markov rank, draft steps) keys."""
    return _MARKOV.shapes()


def markov_chain_blocker(w1, idx_map) -> str | None:
    """None when a package serves the Markov draft chain over ``w1`` [vocab, rank] for ``idx_map`` [rows, steps]."""
    if w1.dim() != 2 or idx_map.dim() != 2:
        return "markov_chain: w1 must be [vocab, rank], idx_map [rows, steps]"
    return _MARKOV.blocker((w1.shape[0], w1.shape[1], idx_map.shape[1]))


def markov_chain(base, w1, w2, input_ids, anchor_idx, idx_map, *args):
    """Every draft step of every request in one launch; arguments and outputs as the shape package's
    ``markov_chain``."""
    check(markov_chain_blocker(w1, idx_map))
    return _MARKOV.package((w1.shape[0], w1.shape[1], idx_map.shape[1])).markov_chain(base, w1, w2, input_ids,
                                                                                       anchor_idx, idx_map, *args)


def warmup_markov_chain(device, vocab: int, rank: int, steps: int, idx_dtype=None, ids_dtype=None) -> None:
    import torch

    _MARKOV.package((vocab, rank, steps)).warmup_markov_chain(
        device, idx_dtype or torch.int64, ids_dtype or torch.int32
    )


def rejection_sample_shapes() -> frozenset:
    """(vocab, draft tokens) keys."""
    return _REJECTION.shapes()


def rejection_sample_blocker(draft_logits, requests: int) -> str | None:
    """None when a package serves standard rejection sampling of ``draft_logits`` [requests, steps, vocab] for
    ``requests`` requests."""
    if draft_logits.dim() != 3:
        return "rejection_sample: draft_logits must be [requests, steps, vocab]"
    key = (draft_logits.shape[2], draft_logits.shape[1])
    reason = _REJECTION.blocker(key)
    if reason is not None:
        return reason
    limit = _REJECTION.package(key).kernel.MAX_REQS
    return first(None if 1 <= requests <= limit else f"rejection_sample: {requests} requests outside 1..{limit}")


def rejection_sample(target_logits, draft_logits, draft_sampled, pos, idx_mapping, temperature, seed,
                     synthetic_rates=None):  # fmt: skip
    """``(sampled [requests, steps + 1], num_sampled [requests])``; as the shape package's ``rejection_sample``."""
    check(rejection_sample_blocker(draft_logits, idx_mapping.shape[0]))
    return _REJECTION.package((draft_logits.shape[2], draft_logits.shape[1])).rejection_sample(
        target_logits, draft_logits, draft_sampled, pos, idx_mapping, temperature, seed, synthetic_rates
    )


def warmup_rejection_sample(device, vocab: int, steps: int) -> None:
    _REJECTION.package((vocab, steps)).warmup_rejection_sample(device)
