# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Standard rejection sampling of draft tokens: routes a call by (vocab, draft tokens) of ``draft_logits`` to the
shape package (``_ROUTES``)."""

from amd_kernels.routes import Routes, check, previous_form

_ROUTES = Routes("rejection_sample", {(129280, 5): "amd_kernels.sampling.rejection_standard_v129280_s5"},
                 ("vocab", "steps"))


def rejection_sample_shapes() -> frozenset:
    """(vocab, draft tokens) keys."""
    return _ROUTES.shapes()


def _supported(draft_logits, requests: int) -> str | None:
    if draft_logits.dim() != 3:
        return "rejection_sample: draft_logits must be [requests, steps, vocab]"
    key = (draft_logits.shape[2], draft_logits.shape[1])
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    limit = _ROUTES.package(key).kernel.MAX_REQS
    return None if 1 <= requests <= limit else f"rejection_sample: {requests} requests outside 1..{limit}"


def _requests_form(args, kwargs) -> bool:
    return len(args) == 2 and type(args[1]) is int


@previous_form(_supported, _requests_form)
def rejection_sample_blocker(target_logits, draft_logits, draft_sampled, pos, idx_mapping, temperature, seed,
                             synthetic_rates=None) -> str | None:  # fmt: skip
    """None when a package serves standard rejection sampling of ``draft_logits`` [requests, steps, vocab] for the
    requests of ``idx_mapping``."""
    return _supported(draft_logits, idx_mapping.shape[0])


def rejection_sample(target_logits, draft_logits, draft_sampled, pos, idx_mapping, temperature, seed,
                     synthetic_rates=None):  # fmt: skip
    """``(sampled [requests, steps + 1], num_sampled [requests])``; as the shape package's ``rejection_sample``."""
    check(_supported(draft_logits, idx_mapping.shape[0]))
    return _ROUTES.package((draft_logits.shape[2], draft_logits.shape[1])).rejection_sample(
        target_logits, draft_logits, draft_sampled, pos, idx_mapping, temperature, seed, synthetic_rates
    )


def warmup_rejection_sample(device, vocab: int, steps: int) -> None:
    _ROUTES.package((vocab, steps)).warmup_rejection_sample(device)
