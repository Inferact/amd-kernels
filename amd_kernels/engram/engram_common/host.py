# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Engram entries: the n-gram row pull over the owner ranks, routed by (ranks, layers, hashed columns)
(``_NGRAM_ROWS``), and the gated update, routed by (residual streams, hidden) (``_GATED_POST``)."""

from amd_kernels.routes import Routes, check, first, prepare, previous_form

_NGRAM_ROWS = Routes("ngram_rows", {(8, 2, 24): "amd_kernels.engram.ngram_rows_mxfp8_d256_c24_ranks8"},
                     ("ranks", "layers", "columns"))
_GATED_POST = Routes("gated_post", {(4, 5120): "amd_kernels.engram.gated_post_bf16_h5120_hc4"}, ("streams", "hidden"))


def ngram_rows_shapes() -> frozenset:
    """(ranks, Engram layers, hashed columns) keys."""
    return _NGRAM_ROWS.shapes()


def _ngram_key(primes, weight_ptrs):
    return (len(weight_ptrs[0]), primes.shape[0], primes.shape[1])


def _integer_form(world: int, layers: int, columns: int) -> str | None:
    return _NGRAM_ROWS.blocker((world, layers, columns))


@previous_form(_integer_form)
def ngram_rows_blocker(token_map, multipliers, primes, offsets, pad_id, peer_rows, weight_ptrs, scale_ptrs,
                       **kwargs) -> str | None:  # fmt: skip
    """None when a package serves the n-gram hash + MXFP8 row pull of these Engram layers over the ranks of one node
    (``primes`` [layers, columns], ``weight_ptrs`` [layers][ranks])."""
    return _NGRAM_ROWS.blocker(_ngram_key(primes, weight_ptrs))


def ngram_rows(token_map, multipliers, primes, offsets, pad_id, peer_rows, weight_ptrs, scale_ptrs, **kwargs):
    """The package's ``NgramRows`` (peer tables mapped, hash constants on the device): call it per step with
    ``(input_ids, positions, query_start_loc, lookback_ids, tile_rows=0)``; ``warm_up()`` before graph capture,
    ``close()`` at the end."""
    key = _ngram_key(primes, weight_ptrs)
    check(_NGRAM_ROWS.blocker(key))
    return _NGRAM_ROWS.package(key).NgramRows(token_map, multipliers, primes, offsets, pad_id, peer_rows, weight_ptrs,
                                              scale_ptrs, **kwargs)


def prepare_ngram_rows(token_map, multipliers, primes, offsets, pad_id, peer_rows, weight_ptrs, scale_ptrs,
                       **kwargs):  # fmt: skip
    """``ngram_rows``' object, or the reason no package serves these layers."""
    args = (token_map, multipliers, primes, offsets, pad_id, peer_rows, weight_ptrs, scale_ptrs)
    return prepare(ngram_rows_blocker(*args, **kwargs), lambda: ngram_rows(*args, **kwargs))


def gated_post_shapes() -> frozenset:
    """(residual streams, hidden) keys."""
    return _GATED_POST.shapes()


def gated_post_blocker(hidden, kv, q_weight=None, k_weight=None, **kwargs) -> str | None:
    """None when a package serves the gated Engram update of ``hidden`` [tokens, streams, hidden] bf16 with ``kv``
    [tokens, (streams + 1) * hidden]."""
    if hidden.dim() != 3 or kv.dim() != 2:
        return "gated_post: hidden must be [tokens, streams, hidden], kv [tokens, (streams + 1) * hidden]"
    key = tuple(hidden.shape[1:])
    return first(
        _GATED_POST.blocker(key),
        None if kv.shape[1] == (key[0] + 1) * key[1] else f"gated_post: kv width {kv.shape[1]} is not "
        f"{(key[0] + 1) * key[1]}",
        None if str(hidden.dtype).endswith("bfloat16") else "gated_post: needs bf16 inputs",
    )


def gated_post(hidden, kv, q_weight, k_weight, **kwargs):
    """The gated update of the residual streams (``token_mask``, ``out`` (may be ``hidden``), eps, clamp as the
    shape package's ``gated_post``)."""
    check(gated_post_blocker(hidden, kv))
    return _GATED_POST.package(tuple(hidden.shape[1:])).gated_post(hidden, kv, q_weight, k_weight, **kwargs)
