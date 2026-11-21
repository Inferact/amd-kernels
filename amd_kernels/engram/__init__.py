# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Engram op entries (``amd_kernels.routes``): the n-gram row pull over the owner ranks and the gated update."""

from amd_kernels.routes import Routes, check, first

_NGRAM_ROWS = Routes("ngram_rows", {(8, 2, 24): "amd_kernels.engram.ngram_rows_mxfp8_d256_c24_ranks8"})
_GATED_POST = Routes("gated_post", {(4, 5120): "amd_kernels.engram.gated_post_bf16_h5120_hc4"})


def ngram_rows_shapes() -> frozenset:
    """(ranks, Engram layers, hashed columns) keys."""
    return _NGRAM_ROWS.shapes()


def ngram_rows_blocker(world: int, layers: int, columns: int) -> str | None:
    """None when a package serves the n-gram hash + MXFP8 row pull of ``layers`` Engram layers of ``columns`` hashed
    columns over ``world`` ranks of one node."""
    return _NGRAM_ROWS.blocker((world, layers, columns))


def ngram_rows(token_map, multipliers, primes, offsets, pad_id, peer_rows, weight_ptrs, scale_ptrs, **kwargs):
    """The package's ``NgramRows`` (peer tables mapped, hash constants on the device): call it per step with
    ``(input_ids, positions, query_start_loc, lookback_ids, tile_rows=0)``; ``warm_up()`` before graph capture,
    ``close()`` at the end. ``primes`` [layers, columns], ``weight_ptrs`` [layers][ranks]."""
    key = (len(weight_ptrs[0]), primes.shape[0], primes.shape[1])
    check(ngram_rows_blocker(*key))
    return _NGRAM_ROWS.package(key).NgramRows(token_map, multipliers, primes, offsets, pad_id, peer_rows,
                                              weight_ptrs, scale_ptrs, **kwargs)


def gated_post_shapes() -> frozenset:
    """(residual streams, hidden) keys."""
    return _GATED_POST.shapes()


def gated_post_blocker(hidden, kv) -> str | None:
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
