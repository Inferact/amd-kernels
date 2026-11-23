# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Vocabulary-sharded embedding lookup for any (hidden, world size) with a package: one factory, routed by the shard's
hidden size and the group's size (``host._ROUTES``); an uncovered shard gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .host import (
    prepare_vocab_gather_comm,
    vocab_gather_blocker,
    vocab_gather_comm,
    vocab_gather_comm_blocker,
    vocab_gather_comm_unsupported_reason,
    vocab_gather_shapes,
    vocab_gather_unsupported_reason,
)

__all__ = ["available", "prepare_vocab_gather_comm", "vocab_gather_blocker", "vocab_gather_comm",
           "vocab_gather_comm_blocker", "vocab_gather_shapes"]
