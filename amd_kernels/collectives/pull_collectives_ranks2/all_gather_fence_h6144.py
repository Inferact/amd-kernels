# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The sequence-parallel attention phase's unpacking all-to-all on two ranks, as a per-phase-stream communicator: the
packed 4096-code / 128-scale attention rows of both ranks -> this rank's o_proj_full input, the scales in the GEMM's
layout (``pull_unpack_q4096_shuffled``: the release / ticket / publish in the unpack launch up to
``FENCE_MAX_TOKENS``, a separate publication above). The payload never uses an atomic output add; only flags /
tickets use atomics.
"""
import torch

from .host import Collectives, Pools, WORLD


class FencedAllGatherComm:
    """Per-phase-stream communicator on the caller's ``Pools``, constructed before serving. Eager-only sequence
    state (the phase's ``Collectives``); no host synchronization."""

    def __init__(self, pools: Pools):
        self.p = pools

    def all_to_all_unpack_shuffled(self, coll: Collectives, packed: torch.Tensor):
        """Packed 4096-code / 128-scale attention rows ``[2 rows, 4224]`` -> o_proj_full's inputs: codes
        ``[rows, 8192]`` and scales in its GEMM layout (``mxfp8_gemm(..., activation_scale_preshuffled=True)``)."""
        from .pull_unpack_q4096_shuffled import all_to_all_unpack_q4096_shuffled

        return all_to_all_unpack_q4096_shuffled(coll, packed)

    def warm_up(self):
        """Compile both publication forms of the unpack before eager serving."""
        from .pull_unpack_q4096_shuffled import FENCE_MAX_TOKENS

        stream = torch.cuda.current_stream(self.p.device)
        for rows in (32, FENCE_MAX_TOKENS // WORLD + 32):
            c = Collectives(self.p, 0).bind(stream)
            packed = c.alloc((WORLD * rows, 4224), torch.uint8)
            packed.zero_()
            self.all_to_all_unpack_shuffled(c, packed)
        torch.cuda.synchronize(self.p.device)
