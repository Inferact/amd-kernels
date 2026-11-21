# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Handle exchange between the eight ranks and the launch of a compiled launcher.

Adapted from norm/allreduce_attnres_rmsnorm_h7168_ranks8/ipc.py of this repository.
"""
import struct

import torch.distributed as dist

from .device import WORLD


def run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def f32_bits(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


def exchange(obj, rank: int, group) -> list:
    """Every rank's ``obj`` (broadcasts on the CPU group: all_gather_object fails on gloo under inference mode)."""
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]
