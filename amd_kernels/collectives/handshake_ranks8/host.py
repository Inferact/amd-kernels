# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Communicator and entries of the eight-rank hand-off (see ``kernel.py``).

Create the communicator on every rank before graph capture (a collective) and call the entries on every rank in the
same order: the call number is the epoch.
"""
import functools

import torch
from amd_kernels.common.rank8_comm import Rank8Comm
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from .kernel import (
    FLAG_MAX_WORDS, LL_MAX_WORDS, block_bytes, compile_barrier, compile_exchange_flag, compile_exchange_ll,
    compile_exchange_pull,
)

LL_SIZES = (64, 128, 256, 512)
FLAG_SIZES = (1024, 2048, 4096, 8192, FLAG_MAX_WORDS)
SIZES = LL_SIZES + FLAG_SIZES


def make_comm(rank, group, device):
    return Rank8Comm(rank, group, device, block_bytes())


@functools.cache
def _barrier(variant, ncta):
    return compile_barrier(variant, ncta)


@functools.cache
def _ll(n, ncta):
    return compile_exchange_ll(n, ncta)


@functools.cache
def _flag(n, ncta):
    return compile_exchange_flag(n, ncta)


@functools.cache
def _pull(n, ncta):
    assert ncta == 1
    return compile_exchange_pull(n)


def _barrier_mode(comm, variant="wave", ncta=1):
    return (variant, ncta)


@entry("collectives/handshake_ranks8:barrier", mode=_barrier_mode, sizes=lambda mode: (1,),
       inputs=lambda mode, n, comm, variant="wave", ncta=1: ((comm, variant, ncta), {}))
def barrier(comm: Rank8Comm, variant="wave", ncta=1):
    """Return (on the stream) once every rank has entered the same call; ``variant`` wave | lane | pull."""
    _run_compiled(_barrier(variant, ncta), comm.ctr.data_ptr(), comm.rank, *comm.rot,
                  torch.cuda.current_stream(comm.device))


def _exchange_mode(comm, x, out, path=None, ncta=1):
    return (path, ncta)


def _exchange_inputs(mode, n, comm, x, out, path=None, ncta=1):
    dev = x.device
    zeros = torch.zeros((8, n), dtype=torch.int32, device=dev)
    return (comm, zeros, torch.zeros_like(zeros), path, ncta), {}


def _exchange_sizes(mode):
    path, ncta = mode
    return tuple(
        n for n in SIZES
        if n % ncta == 0 and (path != "ll" or n // ncta <= LL_MAX_WORDS) and (path == "ll" or (n // ncta) % 4 == 0)
    )


@entry("collectives/handshake_ranks8:exchange", mode=_exchange_mode, sizes=_exchange_sizes, inputs=_exchange_inputs)
def exchange(comm: Rank8Comm, x: torch.Tensor, out: torch.Tensor, path=None, ncta=1):
    """``out[s] = x_s[me]`` for int32 [8, n] ``x`` and ``out`` (row d of ``x`` goes to rank d), n in ``SIZES``.

    ``path``: ll | flag | pull (default ll up to 512 words per peer, else flag); ``ncta`` workgroups
    share the words (``pull`` takes one).
    """
    n = x.shape[1]
    assert n in SIZES and x.shape == out.shape and x.dtype == out.dtype == torch.int32
    assert x.is_contiguous() and out.is_contiguous() and x.shape[0] == 8
    path = path or ("ll" if n <= LL_SIZES[-1] else "flag")
    assert n % ncta == 0 and (path != "ll" or n // ncta <= LL_MAX_WORDS)
    exe = {"ll": _ll, "flag": _flag, "pull": _pull}[path](n, ncta)
    _run_compiled(exe, comm.ctr.data_ptr(), comm.rank, *comm.rot, x.data_ptr(), out.data_ptr(),
                  torch.cuda.current_stream(comm.device))
    return out
