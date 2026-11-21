# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank communicator: IPC gather buffer, flag words, device-side epoch and the launch sequences."""

import ctypes
import functools

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from amd_kernels.common.tensor_shim import _run_compiled
from . import kernel_q_push as KP
from . import kernel_qb as KQ
from . import kernel_wuk as KW
from .flags import DATA_OFF

WORLD = 8
HEADS = 12
MAX_ROWS = 256
GATHER_BYTES = {"latent": MAX_ROWS * 96 * KW.OW * 2, "q": 2 * MAX_ROWS * 96 * KW.QW * 2}


class _DevicePointer:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def exchange(obj, rank: int, group) -> list:
    """Every rank's ``obj`` (broadcasts on the CPU group: all_gather_object fails on gloo under inference mode)."""
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


@functools.cache
def qb_kernel(mt, nt, nw):
    return KQ.compile_qb(mt, nt, nw)


@functools.cache
def wuk_kernel(heads, mt, ns, mode):
    return KW.compile_wuk(heads, mt, ns, mode)


@functools.cache
def q_push_kernel(nt1, nsplit, mmax, nw, stag=0, fp8out=0, timing=False, skeleton=0):
    return KP.compile_q_push(nt1, nsplit, mmax, timing, skeleton, fp8out, 96, nw, stag)


class QueryGatherComm:
    """One communicator per stream, created collectively on the eight DCP ranks.

    ``latent``: q_b, W_UK of the 12 local heads, push all-gather of the latent query into the eight gather buffers.
    ``q``: q_b with the push all-gather of q and the W_UK of all 96 heads from the replicated weight, one launch.
    The ``latent`` result is a view of the communicator's buffer, valid until this communicator's next call; every rank
    makes the same calls in the same order, on one stream, with the consumer of a result on that stream too. Graph-safe:
    the epoch lives on the device."""

    def __init__(self, rank, group, device, mode):
        assert mode in GATHER_BYTES and dist.get_world_size(group) == WORLD
        self.rank, self.group, self.mode, self.device = rank, group, mode, torch.device(device)
        nbytes = DATA_OFF + GATHER_BYTES[mode]
        ptr, handle = ops.allocate_shared_buffer_and_handle(nbytes)
        every = exchange(handle, rank, group)
        self.bufs = [ptr if r == rank else ops.open_mem_handle(every[r]) for r in range(WORLD)]
        self.data = torch.as_tensor(_DevicePointer(ptr + DATA_OFF, nbytes - DATA_OFF), device=self.device).view(
            torch.bfloat16
        )
        self.state = torch.zeros(4, dtype=torch.int32, device=self.device)
        self.dbg = torch.zeros(65536 * 16, dtype=torch.int32, device=self.device)
        self.scratch = torch.empty((MAX_ROWS, HEADS * KW.QW), dtype=torch.bfloat16, device=self.device)
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def _latent(self, q_a, w_qb, w_uk, cfg):
        m = q_a.shape[0]
        (mt1, nt1, nw1), (mt2, ns2) = cfg
        stream = torch.cuda.current_stream()
        _run_compiled(
            qb_kernel(mt1, nt1, nw1), q_a.data_ptr(), q_a.stride(0), w_qb.data_ptr(), self.scratch.data_ptr(), m, stream
        )
        g2 = KW.grid_wuk(m, HEADS, mt2, ns2)
        assert g2 <= 4096
        _run_compiled(
            wuk_kernel(HEADS, mt2, ns2, "push"),
            self.scratch.data_ptr(),
            HEADS * KW.QW,
            w_uk.data_ptr(),
            self.data.data_ptr(),
            m,
            *self.bufs,
            self.rank,
            self.state.data_ptr(),
            g2,
            stream,
        )
        return self.data[: m * 96 * KW.OW].view(m, 96, KW.OW)

    def _push(self, q_a, w_qb, w_uk, out, out8, cfg, fp8out, timing=False, skeleton=0):
        m = q_a.shape[0]
        nt1, nsplit, nw, mmax, stag = cfg
        _run_compiled(
            q_push_kernel(nt1, nsplit, mmax, nw, stag, fp8out, timing, skeleton),
            q_a.data_ptr(),
            q_a.stride(0),
            w_qb.data_ptr(),
            w_uk.data_ptr(),
            out.data_ptr() if out is not None else 0,
            out8.data_ptr() if out8 is not None else 0,
            m,
            *self.bufs,
            self.rank,
            self.state.data_ptr(),
            self.dbg.data_ptr(),
            torch.cuda.current_stream(),
        )

    def close(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        hip = ops._hip()
        hip.hipFree.argtypes = [ctypes.c_void_p]
        hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        for r, p in enumerate(self.bufs):
            rc = (hip.hipFree if r == self.rank else hip.hipIpcCloseMemHandle)(ctypes.c_void_p(p))
            assert rc == 0, rc
        self.bufs = []
