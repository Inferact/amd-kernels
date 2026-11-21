# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank mailbox buffers and the entry of the all-reduce + AttnRes + RMSNorm kernel."""
import ctypes

import torch
import torch.distributed as dist

from amd_kernels import warm_up
from amd_kernels.common import hip_runtime as ops

from . import kernel as K
from .ipc import _f32_bits, _run_compiled, exchange
from .kernel import BANK_ROWS, BUF_BYTES, H, MAX_ROWS, WORLD

NAME = "norm/allreduce_attnres_rmsnorm_h7168_ranks8"


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def _release(ptr, peer):
    hip = ops._hip()
    fn = hip.hipIpcCloseMemHandle if peer else hip.hipFree
    fn.argtypes = [ctypes.c_void_p]
    rc = fn(ctypes.c_void_p(ptr))
    assert rc == 0, f"HIP buffer release failed: {rc}"


def _kernel(has_prefix, out_norm, has_bank=True, reduce=True, packed_out=False):
    return K.compile_allreduce_attnres_rmsnorm(has_prefix, out_norm, has_bank, reduce, packed_out)


def _mode(comm, partial, prefix, bank, norm_weight, qk_weight, num_blocks, block_write_idx=-1, out_norm_weight=None,
          eps=1e-6, out_norm_eps=1e-6, reduce=True, out_packed=False):
    """What a call fixes about its kernel besides the row count (``amd_kernels.warm_up``)."""
    return prefix is not None, out_norm_weight is not None, bank is not None, bool(reduce), bool(out_packed)


def _sizes(mode):
    return (1,)


def _inputs(mode, n, comm, partial, prefix, bank, norm_weight, qk_weight, num_blocks, block_write_idx=-1,
            out_norm_weight=None, eps=1e-6, out_norm_eps=1e-6, reduce=True, out_packed=False):
    dev = partial.device
    z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    return (comm, z(n, H), None if prefix is None else z(n, H), None if bank is None else z(n, BANK_ROWS, H),
            norm_weight, qk_weight, num_blocks, block_write_idx, out_norm_weight, eps, out_norm_eps, reduce, out_packed), {}


@warm_up.entry(NAME, mode=_mode, sizes=_sizes, inputs=_inputs)
def allreduce_attnres_rmsnorm(comm, partial, prefix, bank, norm_weight, qk_weight, num_blocks, block_write_idx=-1,
                              out_norm_weight=None, eps=1e-6, out_norm_eps=1e-6, reduce=True, out_packed=False):
    """Reduce ``partial`` over the eight ranks, add ``prefix`` and mix the bank rows (see reference.py).

    Returns (out, prefix_out): ``prefix_out`` is ``prefix`` updated in place, or a new tensor when ``prefix`` is None.
    ``bank`` None: no mix, ``out`` is the (normed) prefix. ``reduce=False``: ``partial`` is already the reduced delta
    on every rank and the call communicates nothing. ``out_packed``: ``out`` is [ceil(M / 16) * 16, 7168] in the
    (16 rows x 32 columns) tile-packed layout of the MFMA GEMMs: the 16 B pack of row r, columns 8j .. 8j + 7 is at
    byte ((r // 16 * 224 + j // 4) * 64 + j % 4 * 16 + r % 16) * 16; padding rows are not written.
    """
    return comm._launch(partial, prefix, bank, norm_weight, qk_weight, num_blocks, block_write_idx, out_norm_weight,
                        eps, out_norm_eps, reduce, out_packed)


class AttnResComm:
    """One communicator per stream, created collectively on eight ranks (``dist`` group with a CPU backend).

    Every call runs on the current stream, in the same order on every rank, with the same row count; the mailboxes
    are not shared between streams. Inputs are plain tensors: nothing is registered for graph capture.
    """

    def __init__(self, rank, group, device):
        if dist.get_world_size(group) != WORLD:
            raise ValueError("eight ranks required")
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.current_device() != device.index or dist.get_rank(group) != rank:
            raise ValueError("select this rank device before creating the communicator")
        self.rank, self.group, self.device = rank, group, device
        ptr, handle = ops.allocate_shared_buffer_and_handle(BUF_BYTES)
        peers = exchange(handle, rank, group)
        self.bufs = [ptr if r == rank else ops.open_mem_handle(peers[r]) for r in range(WORLD)]
        torch.as_tensor(_DevicePointer(ptr, BUF_BYTES), device=device).fill_(255)
        self.buf_table = torch.tensor(self.bufs, dtype=torch.int64, device=device)
        self.counts = torch.zeros(MAX_ROWS, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def _launch(self, x, prefix, bank, norm_weight, qk_weight, num_blocks, block_write_idx, out_norm_weight, eps,
                out_norm_eps, reduce, out_packed):
        if not self.bufs:
            raise RuntimeError("communicator is closed")
        m = x.shape[0]
        dev = self.device
        if not (0 < m <= MAX_ROWS and tuple(x.shape) == (m, H)):
            raise ValueError(f"expected 1..{MAX_ROWS} rows of hidden {H}")
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != dev:
            raise ValueError("partial must be contiguous bf16 on the communicator device")
        if prefix is not None and (prefix.shape != x.shape or prefix.dtype != x.dtype or prefix.stride(1) != 1
                                   or prefix.stride(0) % 8 or prefix.device != dev):
            raise ValueError("prefix must be bf16 [rows, 7168] with a row stride divisible by eight")
        if bank is not None and (bank.ndim != 3 or bank.shape[0] != m or bank.shape[1] > BANK_ROWS
                                 or bank.shape[2] != H or bank.dtype != x.dtype or bank.stride(2) != 1
                                 or bank.stride(1) % 8 or bank.stride(0) % 8 or bank.device != dev):
            raise ValueError("bank must be bf16 [rows, <=8, 7168] with strides divisible by eight")
        for w in (norm_weight, qk_weight, out_norm_weight):
            if w is not None and (w.numel() != H or w.dtype != x.dtype or w.stride(-1) != 1 or w.device != dev):
                raise ValueError("weights must be contiguous bf16 [7168]")
        if bank is None:
            num_blocks, block_write_idx = 0, -1
        elif not 0 <= num_blocks <= bank.shape[1] or not (block_write_idx < 0 or num_blocks <= block_write_idx
                                                         < bank.shape[1]):
            raise ValueError("num_blocks <= block_write_idx < bank rows, or block_write_idx < 0")
        out = torch.empty(((m + 15) // 16 * 16, H) if out_packed else x.shape, dtype=x.dtype, device=dev)
        pout = prefix if prefix is not None else torch.empty_like(x)
        has_prefix = prefix is not None
        _run_compiled(
            _kernel(has_prefix, out_norm_weight is not None, bank is not None, reduce, bool(out_packed)),
            m, self.rank, x.data_ptr(), self.buf_table.data_ptr(), self.counts.data_ptr(),
            prefix.data_ptr() if has_prefix else 0, prefix.stride(0) if has_prefix else 0,
            pout.data_ptr(), pout.stride(0), bank.data_ptr() if bank is not None else 0,
            bank.stride(0) if bank is not None else 0, bank.stride(1) if bank is not None else 0, num_blocks,
            block_write_idx, norm_weight.data_ptr() if norm_weight is not None else 0,
            qk_weight.data_ptr() if qk_weight is not None else 0,
            out_norm_weight.data_ptr() if out_norm_weight is not None else 0, out.data_ptr(),
            _f32_bits(eps), _f32_bits(out_norm_eps), torch.cuda.current_stream())
        return out, pout

    def allreduce_attnres_rmsnorm(self, partial, prefix, bank, norm_weight, qk_weight, num_blocks,
                                  block_write_idx=-1, out_norm_weight=None, eps=1e-6, out_norm_eps=1e-6, reduce=True,
                                  out_packed=False):
        """The entry of this package on this communicator."""
        return allreduce_attnres_rmsnorm(self, partial, prefix, bank, norm_weight, qk_weight, num_blocks,
                                         block_write_idx, out_norm_weight, eps, out_norm_eps, reduce, out_packed)

    def close(self):
        if not self.bufs:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for r, ptr in enumerate(self.bufs):
            _release(ptr, r != self.rank)
        self.bufs = []
