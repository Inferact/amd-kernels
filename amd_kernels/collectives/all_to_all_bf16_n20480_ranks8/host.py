# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One stream per communicator; construct collectively before capture. Every rank, including empty receivers,
participates in every exchange (the sequence words of all eight senders)."""
import ctypes
import functools
import torch
import torch.distributed as dist
from amd_kernels.common import hip_runtime as ops
from amd_kernels.common.hip_runtime import load_hip_runtime
from . import kernel
from .kernel import BUFFERS, N, RECV_ROWS, V, WORLD

# rows up to which the GEMM pushes its rows itself (owner mode, all-gather mode): the measured crossovers of the README
SERIAL_ABOVE = {True: 96, False: 16}


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = load_hip_runtime()
    lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    return lib


class _DevicePointer:
    """A device allocation as torch sees it (``torch.as_tensor`` wraps it)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (nbytes,),
            "typestr": "|u1",
            "strides": None,
            "version": 3,
        }


def _alloc_pool(nbytes: int, device: torch.device) -> tuple[int, bytes, torch.Tensor]:
    """A hipMalloc pool (coarse-grained, freed by collective close) -> (pointer, IPC handle bytes, the pool as a uint8 tensor)."""
    hip = _hip()
    ptr = ctypes.c_void_p()
    with torch.cuda.device(device):
        rc = hip.hipMalloc(ctypes.byref(ptr), nbytes)
    assert rc == 0, f"hipMalloc({nbytes}) failed: {rc}"
    handle = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    pool = torch.as_tensor(_DevicePointer(ptr.value, nbytes), device=device)
    assert pool.data_ptr() == ptr.value
    # the raw 64 bytes (a c_char field reads back only up to its first NUL)
    raw = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
    return ptr.value, raw, pool


def _open_pool(handle: bytes, device: torch.device) -> int:
    hip = _hip()
    ptr = ctypes.c_void_p()
    h = _IpcHandle()
    assert len(handle) == ctypes.sizeof(h)
    ctypes.memmove(ctypes.addressof(h), handle, len(handle))
    with torch.cuda.device(device):
        rc = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), h, 1)  # lazy peer access
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value


def _exchange(obj, rank: int, world: int, group) -> list:
    """Every rank's ``obj`` (broadcasts: ``all_gather_object`` is unavailable on
    gloo under inference mode, which the profile run enters)."""
    out = [[None] for _ in range(world)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


def _run(exe, *args):
    import flydsl.compiler as flyc
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


class RowAllToAll:
    def __init__(self, rank, group, device, max_rows=RECV_ROWS):
        assert dist.get_world_size(group) == WORLD
        assert rank == dist.get_rank(group) and 0 <= rank < WORLD
        assert 0 < max_rows <= RECV_ROWS
        self.rank, self.device = rank, torch.device(device)
        self.group = group
        self.closed = False
        self.max_rows = max_rows
        ptr, handle, self.pool = _alloc_pool(BUFFERS * max_rows * V * 2, device)
        flag, fh = ops.allocate_shared_buffer_and_handle(512)
        handles = _exchange((handle, fh), rank, WORLD, group)
        bases = [ptr if r == rank else _open_pool(h, device) for r, (h, _) in enumerate(handles)]
        self.ptrs = [b + buf * max_rows * V * 2 for buf in range(BUFFERS) for b in bases]
        self._bases = bases
        self.flags = [flag if r == rank else ops.open_mem_handle(h) for r, (_, h) in enumerate(handles)]
        self.peer_table = torch.tensor(self.ptrs + self.flags, dtype=torch.int64, device=device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self._recv = self.pool.view(torch.bfloat16).view(BUFFERS, max_rows, V)
        self.pushes = {m: kernel.compile_push(m) for m in ("owner", "all")}
        self.waiter = kernel.compile_wait()

    def recv(self, rows, buf=0):
        """This rank's receive buffer ``buf`` [rows, 163840] bf16: the rows pushed by the exchange, in place."""
        assert 0 <= rows <= self.max_rows and 0 <= buf < BUFFERS
        return self._recv[buf, :rows]

    def send_args(self, buf=0):
        """What a producer that pushes the rows itself (`bf16_gemm_n20480_k7168`'s push modes) needs: the device
        table of the buffers and flag blocks, the device sequence counter, this rank's flag block address, rank and
        the buffer of this exchange (alternate 0 / 1 between consecutive exchanges). The producer stores into
        `recv(.., buf)` of the owners (or of every rank) and advances the counter; `wait` then holds the stream
        until all eight senders published."""
        assert not self.closed and 0 <= buf < BUFFERS
        return (self.peer_table, self.counter, self.flags[self.rank], self.rank, buf)

    def wait(self):
        """Hold the stream until every sender's rows of this exchange are in place (one workgroup)."""
        assert not self.closed
        _run(self.waiter, self.counter, self.flags[self.rank], torch.cuda.current_stream(self.device))
        return self

    def __call__(self, x, counts=None, buf=0):
        """Exchange x [rows, 20480] bf16 (this rank's columns of every row) into buffer ``buf``: with ``counts`` (CUDA
        int32 [8], the same on every rank, summing to the rows, the rows sorted by owner) the rows of owner r land in
        rank r's ``recv(.., buf)``; with ``counts=None`` every rank receives all rows (all-gather). Returns the
        communicator; the caller reads ``recv(counts[rank], buf)`` or ``recv(rows, buf)``. Alternate ``buf`` between
        consecutive exchanges. Graph safe, counts may change between replays."""
        assert not self.closed and 0 <= buf < BUFFERS
        assert x.device == self.counter.device and x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape[1] == N
        assert x.is_contiguous() and 0 <= x.shape[0] <= self.max_rows
        mode = "all" if counts is None else "owner"
        if counts is None:
            counts = self._zero_counts()
        assert counts.shape == (WORLD,) and counts.dtype == torch.int32 and counts.is_contiguous()
        rows = x.shape[0]
        nwg = -(-max(rows * 2, 1) // 8) * 8
        stream = torch.cuda.current_stream(self.device)
        _run(self.pushes[mode], x.data_ptr(), self.peer_table, counts, self.counter, self.flags[self.rank], rows,
             self.rank, nwg, buf, stream)
        return self.wait()

    def gemm_exchange(self, x, w, row_indices, counts=None, buf=0, serial_above=None):
        """The lm_head shard GEMM of the rows ``x[row_indices]`` (``bf16_n20480_k7168``) and the exchange of its output:
        the GEMM pushes the rows itself up to ``serial_above`` rows (one launch less), above that it writes them locally
        and one contiguous push moves them (full-row streams suit the xGMI better than the tile's row segments).
        ``serial_above`` defaults to the measured crossover of the mode. Same arguments and result as ``__call__`` with
        that GEMM's output; returns the communicator."""
        from amd_kernels.gemm.bf16_n20480_k7168 import bf16_gemm_n20480_k7168

        assert not self.closed and 0 <= buf < BUFFERS and row_indices.ndim == 1
        rows = row_indices.numel()
        if serial_above is None:
            serial_above = SERIAL_ABOVE[counts is not None]
        if rows <= serial_above:
            bf16_gemm_n20480_k7168(x, w, row_indices=row_indices, push=(self.send_args(buf), counts))
            return self.wait()
        if not hasattr(self, "_local"):
            self._local = torch.empty(self.max_rows, N, dtype=torch.bfloat16, device=self.device)
        out = self._local[:rows]
        bf16_gemm_n20480_k7168(x, w, out, row_indices=row_indices)
        return self(out, counts, buf)

    def _zero_counts(self):
        if not hasattr(self, "_zeros"):
            self._zeros = torch.zeros(WORLD, dtype=torch.int32, device=self.device)
        return self._zeros

    def warmup(self):
        """Collective: compile both pushes and the wait eagerly. Every rank calls this before capture."""
        for rows in (8, self.max_rows):
            x = torch.zeros(rows, N, dtype=torch.bfloat16, device=self.device)
            counts = torch.tensor([rows // WORLD + int(r < rows % WORLD) for r in range(WORLD)], dtype=torch.int32,
                                  device=self.device)
            self(x, counts, 0)
            self(x, None, 1)
        torch.cuda.synchronize(self.device)

    def close(self):
        """Collective teardown after all graphs using this object are retired."""
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        lib = load_hip_runtime()
        lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        lib.hipFree.argtypes = [ctypes.c_void_p]
        for r in range(WORLD):
            if r != self.rank:
                for ptr in (self._bases[r], self.flags[r]):
                    rc = lib.hipIpcCloseMemHandle(ctypes.c_void_p(ptr))
                    assert rc == 0, (r, rc)
        dist.barrier(group=self.group)
        for ptr in (self._bases[self.rank], self.flags[self.rank]):
            rc = lib.hipFree(ctypes.c_void_p(ptr))
            assert rc == 0, rc
        self.pool = None
        self.closed = True
