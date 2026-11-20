# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One stream per communicator; construct collectively before capture.

Two slots suffice: publishing phase c+1 follows the pull of phase c, so all
peers have read slot c before anyone reuses it at phase c+2. Every rank,
including empty receivers, participates in the peer wait.
"""
import ctypes
import functools
import torch
import torch.distributed as dist
from amd_kernels.common import hip_runtime as ops
from amd_kernels.common.hip_runtime import load_hip_runtime
from . import kernel
from .stage_publish import compile_stage_publish

N, WORLD, V = kernel.N, kernel.WORLD, kernel.N * kernel.WORLD
STAGES = (64, 128, 512)
PULL = (16, 4)  # (remote, local) workgroups: best at 4-512 owned rows, 37-55 GB/s (sweep_pull.py in benchmarks/)


def _stage_wgs(rows: int) -> int:
    return 64 if rows <= 128 else (128 if rows <= 512 else 512)


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
    def __init__(self, rank, group, device, max_rows=2048):
        assert dist.get_world_size(group) == WORLD
        assert rank == dist.get_rank(group) and 0 <= rank < WORLD
        assert 0 < max_rows <= 2048
        self.rank, self.device = rank, torch.device(device)
        self.group = group
        self.closed = False
        self.max_rows = max_rows
        self.slot_bytes = max_rows * N * 2
        ptr, handle, self.pool = _alloc_pool(2 * self.slot_bytes, device)
        flag, fh = ops.allocate_shared_buffer_and_handle(256)
        handles = _exchange((handle, fh), rank, WORLD, group)
        self.ptrs = [ptr if r == rank else _open_pool(h, device) for r, (h, _) in enumerate(handles)]
        self.flags = [flag if r == rank else ops.open_mem_handle(h) for r, (_, h) in enumerate(handles)]
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self.stages = {w: compile_stage_publish(w) for w in STAGES}
        self.pull_exe = kernel.compile_pull(*PULL)

    def __call__(self, x, counts, out):
        """Exchange owner-sorted x into preallocated out, bit for bit.

        counts is CUDA int32 [2], identical on all ranks, nonnegative,
        sum(counts) == x.shape[0]; counts[rank] <= out.shape[0]. These device
        values are caller preconditions; checking them would synchronize.
        Counts may change between replays. One stream uses a communicator;
        independent streams need independent communicators. Every rank calls
        every exchange, including ranks receiving zero rows.
        """
        assert not self.closed
        assert x.device == self.counter.device == counts.device == out.device
        assert x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape[1] == N and x.is_contiguous()
        assert x.shape[0] <= self.max_rows
        assert counts.shape == (WORLD,) and counts.dtype == torch.int32 and counts.is_contiguous()
        assert out.dtype == x.dtype and out.ndim == 2 and out.shape[1] == V and out.is_contiguous()
        stream = torch.cuda.current_stream(self.device)
        rows = x.shape[0]
        _run(self.stages[_stage_wgs(rows)], x.data_ptr(), self.ptrs[self.rank], self.counter,
             x.numel() // 2, self.slot_bytes, self.flags[self.rank], stream)
        return self.pull(rows, counts, out)

    def slot(self):
        """This rank's send slot for a producer that writes and publishes it itself (the lm_head GEMM's publish
        epilogue, bf16_gemm_n100032_k6144(..., publish=comm.slot())): (pool address, slot bytes, device counter,
        flag address). The producer writes slot counter[0] & 1 with system-scope stores and its last workgroup
        publishes, as stage_publish does; then call pull. Graph safe: the slot is chosen on the device."""
        assert not self.closed
        return (self.ptrs[self.rank], self.slot_bytes, self.counter, self.flags[self.rank])

    def pull(self, rows, counts, out):
        """The receive half of __call__, after this rank's ``rows`` owner-sorted rows were published (by
        __call__'s stage or by a producer through slot()). Same preconditions as __call__."""
        assert not self.closed
        assert counts.device == out.device == self.counter.device and 0 <= rows <= self.max_rows
        assert counts.shape == (WORLD,) and counts.dtype == torch.int32 and counts.is_contiguous()
        assert out.dtype == torch.bfloat16 and out.ndim == 2 and out.shape[1] == V and out.is_contiguous()
        stream = torch.cuda.current_stream(self.device)
        _run(self.pull_exe, out.data_ptr(), *self.ptrs, *self.flags, self.counter, counts,
             self.slot_bytes, self.rank, stream)
        return out

    def warmup(self):
        """Collective: compile every stage grid the row range reaches and the pull, eagerly.

        Every rank calls this before capture. The buffers are temporary; the
        communicator retains only its IPC pool, flags and device counter.
        """
        rows_list = sorted({min(r, self.max_rows) for r in (4, 256, self.max_rows)})
        assert {_stage_wgs(r) for r in range(1, self.max_rows + 1)} <= {_stage_wgs(r) for r in rows_list}
        for rows in rows_list:
            counts = [rows // WORLD + int(r < rows % WORLD) for r in range(WORLD)]
            x = torch.zeros(rows, N, dtype=torch.bfloat16, device=self.device)
            cs = torch.tensor(counts, dtype=torch.int32, device=self.device)
            out = torch.empty(counts[self.rank], V, dtype=torch.bfloat16, device=self.device)
            self(x, cs, out)
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
                for ptr in (self.ptrs[r], self.flags[r]):
                    rc = lib.hipIpcCloseMemHandle(ctypes.c_void_p(ptr))
                    assert rc == 0, (r, rc)
        dist.barrier(group=self.group)
        for ptr in (self.ptrs[self.rank], self.flags[self.rank]):
            rc = lib.hipFree(ctypes.c_void_p(ptr))
            assert rc == 0, rc
        self.pool = None
        self.closed = True
