# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The IPC pool of the receive regions: a coarse-grained hipMalloc the peer maps, and the handle exchange."""

import ctypes
import functools

import torch
import torch.distributed as dist

from amd_kernels.common.hip_runtime import load_hip_runtime


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = load_hip_runtime()
    lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
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


def alloc_pool(nbytes: int, device: torch.device) -> tuple[int, bytes, torch.Tensor]:
    """A hipMalloc pool (coarse-grained, never freed) -> (pointer, IPC handle bytes, the pool as a uint8 tensor)."""
    hip = _hip()
    ptr = ctypes.c_void_p()
    with torch.cuda.device(device):
        rc = hip.hipMalloc(ctypes.byref(ptr), nbytes)
    assert rc == 0, f"hipMalloc({nbytes}) failed: {rc}"
    handle = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    pool = torch.as_tensor(_DevicePointer(ptr.value, nbytes), device=device)
    # the raw 64 bytes (a c_char field reads back only up to its first NUL)
    return ptr.value, ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle)), pool


def open_pool(handle: bytes, device: torch.device) -> int:
    hip = _hip()
    ptr = ctypes.c_void_p()
    h = _IpcHandle()
    assert len(handle) == ctypes.sizeof(h)
    ctypes.memmove(ctypes.addressof(h), handle, len(handle))
    with torch.cuda.device(device):
        rc = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), h, 1)  # lazy peer access
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value


def exchange(obj, rank: int, world: int, group) -> list:
    """Every rank's ``obj`` (broadcasts: ``all_gather_object`` is unavailable on gloo under inference mode)."""
    out = [[None] for _ in range(world)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]
