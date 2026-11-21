# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Handle exchange and launch helpers of the eight-rank latent tail.

Adapted from norm/allreduce_attnres_rmsnorm_h7168_ranks8/ipc.py of this repository."""
import ctypes
import functools
import struct

import torch
import torch.distributed as dist

from amd_kernels.common.hip_runtime import load_hip_runtime

from .norm_math import WORLD


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


class DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

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


def as_tensor(ptr: int, nbytes: int, device) -> torch.Tensor:
    return torch.as_tensor(DevicePointer(ptr, nbytes), device=device)


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = load_hip_runtime()
    lib.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
    lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    return lib


def alloc_handle(ptr: int) -> tuple[bytes, int]:
    """(the IPC handle's 64 raw bytes of the allocation holding ptr, ptr's offset in it)."""
    hip = _hip()
    base, size = ctypes.c_void_p(), ctypes.c_size_t()
    rc = hip.hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), ptr)
    assert rc == 0, f"hipMemGetAddressRange failed: {rc}"
    h = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(h), base)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    return ctypes.string_at(ctypes.addressof(h), ctypes.sizeof(h)), ptr - base.value


def open_handle(raw: bytes) -> int:
    hip = _hip()
    h = _IpcHandle()
    ctypes.memmove(ctypes.addressof(h), raw, len(raw))
    ptr = ctypes.c_void_p()
    rc = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), h, 1)  # lazy peer access
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value


def close_handle(ptr: int) -> None:
    rc = _hip().hipIpcCloseMemHandle(ctypes.c_void_p(ptr))
    assert rc == 0, f"hipIpcCloseMemHandle failed: {rc}"
