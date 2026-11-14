# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""HIP IPC handles of the symmetric arena (``mega_moe_tp.SymmetricArena``).

From AITER ``aiter/ops/flydsl/ipc_heap.py`` (``UncachedIpcHeap``, ROCm/aiter edf58c67a), the calls this package uses.
"""

import ctypes
import functools

from amd_kernels.common.hip_runtime import load_hip_runtime

_HANDLE_BYTES = 64
_LAZY_ENABLE_PEER_ACCESS = 0x1


class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * _HANDLE_BYTES)]


@functools.cache
def load_hip() -> ctypes.CDLL:
    """torch's HIP runtime with the IPC calls declared (a handle of its own: amd_kernels.common.hip_runtime)."""
    hip = load_hip_runtime()
    hip.hipIpcGetMemHandle.restype = ctypes.c_int
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_Handle), ctypes.c_void_p]
    hip.hipIpcOpenMemHandle.restype = ctypes.c_int
    hip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _Handle, ctypes.c_uint]
    hip.hipIpcCloseMemHandle.restype = ctypes.c_int
    hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    hip.hipGetErrorString.restype = ctypes.c_char_p
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    return hip


def hip_check(err: int, *, what: str) -> None:
    if int(err) == 0:
        return
    try:
        s = load_hip().hipGetErrorString(int(err))
        msg = s.decode("utf-8", errors="replace") if s else f"hipError({err})"
    except Exception:  # noqa: BLE001
        msg = f"hipError({err})"
    raise RuntimeError(f"{what} failed: {msg}")


def get_mem_handle_bytes(base_ptr: int) -> bytes:
    h = _Handle()
    err = load_hip().hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(int(base_ptr)))
    hip_check(err, what="hipIpcGetMemHandle")
    return bytes(ctypes.string_at(ctypes.byref(h), _HANDLE_BYTES))


def open_mem_handle(handle_bytes: bytes) -> int:
    if len(handle_bytes) != _HANDLE_BYTES:
        raise ValueError(f"Expected {_HANDLE_BYTES}B handle")
    h = _Handle()
    ctypes.memmove(ctypes.byref(h), bytes(handle_bytes), _HANDLE_BYTES)
    out = ctypes.c_void_p()
    err = load_hip().hipIpcOpenMemHandle(ctypes.byref(out), h, ctypes.c_uint(_LAZY_ENABLE_PEER_ACCESS))
    hip_check(err, what="hipIpcOpenMemHandle")
    return int(out.value)


def close_mem_handle(base_ptr: int) -> None:
    err = load_hip().hipIpcCloseMemHandle(ctypes.c_void_p(int(base_ptr)))
    hip_check(err, what="hipIpcCloseMemHandle")
