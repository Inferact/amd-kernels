# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The ctypes binding to torch's HIP runtime (``amd_kernels/common/hip_runtime.py``)."""

from amd_kernels.common import hip_runtime


def test_prefers_torch_bundled_hip(monkeypatch, tmp_path):
    """The runtime must be torch's, not whichever libamdhip64.so the loader finds
    first (UCX/NIXL load /opt/rocm's copy)."""
    torch_lib = tmp_path / "lib"
    torch_lib.mkdir()
    (torch_lib / "libamdhip64.so").touch()
    monkeypatch.setattr(hip_runtime, "_torch_lib_dir", lambda: str(torch_lib))
    assert hip_runtime.hip_runtime_path() == str(torch_lib / "libamdhip64.so")
    monkeypatch.setattr(hip_runtime, "_torch_lib_dir", lambda: str(tmp_path / "no"))
    assert hip_runtime.hip_runtime_path() == "libamdhip64.so"


def test_handles_do_not_share_argtypes():
    """The host modules set argtypes with their own _IpcHandle types on the same
    HIP symbols; their handles must not share function objects."""
    a, b = hip_runtime.load_hip_runtime(), hip_runtime.load_hip_runtime()
    assert a.hipIpcGetMemHandle is not b.hipIpcGetMemHandle
