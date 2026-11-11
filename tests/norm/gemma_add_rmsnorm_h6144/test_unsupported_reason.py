# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``allreduce_add_rmsnorm.unsupported_reason``: the conditions a model integration asks the library about."""

from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as AR


def test_reasons():
    assert "4 ranks" in AR.unsupported_reason(2)
    assert "hidden size 6144" in AR.unsupported_reason(4, 4096)


def test_expandable_segments_env(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "aiter.dist.device_communicators.custom_all_reduce", None)
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.9, expandable_segments:True")
    assert AR._expandable_segments()
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    assert not AR._expandable_segments()
