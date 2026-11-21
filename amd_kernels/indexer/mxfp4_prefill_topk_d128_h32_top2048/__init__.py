# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Top-2048 select of the DSA prefill indexer on gfx950 (FlyDSL): per row, the 2048 largest logits of the row's ragged key window,
exact, ascending positions, ties to the smaller position. The torch contract is ``reference.py``."""

import functools


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable."""
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


def __getattr__(name):
    if name in __all__ and name != "available":
        from . import host

        return getattr(host, name)
    raise AttributeError(name)


__all__ = ["available", "topk_rows", "workspace_bytes"]
