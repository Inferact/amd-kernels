# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""FP4 prefill indexer scorer on gfx950 (FlyDSL): MQA logits of ragged query rows (32 heads, head dim 128, MXFP4) over
staged index key pages, several rows per workgroup sharing each key page. The torch contract is ``reference.py``."""

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


__all__ = ["available", "score_rows"]
