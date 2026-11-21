# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shape-agnostic top-p mask: reads the vocabulary from ``logits`` and calls that vocabulary's package
(``_SHAPE_HOSTS``, imported on its first call)."""

import functools
import importlib

import torch

_SHAPE_HOSTS = {
    (163840,): "amd_kernels.sampling.top_p_mask_f32_v163840",
    (200064,): "amd_kernels.sampling.top_p_mask_f32_v200064",
}


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable."""
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def _shape_host(vocab: int):
    name = _SHAPE_HOSTS.get((vocab,))
    return importlib.import_module(name) if name is not None else None


def _no_kernel(vocab: int) -> str:
    return f"top_p_mask: no kernel for vocab={vocab} (covered: {sorted(_SHAPE_HOSTS)})"


def _off_device(**tensors: torch.Tensor) -> str | None:
    """Why the tensors are not all on the current CUDA device (the launch's), else None."""
    index = torch.cuda.current_device()
    for name, t in tensors.items():
        if t.get_device() != index:
            return f"top_p_mask: {name} on {t.device} (the current CUDA device)"
    return None


def top_p_mask_shapes() -> frozenset[tuple[int]]:
    """The (vocab,) keys with a package."""
    return frozenset(_SHAPE_HOSTS)


def top_p_mask(logits: torch.Tensor, p: torch.Tensor, buffer: torch.Tensor | None = None) -> torch.Tensor:
    """Masks fp32 ``logits`` [n, V] in place with -inf outside each row's top-p set (the package's contract);
    returns ``logits``. ``buffer``: the > 64-row workspace of a package that takes one."""
    host = _shape_host(logits.shape[-1])
    if host is None:
        raise ValueError(_no_kernel(logits.shape[-1]))
    if buffer is None:
        return host.top_p_mask(logits, p)
    return host.top_p_mask(logits, p, buffer)


def top_p_mask_unsupported_reason(logits: torch.Tensor, p: torch.Tensor,
                                  buffer: torch.Tensor | None = None) -> str | None:
    """None when ``top_p_mask`` serves this call, else why not (host checks only)."""
    if not available():
        return "top_p_mask: needs gfx950 with FlyDSL"
    vocab = logits.shape[-1] if logits.dim() else 0
    host = _shape_host(vocab)
    if host is None:
        return _no_kernel(vocab)
    blocker = host.top_p_blocker(vocab, float("-inf"))
    if blocker is not None:
        return f"top_p_mask: {blocker}"
    if logits.dtype != torch.float32 or logits.dim() != 2 or logits.stride(1) != 1:
        return f"top_p_mask: logits {logits.dtype} {tuple(logits.shape)} (fp32 [rows, {vocab}], unit column stride)"
    n = logits.shape[0]
    if p.dtype != torch.float32 or tuple(p.shape) != (n,) or not p.is_contiguous():
        return f"top_p_mask: p {p.dtype} {tuple(p.shape)} (fp32 [{n}], contiguous)"
    off = _off_device(logits=logits, p=p)
    if off is not None:
        return off
    rows_kernel = hasattr(host, "rows_grid")
    if n < 1 or (not rows_kernel and n > host.SPLIT_MAX_BATCH):
        takes = ">= 1" if rows_kernel else f"1..{host.SPLIT_MAX_BATCH}"
        return f"top_p_mask: {n} rows (the kernel takes {takes})"
    if buffer is None:
        return None
    if not rows_kernel:
        return f"top_p_mask: buffer (the vocab {vocab} kernel takes none)"
    if n > host.SPLIT_MAX_BATCH:
        grid = host.rows_grid(n, logits.device.index)
        if (buffer.dtype != torch.float32 or buffer.dim() != 2 or buffer.shape[0] < grid
                or buffer.shape[1] != vocab or not buffer.is_contiguous()):
            return (f"top_p_mask: buffer {buffer.dtype} {tuple(buffer.shape)} (fp32 [>= {grid}, {vocab}], "
                    "contiguous)")
        return _off_device(buffer=buffer)
    return None
