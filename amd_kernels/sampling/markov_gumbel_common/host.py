# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shape-agnostic fused Markov draft chain: reads (vocab, rank) from ``w1`` and calls that shape's package
(``_SHAPE_HOSTS``, imported on its first call)."""

import functools
import importlib

import torch

_SHAPE_HOSTS = {
    (163840, 512): "amd_kernels.sampling.markov_gumbel_fused_v163840_r512",
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
def _shape_host(vocab: int, rank: int):
    name = _SHAPE_HOSTS.get((vocab, rank))
    return importlib.import_module(name) if name is not None else None


def _package(w1: torch.Tensor):
    pkg = _shape_host(*w1.shape) if w1.dim() == 2 else None
    if pkg is None:
        raise ValueError(_no_kernel(w1))
    return pkg


def _no_kernel(w1: torch.Tensor) -> str:
    return (f"markov_gumbel_fused: no kernel for (vocab, rank)={tuple(w1.shape)} "
            f"(covered: {sorted(_SHAPE_HOSTS)})")


def markov_gumbel_fused_shapes() -> frozenset[tuple[int, int]]:
    """The (vocab, Markov rank) keys with a package."""
    return frozenset(_SHAPE_HOSTS)


def markov_max_rows(w1: torch.Tensor) -> int:
    """The most requests one call of this shape takes."""
    return _package(w1).MAX_ROWS


def markov_scratch_dwords(w1: torch.Tensor, n: int) -> int:
    """int32 words of the scratch a call of ``n`` requests needs (zeroed once, then owned by the kernel)."""
    return _package(w1).scratch_dwords(n)


def markov_gumbel_fused(base, w1, w2, input_ids, anchor_idx, draft_tokens, idx_mapping, sample_pos, temperature,
                        seeds, cache, scratch, row_mapping=None) -> None:
    """All k Markov draft steps over ``base`` [n, k, V] into ``draft_tokens`` and ``cache`` (the package's
    contract), one persistent launch."""
    _package(w1).markov_gumbel_fused(base, w1, w2, input_ids, anchor_idx, draft_tokens, idx_mapping, sample_pos,
                                  temperature, seeds, cache, scratch, row_mapping)


def _off_device(**tensors: torch.Tensor | None) -> str | None:
    """Why the tensors are not all on the current CUDA device (the launch's), else None."""
    index = torch.cuda.current_device()
    for name, t in tensors.items():
        if t is not None and t.get_device() != index:
            return f"markov_gumbel_fused: {name} on {t.device} (the current CUDA device)"
    return None


def _int64_nk(t: torch.Tensor, n: int, k: int) -> bool:
    return t.dtype == torch.int64 and t.dim() == 2 and t.shape[0] >= n and t.shape[1] >= k and t.stride(1) == 1


def markov_gumbel_fused_unsupported_reason(base, w1, w2, input_ids, anchor_idx, draft_tokens, idx_mapping,
                                           sample_pos, temperature, seeds, cache, scratch,
                                           row_mapping=None) -> str | None:
    """None when ``markov_gumbel_fused`` serves this call, else why not (host checks only)."""
    op = "markov_gumbel_fused"
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    pkg = _shape_host(*w1.shape) if w1.dim() == 2 else None
    if pkg is None:
        return _no_kernel(w1)
    vocab, rank = w1.shape
    for name, w in (("w1", w1), ("w2", w2)):
        if w.dtype != torch.bfloat16 or tuple(w.shape) != (vocab, rank) or not w.is_contiguous():
            return f"{op}: {name} {w.dtype} {tuple(w.shape)} (bf16 [{vocab}, {rank}], contiguous)"
    if (base.dtype != torch.bfloat16 or base.dim() != 3 or base.shape[2] != vocab or base.stride(2) != 1
            or base.stride(1) != vocab):
        return (f"{op}: base {base.dtype} {tuple(base.shape)} stride {base.stride()} "
                f"(bf16 [n, k, {vocab}], unit column stride, step stride {vocab})")
    n, k = base.shape[0], base.shape[1]
    if not 1 <= n <= pkg.MAX_ROWS:
        return f"{op}: {n} rows (the kernel takes 1..{pkg.MAX_ROWS})"
    if not 1 <= k <= pkg.host.MAX_STEPS:
        return f"{op}: {k} steps (the kernel takes 1..{pkg.host.MAX_STEPS})"
    if input_ids.dtype != torch.int32 or input_ids.dim() != 1:
        return f"{op}: input_ids {input_ids.dtype} {tuple(input_ids.shape)} (int32 [tokens])"
    if anchor_idx.dtype != torch.int64 or anchor_idx.dim() != 1 or anchor_idx.shape[0] < n:
        return f"{op}: anchor_idx {anchor_idx.dtype} {tuple(anchor_idx.shape)} (int64 [>= {n}])"
    for name, t in (("draft_tokens", draft_tokens), ("idx_mapping", idx_mapping), ("sample_pos", sample_pos)):
        if not _int64_nk(t, n, k):
            return f"{op}: {name} {t.dtype} {tuple(t.shape)} (int64 [>= {n}, >= {k}], unit column stride)"
    if temperature.dtype != torch.float32 or seeds.dtype != torch.int64:
        return f"{op}: temperature {temperature.dtype}, seeds {seeds.dtype} (fp32, int64)"
    if (cache.dtype != torch.bfloat16 or cache.dim() != 3 or cache.shape[1] < k or cache.shape[2] != vocab
            or cache.stride(2) != 1):
        return f"{op}: cache {cache.dtype} {tuple(cache.shape)} (bf16 [rows, >= {k}, {vocab}], unit column stride)"
    if cache.numel() * 2 >= 1 << 32 or base.numel() * 2 >= 1 << 32:
        return f"{op}: cache or base of 4 GiB or more (32-bit offsets)"
    if row_mapping is not None and row_mapping.dtype != torch.int32:
        return f"{op}: row_mapping {row_mapping.dtype} (int32)"
    if scratch.dtype != torch.int32 or scratch.numel() < pkg.scratch_dwords(n):
        return f"{op}: scratch {scratch.dtype} of {scratch.numel()} (int32, >= {pkg.scratch_dwords(n)} words)"
    return _off_device(base=base, w1=w1, w2=w2, input_ids=input_ids, anchor_idx=anchor_idx, draft_tokens=draft_tokens,
                       idx_mapping=idx_mapping, sample_pos=sample_pos, temperature=temperature, seeds=seeds,
                       cache=cache, scratch=scratch, row_mapping=row_mapping)
