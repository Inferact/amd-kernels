# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache and launches of the decode GEMM's row-major (``A_RM``), bf16 and swiglu-epilogue forms."""

import functools
from dataclasses import dataclass

import torch
from amd_kernels.common.tensor_shim import _run_compiled, ptr_arg
from amd_kernels.gemm.mxfp8_common.gemm import _dummy_i32, _reduce_launcher

from .kernel import CNT_STRIDE, compile_mxfp8_gemm_decode

CNT_TILES = 1 << 15  # tile counters of a workspace
SLAB_BYTES = 64 << 20  # the split-K partial slab of a workspace


@dataclass(frozen=True)
class Config:
    """One row of the kernel: 16 * ``rt`` rows and 16 * ``ni`` columns per workgroup, ``kw`` K-waves, ``ks`` K
    splits over workgroups, ``sched`` scheduling barriers, ``w_cm`` the weight cache modifier, ``fuse`` the split-K
    sum by the last workgroup of a tile (``SK_FUSE``) instead of a reduce launch."""

    rt: int
    ni: int
    kw: int
    ks: int = 1
    sched: int = 0
    w_cm: int | None = None
    fuse: int = 0


@functools.cache
def _compiled(n: int, k: int, cfg: Config, mode: str, limit: float | None):
    return compile_mxfp8_gemm_decode(
        N=n, K=k, RT=cfg.rt, NI=cfg.ni, KW=cfg.kw, KS=cfg.ks, sched_barriers=bool(cfg.sched),
        A_BF16=mode == "bf16", A_RM=mode == "rowmajor", EPI_LIMIT=limit, SK_FUSE=bool(cfg.fuse), W_CM=cfg.w_cm,
    )  # fmt: skip


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8).view(-1)


@functools.cache
def workspace(device: torch.device, key: str = "") -> tuple[torch.Tensor, torch.Tensor]:
    """The tile counters (zero, epoch counters: every launch leaves them at a multiple of their epoch) and the
    split-K slab of the fused split-K rows, one pair per (device, ``key``). Launches that share a pair must not
    overlap on a device. Created on first use: call a fused row once before CUDA graph capture."""
    cnt = torch.zeros(CNT_TILES * CNT_STRIDE, dtype=torch.int32, device=device)
    slab = torch.empty(SLAB_BYTES // 4, dtype=torch.int32, device=device)
    return cnt, slab


def _launch(cfg, mode, a, a_scale, w, ws, out, sc_out, limit, m, key=""):
    n, k = w.shape
    exe = _compiled(n, k, cfg, mode, limit)
    n_mb = -(-m // (16 * cfg.rt))
    assert mode in ("bf16", "rowmajor", "shuffled"), mode
    grid = cfg.ks * n_mb * exe.n_blocks
    dummy = _dummy_i32(a.device)
    a_scale = dummy if a_scale is None else a_scale
    sc_out = dummy if sc_out is None else sc_out
    stream = torch.cuda.current_stream()
    target = out
    cnt = slab = dummy
    if cfg.fuse:
        assert cfg.ks > 1 and cfg.ks * m * n * 4 <= SLAB_BYTES and n_mb * exe.n_blocks <= CNT_TILES
        cnt, slab = workspace(a.device, key)
    elif cfg.ks > 1:
        target = torch.empty((cfg.ks, m, n), dtype=torch.float32, device=a.device)
    _run_compiled(
        exe, _u8(a), _u8(w), _u8(target), _u8(a_scale), _u8(ws), _u8(sc_out), _u8(cnt), _u8(slab), m,
        a_scale.numel() * a_scale.element_size(), n_mb, grid, stream,
    )  # fmt: skip
    if cfg.ks > 1 and not cfg.fuse:
        _run_compiled(
            _reduce_launcher((cfg.ks, "bf16")), ptr_arg(target), ptr_arg(out), (m * n) // 2, m * n, stream, cfg.ks,
            "bf16",
        )  # fmt: skip


def gemm(
    a, a_scale, weight, weight_scale, cfg: Config, *, layout: str = "rowmajor", rows: int | None = None,
    out: torch.Tensor | None = None, key: str = "",
) -> torch.Tensor:
    """``A @ W^T`` into bf16 [M, N]. ``a``: pre-quantized E4M3 codes [M, K] with row-major E8M0 scales [M, K / 32]
    (uint8), or (``a_scale`` None) a bf16 [M, K] activation quantized in registers; ``layout="shuffled"``: codes
    and scales in the weight's (16, 16) / AITER-shuffled layouts (the common quantizer's, rows padded to
    ``16 * cfg.rt``; ``rows``: the row count M then). ``weight`` / ``weight_scale``: the (16, 16) preshuffled codes
    and AITER-shuffled per-row scales. ``key``: names the ``workspace`` of the fused split-K rows."""
    m = a.shape[0] if rows is None else rows
    n = weight.shape[0]
    bf16_in = a_scale is None
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    if m:
        _launch(cfg, "bf16" if bf16_in else layout, a, a_scale, weight, weight_scale, out, None, None, m, key)
    return out


def gemm_swiglu_mxfp8(
    a, a_scale, weight, weight_scale, cfg: Config, limit: float = 10.0,
    *, layout: str = "rowmajor", rows: int | None = None, codes: torch.Tensor | None = None,
    scale: torch.Tensor | None = None, key: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate_up GEMM (``weight`` rows [gate; up], N = 2 I) with the clamped SiLU-mul and the MXFP8 quantization of
    its result in the epilogue: E4M3 codes [M, I] and E8M0 bytes [M, I / 32], row-major. ``a`` as in ``gemm``;
    ``layout`` as there; ``cfg.ni`` is a multiple of 4; a split-K row needs ``cfg.fuse``."""
    m = a.shape[0] if rows is None else rows
    i = weight.shape[0] // 2
    if codes is None:
        codes = torch.empty((m, i), dtype=torch.float8_e4m3fn, device=a.device)
    if scale is None:
        scale = torch.empty((m, i // 32), dtype=torch.uint8, device=a.device)
    if m:
        _launch(cfg, "bf16" if a_scale is None else layout, a, a_scale, weight, weight_scale, codes, scale, limit, m, key)
    return codes, scale
