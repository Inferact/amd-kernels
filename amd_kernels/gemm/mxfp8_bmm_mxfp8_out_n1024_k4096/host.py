# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Tile selection and launch of the batched MXFP8 GEMM with MXFP8 output."""

import functools
import re

import torch

from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from . import kernel as _kernel

N, K = _kernel.N, _kernel.K

# AITER's tuned FlyDSL rows for B 8 x 1024 x 4096, 32x32 w_scale (dsv4_batched_gemm_a8w8_blockscale_mxscale_
# bpreshuffle_tuned.csv, gfx950), merged into runs; M rounds up to the next run, M where AITER tuned OPUS takes
# the next FlyDSL row.
_TABLE = (
    (16, "t16x32x256_w1x2_nb5_sk1_nt_xcd_sps"),
    (64, "t64x32x512_w2x2_nb3_sk1_nt"),
    (128, "t64x64x256_w2x2_nb3_sk1_xcd"),
    (160, "t96x64x256_w2x2_nb3_sk1_xcd"),
    (192, "t96x64x256_w2x2_nb3_sk1_nt_xcd"),
    (256, "t128x64x128_w2x2_nb4_sk1_xcd"),
    (384, "t96x128x128_w2x2_nb4_sk1_nt_xcd"),
    (480, "t128x128x128_w1x4_nb4_sk1_bd3_xcd"),
    (512, "t128x128x128_w1x4_nb4_sk1_bd2_nt_xcd"),
    (576, "t96x128x128_w1x2_nb4_sk1_bd2_xcd"),
    (608, "t96x128x128_w1x2_nb3_sk1_bd2_xcd"),
    (640, "t96x128x128_w1x2_nb4_sk1_bd3_xcd"),
    (672, "t96x128x128_w1x2_nb3_sk1_bd2_xcd"),
    (768, "t96x128x128_w1x2_nb4_sk1_bd2_xcd"),
    (800, "t128x256x128_w1x4_nb3_sk1_bd2_nt_xcd"),
    (960, "t128x256x128_w1x4_nb3_sk1_bd2_xcd"),
    (992, "t128x256x128_w1x4_nb3_sk1_bd2_nt_xcd"),
    (1024, "t128x256x128_w1x4_nb3_sk1_bd2_xcd"),
    (1152, "t192x256x128_w1x4_nb3_sk1_bd2_xcd"),
    (1216, "t192x256x128_w1x4_nb3_sk1_bd2_nt_xcd"),
    (1344, "t192x256x128_w1x4_nb4_sk1_bd2_xcd"),
    (1408, "t192x256x128_w1x4_nb3_sk1_bd2_nt_xcd"),
    (1536, "t192x256x128_w1x4_nb3_sk1_bd2_xcd"),
    (1600, "t128x256x128_w1x4_nb3_sk1_bd2_xcd"),
    (1792, "t224x256x128_w1x4_nb2_sk1_bd1_xcd"),
    (None, "t256x256x128_w1x4_nb2_sk1_bd1_xcd"),
)

_NAME_RE = re.compile(
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_w(?P<m_warp>\d+)x(?P<n_warp>\d+)_nb(?P<num_buffers>\d+)_sk1"
    r"(?:_bd(?P<b_ahead>\d+))?(?P<b_nt>_nt)?(?P<xcd>_xcd(?P<xcd_swizzle>\d+)?)?(?P<sps>_sps)?"
)


@functools.cache
def parse_config(name: str) -> _kernel.Config:
    """The Config of an AITER kernelName suffix (``_kernel.config_name`` inverts it)."""
    g = _NAME_RE.fullmatch(name).groupdict()
    return _kernel.Config(
        *(int(g[f]) for f in ("tile_m", "tile_n", "tile_k", "m_warp", "n_warp", "num_buffers")),
        b_direct=g["b_ahead"] is not None,
        b_ahead=int(g["b_ahead"] or 1),
        b_nt=g["b_nt"] is not None,
        xcd_order=g["xcd"] is not None,
        xcd_swizzle=int(g["xcd_swizzle"] or 0),
        scale_preload=g["sps"] is None,
    )


def select(m: int) -> _kernel.Config:
    return parse_config(next(name for top, name in _TABLE if top is None or m <= top))


# One M per run, so a warm-up compiles every config.
WARM_UP_SIZES = tuple(top for top, _ in _TABLE if top is not None) + (2048,)


@functools.cache
def _launcher(batch: int, cfg: _kernel.Config):
    return _kernel.compile_bmm_mxfp8_out(batch, tuple(cfg))


def _mxfp8_bmm_mxfp8_out(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    q_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    cfg: _kernel.Config | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes, scales) of ``bf16(dequant(x[:, g]) @ dequant(w[g]).T)`` quantized per 32 columns, for every group g.

    Args:
        x: [M, G * 4096] (or [M, G, 4096]) e4m3, rows contiguous.
        x_scale: [M, G * 128] E8M0 bytes, row-major.
        w: [G, 1024, 4096] e4m3, ``shuffle_weight(., (16, 16))``.
        w_scale: [G, 32, 128] E8M0 bytes, 32x32 blocks.
        cfg: a tile config instead of ``select(M)`` (tests and tuning).

    Returns:
        codes [M, G * 1024] e4m3 and scales [M, G * 32] uint8, row-major.
    """
    m = x.shape[0]
    g = w.numel() // (N * K)
    assert x.dtype == torch.float8_e4m3fn and w.dtype == torch.float8_e4m3fn
    assert x.numel() == m * g * K and x.is_contiguous() and w.is_contiguous()
    assert x_scale.numel() == m * g * K // 32 and x_scale.is_contiguous() and x_scale.element_size() == 1
    assert w_scale.numel() == g * (N // 32) * (K // 32) and w_scale.is_contiguous() and w_scale.element_size() == 1
    assert m * g * K < 2**31, "A is addressed with 32-bit offsets"
    q = q_out if q_out is not None else torch.empty((m, g * N), dtype=torch.float8_e4m3fn, device=x.device)
    s = scale_out if scale_out is not None else torch.empty((m, g * N // 32), dtype=torch.uint8, device=x.device)
    assert q.shape == (m, g * N) and q.is_contiguous() and s.shape == (m, g * N // 32) and s.is_contiguous()
    if m:
        _run_compiled(
            _launcher(g, cfg or select(m)),
            q.data_ptr(),
            s.data_ptr(),
            x.data_ptr(),
            w.data_ptr(),
            x_scale.data_ptr(),
            w_scale.data_ptr(),
            m,
            torch.cuda.current_stream(),
        )
    return q, s


def _mode(x, x_scale, w, w_scale, q_out=None, scale_out=None, cfg=None):
    return (w.numel() // (N * K), x.device, cfg)


def _inputs(mode, n, x, x_scale, w, w_scale, q_out=None, scale_out=None, cfg=None):
    g, dev, cfg = mode
    xn = torch.zeros((n, g * K), dtype=torch.float8_e4m3fn, device=dev)
    sn = torch.zeros((n, g * K // 32), dtype=torch.uint8, device=dev)
    return (xn, sn, w, w_scale), {"cfg": cfg}


mxfp8_bmm_mxfp8_out = entry(
    "gemm/mxfp8_bmm_mxfp8_out_n1024_k4096",
    mode=_mode,
    sizes=lambda mode: WARM_UP_SIZES if mode[2] is None else (16,),
    inputs=_inputs,
)(_mxfp8_bmm_mxfp8_out)
