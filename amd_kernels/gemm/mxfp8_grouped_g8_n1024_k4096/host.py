# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Support check, workspace, row -> config dispatch and entries of the grouped MXFP8 GEMM."""

import functools

import torch

from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from .kernel import BLOCK, SPLIT_COUNTERS, Config, G, K, N, check_config, compile_grouped_mxfp8

MAX_M = 1024

# (largest M of the window, config): the fastest of 920 swept configs (tile, waves, stages, split K, B path, XCD
# order; benchmarks/gemm/mxfp8_grouped_g8_n1024_k4096/benchmark_config_sweep.py), windows grown while a config stays
# within 2 % of the best at every measured M. The MXFP8-out row is the same config with the epilogue (no measurable
# time).
_ROWS: tuple[tuple[int, Config], ...] = (
    (32, Config(32, 32, 256, 1, 2, 5, b_nt=True, xcd_order=True, scale_preload=False)),
    (64, Config(64, 32, 128, 4, 1, 5, b_nt=True, xcd_order=True)),
    (96, Config(48, 64, 256, 1, 2, 5, b_direct=True, b_ahead=2, b_nt=True, xcd_order=True)),
    (112, Config(64, 64, 256, 2, 4, 4, xcd_order=True)),
    (128, Config(64, 64, 256, 2, 2, 3, b_nt=True, xcd_order=True)),
    (192, Config(96, 64, 128, 2, 2, 4, xcd_order=True)),
    (256, Config(64, 128, 256, 1, 4, 3, b_direct=True, b_ahead=2, b_nt=True, xcd_order=True)),
    (384, Config(96, 128, 128, 1, 4, 4, b_direct=True, b_ahead=3, b_nt=True, xcd_order=True)),
    (448, Config(128, 128, 128, 1, 4, 4, b_direct=True, b_ahead=2, b_nt=True, xcd_order=True)),
    (480, Config(128, 128, 128, 1, 4, 4, b_direct=True, b_ahead=2, b_nt=True, xcd_order=True, xcd_swizzle=8)),
    (512, Config(128, 128, 128, 1, 4, 4, b_direct=True, b_ahead=2, b_nt=True, xcd_order=True)),
    (672, Config(96, 128, 128, 1, 2, 4, b_direct=True, b_ahead=2, xcd_order=True)),
    (768, Config(96, 128, 128, 1, 2, 3, b_direct=True, b_ahead=2, xcd_order=True, xcd_swizzle=2)),
    (MAX_M, Config(128, 256, 128, 1, 4, 3, b_direct=True, b_ahead=2, xcd_order=True)),
)


def select(m: int, mxfp8_out: bool = False) -> Config:
    """The config of a row count."""
    assert 1 <= m <= MAX_M, m
    config = next(c for bound, c in _ROWS if m <= bound)
    return config._replace(mxfp8_out=mxfp8_out)


@functools.cache
def _kernel_exe(config: Config):
    return compile_grouped_mxfp8(config)


def _partial_floats(config: Config, m: int) -> int:
    tiles = -(-m // config.tile_m) * (N // config.tile_n) * G
    return tiles * config.splits * config.tile_m * config.tile_n if config.splits > 1 else 0


_WORKSPACES: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-K partials and arrival counters, one set per device (launches on one stream); the counters are
    zeroed once, every launch leaves them at zero. Allocated before graph capture (``warmup_grouped_gemm``)."""
    if device not in _WORKSPACES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("call warmup_grouped_gemm before CUDA-graph capture")
        floats = max([_partial_floats(c, m) for m in range(1, MAX_M + 1) for c in (select(m),)] + [1])
        _WORKSPACES[device] = (torch.empty(floats, dtype=torch.float32, device=device),
                               torch.zeros(SPLIT_COUNTERS, dtype=torch.int32, device=device))
    return _WORKSPACES[device]


def _check(x, x_scale, weight, weight_scale):
    m = x.shape[0]
    assert 1 <= m <= MAX_M, f"rows {m} outside 1..{MAX_M}"
    assert x.dtype == torch.float8_e4m3fn and x.is_contiguous() and x.numel() == m * G * K, (x.dtype, x.shape)
    assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous() and weight.numel() == G * N * K
    for s, n in ((x_scale, m * G * K // BLOCK), (weight_scale, G * N * K // BLOCK // BLOCK)):
        assert s.dtype in (torch.uint8, torch.float8_e8m0fnu) and s.is_contiguous(), s.dtype
        assert s.numel() == n, (s.numel(), n)
    assert x.device == x_scale.device == weight.device == weight_scale.device


def _launch(config, x, x_scale, weight, weight_scale, out, codes, scales):
    m = x.shape[0]
    partials, counters = _workspace(x.device)
    assert partials.numel() >= _partial_floats(config, m)
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    _run_compiled(
        _kernel_exe(config), out.data_ptr(), codes.data_ptr(), scales.data_ptr(), x.data_ptr(), weight.data_ptr(),
        u8(x_scale).data_ptr(), u8(weight_scale).data_ptr(), partials.data_ptr(), counters.data_ptr(), m,
        torch.cuda.current_stream(),
    )


def _grouped_gemm(x, x_scale, weight, weight_scale, out=None):
    _check(x, x_scale, weight, weight_scale)
    m = x.shape[0]
    if out is None:
        out = torch.empty(m, G * N, dtype=torch.bfloat16, device=x.device)
    assert out.dtype == torch.bfloat16 and out.is_contiguous() and out.shape == (m, G * N) and out.device == x.device
    assert out.data_ptr() != x.data_ptr()
    _launch(select(m), x, x_scale, weight, weight_scale, out, out, out)
    return out


def _grouped_gemm_mxfp8(x, x_scale, weight, weight_scale):
    _check(x, x_scale, weight, weight_scale)
    m = x.shape[0]
    codes = torch.empty(m, G * N, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(m, G * N // BLOCK, dtype=torch.uint8, device=x.device)
    _launch(select(m, True), x, x_scale, weight, weight_scale, scales, codes, scales)
    return codes, scales


def _mode(*args, **kwargs):
    return ()


def _sizes(mode, mxfp8_out=False):
    """The first row count of each config window."""
    first, prev = [], 0
    for bound, _ in _ROWS:
        first.append(prev + 1)
        prev = bound
    return tuple(first)


def _inputs(mode, n, x, x_scale, weight, weight_scale, **kwargs):
    dev = x.device
    return (torch.zeros(n, G * K, dtype=torch.float8_e4m3fn, device=dev),
            torch.zeros(n, G * K // BLOCK, dtype=torch.uint8, device=dev), weight, weight_scale), {}


grouped_gemm = entry("gemm/mxfp8_grouped_g8_n1024_k4096", mode=_mode, sizes=_sizes, inputs=_inputs)(_grouped_gemm)
grouped_gemm.__doc__ = """``out[m, g, :] = x[m, g, :] @ W[g]^T`` for 8 groups (``reference.py``): ``x`` E4M3 codes
[M, 8 * 4096] with row-major E8M0 scales [M, 8 * 128] (uint8 or float8_e8m0fnu), ``weight`` the (16, 16)-preshuffled
codes [8 * 1024, 4096] (aiter ``shuffle_weight``) with 32 x 32 E8M0 scales [8, 32, 128]; 1 <= M <= ``MAX_M``. Returns
bf16 [M, 8 * 1024], or fills the caller's contiguous ``out``. One workspace per device: launches on one stream (or
graphs replayed one after another). CUDA-graph safe after ``warmup_grouped_gemm``."""

grouped_gemm_mxfp8 = entry("gemm/mxfp8_grouped_g8_n1024_k4096/mxfp8", mode=_mode, sizes=_sizes, inputs=_inputs)(
    _grouped_gemm_mxfp8
)
grouped_gemm_mxfp8.__doc__ = """``grouped_gemm`` with the MXFP8 epilogue: returns E4M3 codes [M, 8 * 1024] and
row-major E8M0 scales [M, 8 * 1024 // 32] (uint8) of the bf16 output (``reference.py``), the pre-quantized activation
of the next MXFP8 GEMM. Same inputs, workspace and graph rules as ``grouped_gemm``."""


def warmup_grouped_gemm(device: torch.device) -> None:
    """Compile every config of both entries and allocate the workspace, before graph capture; nothing compiles
    after it."""
    x = torch.zeros(1, G * K, dtype=torch.float8_e4m3fn, device=device)
    xs = torch.zeros(1, G * K // BLOCK, dtype=torch.uint8, device=device)
    w = torch.zeros(G * N, K, dtype=torch.float8_e4m3fn, device=device)
    ws = torch.zeros(G, N // BLOCK, K // BLOCK, dtype=torch.uint8, device=device)
    _workspace(x.device)
    for m in _sizes(()):
        args, _ = _inputs((), m, x, xs, w, ws)
        grouped_gemm(*args)
        grouped_gemm_mxfp8(*args)
    torch.cuda.synchronize(device)
