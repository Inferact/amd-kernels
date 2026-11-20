# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the SwiGLU-OAI + MXFP8 quant kernel."""

import functools
import struct

import torch

from . import kernel as _kernel

I = _kernel.I

# (max rows, pieces, threads): 12 one-wave pieces per row up to 704 rows, then one 12-wave workgroup per row
# (benchmark_swiglu_oai_mxfp8.py --sweep, both forms)
SHAPES = ((704, 12, 64), (1 << 31, 1, 768))


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(form: str, shuffled: bool, pieces: int, threads: int):
    return _kernel.compile_swiglu_oai_mxfp8(form, shuffled, pieces, threads)


def _shape(m: int) -> tuple[int, int]:
    return next((p, t) for top, p, t in SHAPES if m <= top)


def _f32_bits(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


def swiglu_oai_mxfp8(
    gate_up: torch.Tensor,  # [m, 2 * 6144] bf16, unit column stride (split layout: gate | up)
    alpha: float,
    beta: float,
    limit: float,
    code_tile_rows: int = 0,
    form: str = "bf16_chain",
    q_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes, scales) of ``quantize(swiglu(gate_up))`` for the down GEMM: codes row-major [m, I]
    (``code_tile_rows`` 0) or in the decode GEMM's (16, 16) layout, rows padded to the tile (padding rows
    unwritten); e8m0 scales [ceil(rows / 32) * 32, I / 32] in AITER's shuffled layout (padding unwritten).
    ``form``: the SwiGLU numerics, the caller's (``bf16_chain`` = the draft's aten chain, ``fp32`` = the
    target's ``swiglu_oai_split``). ``shape`` (pieces, threads) overrides the row-count choice (sweeps only)."""
    m = gate_up.shape[0]
    assert gate_up.dtype == torch.bfloat16 and gate_up.shape[1] == 2 * I and gate_up.stride(1) == 1
    shuffled = code_tile_rows > 0
    if shuffled:
        assert code_tile_rows % 16 == 0
        rows = (m + code_tile_rows - 1) // code_tile_rows * code_tile_rows
    else:
        rows = m
    q = q_out if q_out is not None else torch.empty((rows, I), dtype=torch.float8_e4m3fn, device=gate_up.device)
    s = (
        scale_out
        if scale_out is not None
        else torch.empty(((rows + 31) // 32 * 32, I // 32), dtype=torch.uint8, device=gate_up.device)
    )
    assert q.shape[0] >= rows and q.shape[1] == I and q.is_contiguous()
    assert s.shape[0] >= (rows + 31) // 32 * 32 and s.shape[1] == I // 32 and s.is_contiguous()
    if m:
        _run_compiled(
            _get_kernel(form, shuffled, *(shape or _shape(m))),
            gate_up.data_ptr(),
            q.data_ptr(),
            s.data_ptr(),
            q.numel() * q.element_size(),
            s.numel(),
            m,
            gate_up.stride(0),
            _f32_bits(alpha),
            _f32_bits(beta),
            _f32_bits(limit),
            torch.cuda.current_stream(),
        )
    return q, s


def warmup_swiglu_oai_mxfp8(device: torch.device | None = None, forms=("bf16_chain", "fp32")) -> None:
    """Compile every variant (each row-count shape x form x row-major / tiled codes; the row count is a runtime
    argument) before serving or graph capture (README rule 4)."""
    for top, _, _ in SHAPES:
        m = min(top, 2048)
        x = torch.zeros((m, 2 * I), dtype=torch.bfloat16, device=device or "cuda")
        for form in forms:
            for tile in (0, 16):
                swiglu_oai_mxfp8(x, 1.0, 0.0, 1.0, tile, form)
