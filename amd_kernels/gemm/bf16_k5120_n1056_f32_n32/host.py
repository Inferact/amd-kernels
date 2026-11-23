# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the fused fp32 / bf16 GEMM: the load-time weight layout, the tile by M, the launch, the warm-up."""

import functools

import torch

from amd_kernels.warm_up import entry

from . import kernel as _kernel

K = _kernel.K
MAX_M = 1024
MAX_PART_BYTES = 8 << 20  # K split partials: at most 6.6 MB over the tile table (splits x tile groups x tile bytes)
MAX_TICKETS = 512  # tile groups (at most 224 over the tile table)
# (rows up to, 16-column tiles, 16-row tiles, K steps in flight, waves, K splits) of a workgroup, by the fp32 columns; the best of
# a sweep per row count (the last number: the K split over workgroups)
TILES = {
    1024: ((16, 2, 1, 2, 8, 5), (48, 4, 1, 2, 8, 4), (96, 4, 2, 4, 8, 5), (192, 4, 2, 4, 8, 5), (384, 5, 3, 2, 8, 2),
           (768, 5, 3, 2, 8, 1), (MAX_M, 6, 3, 1, 8, 1)),
    512: ((16, 1, 1, 4, 4, 4), (48, 2, 1, 2, 8, 4), (96, 2, 2, 4, 8, 5), (192, 4, 1, 2, 4, 2), (384, 4, 2, 4, 8, 2),
          (768, 5, 3, 2, 4, 4), (MAX_M, 5, 2, 1, 4, 1)),
    0: ((16, 1, 1, 4, 4, 10), (96, 2, 1, 2, 4, 10), (192, 1, 1, 4, 4, 5), (384, 1, 1, 4, 4, 4), (768, 2, 1, 4, 8, 5),
        (MAX_M, 2, 1, 2, 2, 4)),
}


@functools.cache
def _on_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def supports(x: torch.Tensor, w: torch.Tensor, na: int) -> bool:
    """gfx950, bf16 ``x [M <= 1024, 5120]`` and ``w`` in ``shuffle_weight``'s layout, ``na`` one of 1024 / 512 / 0 and ``w`` of ``na + 32`` rows."""
    return (
        _on_gfx950()
        and x.dtype == w.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] == K
        and x.shape[0] <= MAX_M
        and x.is_contiguous()
        and w.dim() == 2
        and w.shape[1] == K
        and w.shape[0] % 16 == 0
        and na % 16 == 0
        and na in TILES
        and w.shape[0] == na + 32
    )


def shuffle_weight(wa: torch.Tensor | None, wb: torch.Tensor) -> torch.Tensor:
    """The load-time layout of ``[wa; wb]`` (``[NA + NB, 5120]`` bf16; ``wa`` may be None): ``[column tile, K step, lane,
    8]`` with lane = 16 g + l and element ``w[16 tile + l, 32 step + 8 g + e]``, so a wave's fragment load is 1 KB in a
    row. Bit-exact (a permutation); the caller replaces the weights once, before graph capture."""
    w = wb if wa is None else torch.cat([wa, wb])
    n = w.shape[0]
    assert w.dtype == torch.bfloat16 and w.shape[1] == K and n % 16 == 0
    return w.reshape(n // 16, 16, K // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().reshape(n, K)


def _tiles(m: int, na: int) -> tuple[int, int, int, int, int]:
    for limit, nt, mt, ch, nw, *ks in TILES[na]:
        if m <= limit:
            return nt, mt, ch, nw, (ks[0] if ks else 1)
    raise ValueError(f"{m} rows (at most {MAX_M})")


@functools.cache
def _get_kernel(na: int, nb: int, nt: int, mt: int, ch: int, nw: int, ksp: int):
    return _kernel.compile_gemm(na, nb, nt, mt, ch, nw, ksp)


@functools.cache
def _split_buffers(device: torch.device):
    """The K-split partials and zero tickets (graph capture records no allocation; the last workgroup of a tile resets
    its ticket)."""
    return torch.empty(MAX_PART_BYTES, dtype=torch.uint8, device=device), torch.zeros(
        MAX_TICKETS, dtype=torch.int32, device=device
    )


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _gemm(
    x: torch.Tensor,  # [M, 5120] bf16
    w: torch.Tensor,  # [NA + NB, 5120] bf16, shuffle_weight's layout
    out_a: torch.Tensor | None,  # [M, NA] fp32 (None when NA = 0)
    out_b: torch.Tensor,  # [M, NB] bf16
) -> None:
    """``out_a, out_b = x @ w^T``: the first NA columns in fp32, the rest in bf16."""
    m = x.shape[0]
    if m == 0:
        return
    na = 0 if out_a is None else out_a.shape[1]
    nb = out_b.shape[1]
    assert w.shape[0] == na + nb and x.is_contiguous() and out_b.is_contiguous()
    assert out_a is None or (out_a.dtype == torch.float32 and out_a.is_contiguous() and out_a.shape[0] == m)
    nt, mt, ch, nw, ksp = _tiles(m, na)
    groups = -(-(na + nb) // 16 // nt) * -(-m // (16 * mt))
    assert groups <= MAX_TICKETS and ksp * groups * nt * mt * 1024 <= MAX_PART_BYTES
    part, cnt = _split_buffers(x.device)
    _run_compiled(
        _get_kernel(na, nb, nt, mt, ch, nw, ksp),
        x.data_ptr(),
        w.data_ptr(),
        out_a.data_ptr() if out_a is not None else out_b.data_ptr(),
        out_b.data_ptr(),
        part.data_ptr(),
        cnt.data_ptr(),
        m,
        groups * ksp,
        torch.cuda.current_stream(x.device),
    )


def _mode(x, w, out_a, out_b):
    """What a call fixes about its kernels (``amd_kernels.warm_up``): the column split."""
    return (0 if out_a is None else out_a.shape[1], out_b.shape[1])


def _sizes(mode):
    """One row count per M tile."""
    return tuple(t[0] for t in TILES[mode[0]])


def _inputs(mode, m, x, w, out_a, out_b):
    dev = x.device
    na, nb = mode
    return (
        torch.zeros((m, K), dtype=x.dtype, device=dev),
        w,
        None if na == 0 else torch.empty((m, na), dtype=torch.float32, device=dev),
        torch.empty((m, nb), dtype=torch.bfloat16, device=dev),
    ), {}


gemm = entry("gemm/bf16_k5120_n1056_f32_n32", mode=_mode, sizes=_sizes, inputs=_inputs)(_gemm)
