# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the skinny bf16 linear: the load-time weight layout, the tile by (N, K, M), the workspace, the launch."""
import functools

import torch

from amd_kernels.warm_up import entry

from .config import CONFIGS, SHAPES, STOCK_FASTER
from .kernel import compile_bf16_skinny, slab_floats

MT_MAX = 12
MAX_ROWS = 16 * MT_MAX
_WORKSPACE: dict = {}


def supports(n: int, k: int) -> bool:
    """Whether a [n, k] weight has a tuned tile table (every entry of ``SHAPES``)."""
    return (n, k) in SHAPES


def prefers_stock(n: int, k: int, m: int) -> bool:
    """Whether the stock op was measured as fast as this kernel at ``m`` rows (the next measured row count up; above
    192 rows no row count was tuned)."""
    rows = STOCK_FASTER.get((n, k))
    above = [r for r in rows if r >= m] if rows else []
    return not above or rows[min(above)]


def pack_weight(w: torch.Tensor) -> torch.Tensor:
    """The load-time layout of a [N, K] bf16 weight, as [N, K]: [N / 16 row groups, K / 32, 4 lane groups, 16 rows, 8 K].

    A permutation: every 16-byte lane load of the kernel reads 1 KB in one piece of a row group's K stream. The caller
    replaces the parameter with it once, before graph capture, and frees the row-major tensor.
    """
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 32 == 0
    return w.view(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(n, k)


def unpack_weight(wp: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """The inverse of ``pack_weight`` (into ``out`` when given: a shared scratch for readers that need row-major)."""
    n, k = wp.shape
    src = wp.view(n // 16, k // 32, 4, 16, 8).permute(0, 3, 1, 2, 4)
    if out is None:
        return src.contiguous().view(n, k)
    out.view(n // 16, 16, k // 32, 4, 8).copy_(src)
    return out


def m_tier(m: int) -> int:
    """16-row fragments of the tile that serves ``m`` rows (above 192 rows: 12-fragment tiles, one grid row per tile)."""
    return min(MT_MAX, (m + 15) // 16)


def config_for(n: int, k: int, mt: int) -> tuple:
    """(wn, wk, ng, s, depth, ring) of the tile: the tuned entry of the smallest tuned tier >= mt."""
    table = CONFIGS[(n, k)]
    return table[min(t for t in table if t >= mt)]


def _geometry(n: int, k: int, mt: int, cfg: tuple, mtiles: int):
    wn, wk, ng, s, _, _ = cfg
    return slab_floats(n, mt, wn, wk, ng, s, mtiles), mtiles * -(-(n // 16) // (wn * ng))


def _workspace(dev: torch.device, n: int, k: int, mt: int, cfg: tuple, mtiles: int):
    slab_floats, counters = _geometry(n, k, mt, cfg, mtiles)
    key = (dev, n, k, mt, mtiles)
    ws = _WORKSPACE.get(key)
    if ws is None:
        ws = (torch.empty(max(slab_floats, 1), dtype=torch.float32, device=dev),
              torch.zeros(counters * 32, dtype=torch.int32, device=dev))
        _WORKSPACE[key] = ws
    return ws


@functools.cache
def _kernel(n: int, k: int, mt: int, cfg: tuple):
    return compile_bf16_skinny(n, k, mt, *cfg)


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _bf16_linear(x: torch.Tensor, wp: torch.Tensor, out: torch.Tensor | None = None, *,
                 workspace: tuple | None = None) -> torch.Tensor:
    """out [M, N] bf16 = x [M, K] bf16 @ W^T, fp32 accumulate, ``wp`` = ``pack_weight(W)``.

    Deterministic (K split across workgroups is summed in split order by the last workgroup of the tile; no output
    atomics). CUDA-graph safe once the first call of the shape has run outside capture (the entry's warm-up does it).
    Concurrent calls of one shape on two streams need their own ``workspace`` = ``new_workspace(...)``.
    """
    m, k = x.shape
    n = wp.shape[0]
    assert wp.shape == (n, k) and wp.dtype == x.dtype == torch.bfloat16 and wp.is_contiguous()
    assert x.is_contiguous() and x.device == wp.device and (n, k) in SHAPES, (n, k)
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16 and out.is_contiguous()
    if m == 0:
        return out
    mt = m_tier(m)
    cfg = config_for(n, k, mt)
    mtiles = -(-m // (16 * mt))
    slab, cnt = workspace if workspace is not None else _workspace(x.device, n, k, mt, cfg, mtiles)
    exe = _kernel(n, k, mt, cfg)
    _run(exe, x.data_ptr(), wp.data_ptr(), out.data_ptr(), slab.data_ptr(), cnt.data_ptr(), m, mtiles,
         torch.cuda.current_stream())
    return out


def new_workspace(dev: torch.device, n: int, k: int, m: int) -> tuple:
    """A private (slab, counters) pair for calls of (n, k) with up to ``m`` rows."""
    mt = m_tier(m)
    cfg = config_for(n, k, mt)
    slab_floats, counters = _geometry(n, k, mt, cfg, -(-m // (16 * mt)))
    return torch.empty(slab_floats, dtype=torch.float32, device=dev), torch.zeros(counters * 32, dtype=torch.int32, device=dev)


def _mode(x, wp, out=None, *, workspace=None):
    return tuple(wp.shape)


def _inputs(mode, n_rows, x, wp, out=None, *, workspace=None):
    return (torch.zeros(n_rows, mode[1], dtype=torch.bfloat16, device=x.device), wp), {}


# one row count per tile tier, and one above the largest tile (a second grid row of M tiles)
WARMUP_ROWS = [16 * mt for mt in range(1, MT_MAX + 1)] + [MAX_ROWS + 1]

bf16_linear = entry("gemm/bf16_skinny_common", mode=_mode, sizes=lambda mode: WARMUP_ROWS, inputs=_inputs)(_bf16_linear)


def warmup_bf16_linear(wp: torch.Tensor) -> None:
    """Compile every tile variant of a weight shape and allocate its workspaces, before graph capture."""
    x = torch.zeros(WARMUP_ROWS[-1], wp.shape[1], dtype=torch.bfloat16, device=wp.device)
    for m in WARMUP_ROWS:
        bf16_linear(x[:m], wp)
    torch.cuda.synchronize(wp.device)
