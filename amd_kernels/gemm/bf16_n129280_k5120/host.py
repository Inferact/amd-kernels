# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the lm_head GEMM: the load-time weight layout, the tile by M, the launch, the warm-up."""
import functools

import torch

from amd_kernels.warm_up import entry

from .kernel import compile_bf16_gemm_n129280_k5120
from .large import NB, compile_large_bf16_gemm_n129280_k5120
from .reference import K, N

MAX_M = 2048


def supports(w: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    """Whether ``w`` is the [129280, 5120] bf16 contiguous weight this GEMM (and ``shuffle_weight_``) takes; the GEMM
    has no bias."""
    return bias is None and tuple(w.shape) == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous()


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """The load-time layout [8080 groups, 80 K blocks, 16 rows, 64 K] of the [129280, 5120] weight, as [129280, 5120].

    Bit-exact (a permutation). The caller replaces the parameter with it once, before graph capture, and frees the
    row-major tensor (no second copy); every reader of the weight then calls bf16_gemm_n129280_k5120(shuffled=True).
    """
    assert w.shape == (N, K) and w.dtype == torch.bfloat16
    return w.view(N // 16, 16, K // 64, 64).permute(0, 2, 1, 3).contiguous().view(N, K)


def shuffle_weight_(w: torch.Tensor) -> torch.Tensor:
    """shuffle_weight in place: ``w`` keeps its storage and holds the load-time layout (the temporary is freed)."""
    ws = shuffle_weight(w)
    w.copy_(ws)
    del ws
    return w


def unshuffle_weight(ws: torch.Tensor) -> torch.Tensor:
    """The inverse of shuffle_weight (row-major [129280, 5120])."""
    assert ws.shape == (N, K) and ws.dtype == torch.bfloat16
    return ws.view(N // 16, K // 64, 16, 64).permute(0, 2, 1, 3).contiguous().view(N, K)


SMALL_MAX = 64  # rows up to which one four-wave workgroup covers all rows of its tile
ROT_MAX = 256  # rows up to which the vocabulary tiles start their K loops at different blocks (see large.py)
LARGE_MTS = (8, 6, 5, 7, 4, 3)  # eight-wave tiles of 32 mt rows, in the order a tie is broken (measured, see README)


def m_tile_fragments(m: int) -> int:
    """16-row fragments per wave. Up to 64 rows four waves take every row; above that, eight waves take two M halves
    and the M tile with the fewest padded rows wins."""
    if m <= SMALL_MAX:
        return (m + 15) // 16
    return min(LARGE_MTS, key=lambda mt: (-(-m // (32 * mt)) * 32 * mt, LARGE_MTS.index(mt)))


def k_starts(m: int) -> int:
    """Distinct first K blocks of an 8-wave launch's vocabulary tiles: NB while the activations fit L2 (rotation 80 costs
    9 % at 640-768 rows), else 1 (lockstep)."""
    return NB if m <= ROT_MAX else 1


def m_rows(m: int) -> int:
    """Rows of one M tile."""
    return 16 * m_tile_fragments(m) * (1 if m <= SMALL_MAX else 2)


@functools.cache
def _kernel(mt: int, shuf: bool, large: bool, starts: int = 0):
    if large:
        return compile_large_bf16_gemm_n129280_k5120(mt, int(shuf), starts)
    return compile_bf16_gemm_n129280_k5120(mt, int(shuf))


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _bf16_gemm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, *,
               shuffled: bool = False) -> torch.Tensor:
    """out [M, 129280] bf16 = x [M, 5120] bf16 @ W^T, fp32 accumulate; W = w (row-major) or shuffle_weight's layout.

    Any M (above 2048 rows: launches of 2048 rows each, the same bits as separate calls). Deterministic, no atomic
    output adds. CUDA-graph safe (no host sync, no allocation when out is given). The first call of each layout
    compiles its tile variants (``amd_kernels.warm_up``).
    """
    m = x.shape[0]
    assert x.ndim == 2 and x.shape[1] == K and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert w.shape == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous()
    if out is None:
        out = torch.empty(m, N, dtype=torch.bfloat16, device=x.device)
    assert out.shape == (m, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    if m > MAX_M:
        for i in range(0, m, MAX_M):
            _bf16_gemm(x[i:i + MAX_M], w, out[i:i + MAX_M], shuffled=shuffled)
        return out
    assert x.device == w.device == out.device
    if m:
        exe = _kernel(m_tile_fragments(m), bool(shuffled), m > SMALL_MAX, k_starts(m) if m > SMALL_MAX else 0)
        _run(exe, x.data_ptr(), w.data_ptr(), out.data_ptr(), m, -(-m // m_rows(m)), torch.cuda.current_stream())
    return out


def _mode(x, w, out=None, *, shuffled=False):
    """What a call fixes about its kernels besides M (``amd_kernels.warm_up``): the weight layout."""
    return bool(shuffled)


def _inputs(mode, n, x, w, out=None, *, shuffled=False):
    return (torch.zeros(n, K, dtype=torch.bfloat16, device=x.device), w), {"shuffled": shuffled}


def _warmup_rows() -> list[int]:
    """One row count per kernel variant (the M tile and, above SMALL_MAX rows, the K starts)."""
    first = {}
    for m in range(1, MAX_M + 1):
        first.setdefault((m_tile_fragments(m), k_starts(m) if m > SMALL_MAX else 0), m)
    return sorted(first.values())


# one row count per kernel variant; every other row count up to MAX_M reuses one of these
WARMUP_ROWS = _warmup_rows()

bf16_gemm_n129280_k5120 = entry("gemm/bf16_n129280_k5120", mode=_mode, sizes=lambda mode: WARMUP_ROWS,
                                inputs=_inputs)(_bf16_gemm)


def warmup_bf16_gemm_n129280_k5120(device: torch.device, shuffled: bool) -> None:
    """Compile the tile variants for the caller's weight layout, before graph capture."""
    w = torch.zeros(N, K, dtype=torch.bfloat16, device=device)
    x = torch.zeros(max(WARMUP_ROWS), K, dtype=torch.bfloat16, device=device)
    for m in WARMUP_ROWS:
        bf16_gemm_n129280_k5120(x[:m], w, shuffled=shuffled)
    torch.cuda.synchronize(device)
