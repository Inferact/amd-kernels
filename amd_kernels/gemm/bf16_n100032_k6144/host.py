# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the lm_head shard GEMM: the load-time weight layout, the tile by M, the launch, the warm-up."""
import functools

import torch

from amd_kernels.warm_up import entry

from .kernel import compile_bf16_gemm_n100032_k6144
from .indexed import compile_indexed_bf16_gemm_n100032_k6144
from .large import compile_large_bf16_gemm_n100032_k6144
from .large_indexed import compile_large_indexed_bf16_gemm_n100032_k6144
from .reference import K, N

MAX_M = 2048


def supports(w: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    """Whether ``w`` is the [100032, 6144] bf16 contiguous shard this GEMM (and ``shuffle_weight_``) takes; the GEMM
    has no bias."""
    return bias is None and tuple(w.shape) == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous()


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """The load-time layout [6252 groups, 96 K blocks, 16 rows, 64 K] of the [100032, 6144] shard, as [100032, 6144].

    Bit-exact (a permutation). The caller replaces the parameter with it once, before graph capture, and frees the
    row-major tensor (no second copy); every reader of the weight then calls bf16_gemm_n100032_k6144(shuffled=True).
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
    """The inverse of shuffle_weight (row-major [100032, 6144])."""
    assert ws.shape == (N, K) and ws.dtype == torch.bfloat16
    return ws.view(N // 16, K // 64, 16, 64).permute(0, 2, 1, 3).contiguous().view(N, K)


def tile_config(m: int) -> tuple:
    """(kernel, 16-row fragments per wave, M halves, vocabulary tiles, ring) for m rows; the bounds and tiles are the
    fastest of the graph-replay sweep over 1-2048 rows (README). Above 256 rows the M tiles of one vocabulary tile
    run on one XCD (the weight tile is re-read from L2); their fragment counts are balanced."""
    if m <= 96:
        return "small", (m + 15) // 16, 1, 256, 0
    if m <= 128:
        return "small", (m + 31) // 32, 2, 256, 0
    if m <= 144:
        return "small", 9, 1, 256, 2
    mtiles = (m + 255) // 256
    mt = (m + 32 * mtiles - 1) // (32 * mtiles)
    if m <= 192 or 256 < m <= 384:
        return "small", mt, 2, 512, 3
    return "large", mt, 2, (400 if m <= 224 else 512 if m <= 512 else 448), 0


FORMS = ("contiguous", "rows", "publish")


@functools.cache
def _kernel(cfg: tuple, shuf: bool, form: str):
    """The tile variant: contiguous rows, mapped rows (``rows``), or mapped rows published into the exchange."""
    kind, mt, mh, tiles, ring = cfg
    if form != "contiguous":
        if kind == "large":
            return compile_large_indexed_bf16_gemm_n100032_k6144(mt, int(shuf), tiles, publish=form == "publish")
        return compile_indexed_bf16_gemm_n100032_k6144(mt, int(shuf), mh, tiles, ring, publish=form == "publish")
    if kind == "large":
        return compile_large_bf16_gemm_n100032_k6144(mt, int(shuf), tiles)
    return compile_bf16_gemm_n100032_k6144(mt, int(shuf), mh, tiles, ring)


def _tiling(m: int, shuffled: bool, form: str):
    cfg = tile_config(m)
    rows = 16 * cfg[1] * cfg[2]
    return _kernel(cfg, bool(shuffled), form), (m + rows - 1) // rows


def _check_out(out: torch.Tensor, m: int):
    """A row-major [m, 100032] bf16 view whose rows may be wider (a column slice): row stride in dwords."""
    assert out.shape == (m, N) and out.dtype == torch.bfloat16 and out.stride(1) == 1
    assert out.data_ptr() % 4 == 0 and (m <= 1 or out.stride(0) % 2 == 0) and out.stride(0) >= N
    assert m * out.stride(0) * 2 <= 0xFFFFFFFF
    return out.stride(0) // 2 if m > 1 else N // 2


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _bf16_gemm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, *,
              shuffled: bool = False, row_indices: torch.Tensor | None = None,
              publish: tuple | None = None) -> torch.Tensor | None:
    """out [M, 100032] bf16 = x [M, 6144] bf16 @ W^T, fp32 accumulate; W = w (row-major) or shuffle_weight's layout.

    Any M (above 2048 rows: launches of 2048 rows each, the same bits as separate calls). Deterministic, no atomic
    output adds. CUDA-graph safe (no host sync, no allocation when out is given). ``out`` may be a column slice of a
    wider row-major tensor (unit column stride, even row stride). The first call of each mode (layout, form)
    compiles the mode's variants (``amd_kernels.warm_up``).

    row_indices (contiguous device int64 [M], each index in [0, len(x)), duplicates valid, re-read on every graph
    replay, 1 <= M <= 2048): the rows x[row_indices], in map order (form ``rows``); the same bits as the contiguous
    call on x[row_indices].
    publish = RowAllToAll.slot() of all_to_all_bf16_n100032_ranks2, with row_indices and no out: the mapped rows go
    straight into the exchange's send slot and the kernel publishes them (no stage copy); the caller then runs
    RowAllToAll.pull(M, counts, out). Returns None.
    """
    assert publish is None or row_indices is not None, "publish needs row_indices"
    assert x.ndim == 2 and x.shape[1] == K and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert w.shape == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous() and w.device == x.device
    stream = torch.cuda.current_stream()
    if row_indices is not None:
        m = row_indices.numel()
        assert row_indices.ndim == 1 and row_indices.dtype == torch.int64 and row_indices.is_contiguous()
        assert row_indices.device == x.device and x.numel() * x.element_size() <= 0xFFFFFFFF and 1 <= m <= MAX_M
        if publish is not None:
            pool, slot_bytes, counter, flag = publish
            assert out is None and m * N * 2 <= slot_bytes
            assert counter.dtype == torch.int32 and counter.device == x.device
            exe, mtiles = _tiling(m, shuffled, "publish")
            _run(exe, x.data_ptr(), w.data_ptr(), pool, row_indices, x.shape[0], m, mtiles, counter, slot_bytes,
                 flag, stream)
            return None
        if out is None:
            out = torch.empty(m, N, dtype=torch.bfloat16, device=x.device)
        ldy = _check_out(out, m)
        assert out.device == x.device
        exe, mtiles = _tiling(m, shuffled, "rows")
        _run(exe, x.data_ptr(), w.data_ptr(), out.data_ptr(), row_indices, x.shape[0], m, mtiles, ldy, stream)
        return out
    m = x.shape[0]
    if out is None:
        out = torch.empty(m, N, dtype=torch.bfloat16, device=x.device)
    if m > MAX_M:
        for i in range(0, m, MAX_M):
            _bf16_gemm(x[i:i + MAX_M], w, out[i:i + MAX_M], shuffled=shuffled)
        return out
    ldy = _check_out(out, m)
    assert out.device == x.device
    if m:
        exe, mtiles = _tiling(m, shuffled, "contiguous")
        _run(exe, x.data_ptr(), w.data_ptr(), out.data_ptr(), m, mtiles, ldy, stream)
    return out


def _form(row_indices, publish) -> str:
    return "contiguous" if row_indices is None else ("rows" if publish is None else "publish")


def _mode(x, w, out=None, *, shuffled=False, row_indices=None, publish=None):
    """What a call fixes about its kernels besides M (``amd_kernels.warm_up``): the weight layout and the call form."""
    return bool(shuffled), _form(row_indices, publish)


_KEEP: list = []  # the private publish slot of the warm-up call being made


def _inputs(mode, n, x, w, out=None, *, shuffled=False, row_indices=None, publish=None):
    """One warm-up call at ``n`` rows (zeros, row 0) into a private output. A variant compiled before the warm-up
    launches even in compile-only mode, so a publish call gets a private slot pair, counter and flag, never the
    caller's exchange."""
    if mode[1] == "contiguous":
        return (torch.zeros(n, K, dtype=torch.bfloat16, device=x.device), w), {"shuffled": shuffled}
    idx = torch.zeros(n, dtype=torch.int64, device=x.device)
    if mode[1] == "rows":
        return (x, w), {"shuffled": shuffled, "row_indices": idx}
    slot_bytes = n * N * 2
    _KEEP[:] = [torch.empty(2 * slot_bytes, dtype=torch.uint8, device=x.device),
                torch.zeros(1, dtype=torch.int32, device=x.device),
                torch.zeros(16, dtype=torch.int32, device=x.device)]
    publish = (_KEEP[0].data_ptr(), slot_bytes, _KEEP[1], _KEEP[2].data_ptr())
    return (x, w), {"shuffled": shuffled, "row_indices": idx, "publish": publish}


# one row count per tile variant (tile_config); every other row count up to MAX_M reuses one of these
WARMUP_ROWS = sorted({tile_config(m): m for m in range(MAX_M, 0, -1)}.values())
WARMUP_MODES = FORMS

bf16_gemm_n100032_k6144 = entry("gemm/bf16_n100032_k6144", mode=_mode, sizes=lambda mode: WARMUP_ROWS,
                               inputs=_inputs)(_bf16_gemm)


def warmup_bf16_gemm_n100032_k6144(device: torch.device, shuffled: bool, mode: str = "contiguous") -> None:
    """Compile the tile variants of one call mode for the caller's weight layout, before graph capture.

    Each mode is its own kernel, so warm up only the modes the caller runs (once per mode):
    contiguous = bf16_gemm_n100032_k6144(x, w), rows = row_indices=..., publish = row_indices=..., publish=...
    """
    assert mode in WARMUP_MODES, mode
    w = torch.zeros(N, K, dtype=torch.bfloat16, device=device)
    x = torch.zeros(max(WARMUP_ROWS), K, dtype=torch.bfloat16, device=device)
    indices = torch.arange(x.shape[0], dtype=torch.int64, device=device)
    publish = None
    if mode == "publish":  # a private scratch slot pair, counter and flag (no peer is involved)
        slot_bytes = x.shape[0] * N * 2
        pool = torch.empty(2 * slot_bytes, dtype=torch.uint8, device=device)
        counter = torch.zeros(1, dtype=torch.int32, device=device)
        flag = torch.zeros(16, dtype=torch.int32, device=device)
        publish = (pool.data_ptr(), slot_bytes, counter, flag.data_ptr())
    for m in WARMUP_ROWS:
        if mode == "contiguous":
            bf16_gemm_n100032_k6144(x[:m], w, shuffled=shuffled)
        else:
            bf16_gemm_n100032_k6144(x, w, shuffled=shuffled, row_indices=indices[:m], publish=publish)
    torch.cuda.synchronize(device)
