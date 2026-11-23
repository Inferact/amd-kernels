# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entries of the N 32768 x K 1280 MXFP8 GEMM: weight preshuffle, row dispatch, warm-up."""

import torch

from amd_kernels.gemm.mxfp8_common import gemm as _common
from amd_kernels.gemm.mxfp8_decode_prequant import gemm as _decode
from amd_kernels.gemm.mxfp8_decode_prequant import gemm_swiglu_mxfp8 as _decode_swiglu
from amd_kernels.warm_up import entry

from . import kernel as _kernel

N, K = _kernel.N, _kernel.K
KEY = "mxfp8_n32768_k1280"
LIMIT = 10.0


def preshuffle_weight(weight: torch.Tensor, weight_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 codes [N, K] and E8M0 scales, [N, K / 32] or the checkpoint's 32 x 32 blocks [N / 32, K / 32] (expanded
    to one row each), -> the layouts the entries read (load time)."""
    assert tuple(weight.shape) == (N, K) and weight.dtype == torch.float8_e4m3fn and weight_scale.dtype == torch.uint8
    if tuple(weight_scale.shape) == (N // 32, K // 32):
        weight_scale = weight_scale.repeat_interleave(32, dim=0)
    assert tuple(weight_scale.shape) == (N, K // 32)
    return _common.preshuffle_mxfp8_weight(weight, weight_scale)


def preshuffle_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """The scales the entries read, for a weight that already has ``preshuffle_weight``'s code layout (AITER's
    (16, 16) shuffle, as the served linear's): E8M0 [N, K / 32] or the 32 x 32 blocks [N / 32, K / 32]."""
    assert weight_scale.dtype == torch.uint8
    if tuple(weight_scale.shape) == (N // 32, K // 32):
        weight_scale = weight_scale.repeat_interleave(32, dim=0)
    assert tuple(weight_scale.shape) == (N, K // 32)
    return _shuffle_scale(weight_scale)


def _shuffle_scale(scale: torch.Tensor) -> torch.Tensor:
    from aiter.ops.shuffle import shuffle_scale_a16w4

    return shuffle_scale_a16w4(scale, 1, False)


def serves(m: int, table=None) -> bool:
    """Whether the entry beat the served path at ``m`` rows (``kernel.py``); callers fall back where it did not."""
    r = _kernel.row(_kernel.LINEAR if table is None else table, m)
    return r is not None and r[2]


def _run_rows(table, x, weight, weight_scale):
    m = x.shape[0]
    r = _kernel.row(table, m)
    if r is None:
        return _common.mxfp8_linear(x, weight, weight_scale, torch.bfloat16)
    kind, _, _, cfg = r
    if kind == "fq":
        return _decode(x, None, weight, weight_scale, cfg, key=KEY)
    if kind == "shuf":
        codes, scale = _common.mxfp8_e4m3_quantize_shuffled(x, 16 * cfg.rt)
        return _decode(codes, scale, weight, weight_scale, cfg, layout="shuffled", rows=m, key=KEY)
    return _common.mxfp8_linear(x, weight, weight_scale, torch.bfloat16, config=_kernel.common_config(cfg))


def _mxfp8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.shape[1] == K and x.dtype == torch.bfloat16 and x.is_contiguous()
    return _run_rows(_kernel.LINEAR, x, weight, weight_scale)


def _mode(x, weight, weight_scale):
    return ()


def _sizes(mode):
    return tuple(row[1] for row in _kernel.LINEAR)


def _inputs(mode, n, x, weight, weight_scale):
    return (torch.zeros(n, K, dtype=torch.bfloat16, device=x.device), weight, weight_scale), {}


mxfp8_linear = entry("gemm/mxfp8_n32768_k1280", mode=_mode, sizes=_sizes, inputs=_inputs)(_mxfp8_linear)
mxfp8_linear.__doc__ = """``x @ W^T`` for a bf16 activation ``x`` [M, 1280] and a preshuffled MXFP8 weight (``preshuffle_weight``):
quantized per 1 x 32 block as ``reference.py`` says, bf16 [M, 32768] out. A fresh output per call; CUDA-graph safe
(static launch arguments, no host sync); deterministic. Rows above ``kernel.LINEAR``'s last run the common
dispatcher. Rows whose split-K sum runs in the last workgroup of a tile share one counter arena per device: one
launch at a time."""


MAX_PREQUANT_ROWS = _kernel.PREQUANT[-1][1]
SCALE_COLS = K // 32
_SCALE_INDEX: dict = {}


def _scale_index(device: torch.device) -> torch.Tensor | None:
    """AITER's activation-scale layout as one gather index over ``MAX_PREQUANT_ROWS`` x 40 row-major scales (int32).
    The layout is block-major in 32-row groups, so the first ``32 g`` rows' part is the index of an ``32 g``-row tensor.
    Built on first use outside graph capture; None while capturing without it (the caller takes the copying path)."""
    idx = _SCALE_INDEX.get(device)
    if idx is None and not torch.cuda.is_current_stream_capturing():
        base = torch.arange(MAX_PREQUANT_ROWS * SCALE_COLS, dtype=torch.int32).view(MAX_PREQUANT_ROWS, SCALE_COLS)
        idx = _SCALE_INDEX[device] = _shuffle_scale(base).reshape(-1).to(device)
    return idx


def _activation_scale(scale: torch.Tensor) -> torch.Tensor:
    """Row-major e8m0 scales [M, 40] -> the kernel's layout, rows padded to 32: one gather launch when ``scale``'s
    storage has the padding rows (the producer allocates ``ceil(M / 32)`` rows and passes the first M; the gather reads
    the padding rows, whose outputs the GEMM drops), else the copying path (``mxfp8_common``)."""
    m = scale.shape[0]
    padded = (m + 31) // 32 * 32
    idx = _scale_index(scale.device) if padded <= MAX_PREQUANT_ROWS else None
    room = scale.untyped_storage().nbytes() - scale.storage_offset() * scale.element_size()
    if idx is None or room < padded * SCALE_COLS:
        return _common._preshuffle_activation_scale(scale)
    src = torch.as_strided(scale, (padded * SCALE_COLS,), (1,), scale.storage_offset())
    return torch.index_select(src, 0, idx[: padded * SCALE_COLS]).view(padded, SCALE_COLS)


def _prequant_mode(codes, scale, weight, weight_scale):
    return ()


def _padded_scale(n: int, device) -> torch.Tensor:
    """Unit scales [n, 40] in the storage of ``ceil(n / 32)`` rows (what the producer hands over)."""
    return torch.full(((n + 31) // 32 * 32, SCALE_COLS), 127, dtype=torch.uint8, device=device)[:n]


def _prequant_inputs(mode, n, codes, scale, weight, weight_scale):
    dev = codes.device
    return (torch.zeros(n, K, dtype=torch.float8_e4m3fn, device=dev), _padded_scale(n, dev), weight, weight_scale), {}


def _mxfp8_linear_prequant(codes: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor):
    assert codes.dim() == 2 and codes.shape[1] == K and codes.dtype == torch.float8_e4m3fn and codes.is_contiguous()
    assert scale.dtype == torch.uint8 and tuple(scale.shape) == (codes.shape[0], SCALE_COLS) and scale.is_contiguous()
    r = _kernel.row(_kernel.PREQUANT, codes.shape[0])
    if r is None:
        raise ValueError(f"no pre-quantized row above {MAX_PREQUANT_ROWS} rows: use AITER there")
    return _common.mxfp8_gemm(codes, _activation_scale(scale), weight, weight_scale, torch.bfloat16,
                              activation_scale_preshuffled=True, config=_kernel.common_config(r[3]))


mxfp8_linear_prequant = entry("gemm/mxfp8_n32768_k1280:prequant", mode=_prequant_mode, sizes=lambda mode: tuple(
    row[1] for row in _kernel.PREQUANT), inputs=_prequant_inputs)(_mxfp8_linear_prequant)
mxfp8_linear_prequant.__doc__ = """``A @ W^T`` for pre-quantized activations: E4M3 codes [M, 1280] and row-major E8M0 scales
[M, 40] (vLLM's ``QuantizedActivation``), bf16 [M, 32768] out, for M up to 1024; ``weight`` in AITER's (16, 16) layout
(the served linear's weight as it is) and ``weight_scale`` from ``preshuffle_weight_scale``. ``serves(m, table=
kernel.PREQUANT)`` says where it beats AITER on the same codes. One launch moves the scales when ``scale`` is the first
M rows of a tensor with ``ceil(M / 32)`` rows of storage. Same contract and safety as ``mxfp8_linear``."""


def warmup_mxfp8_linear(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    """Compile every row's variants and create the counter arena before graph capture (a first call inside a
    capture does not warm up)."""
    dev = weight.device
    for m in _sizes(()):
        mxfp8_linear(torch.zeros(m, K, dtype=torch.bfloat16, device=dev), weight, weight_scale)
    for row in _kernel.PREQUANT:
        mxfp8_linear_prequant(torch.zeros(row[1], K, dtype=torch.float8_e4m3fn, device=dev),
                              _padded_scale(row[1], dev), weight, weight_scale)
    torch.cuda.synchronize(dev)


# The folder's protocol with ``gemm.mxfp8_common`` (``gemm._ROUTES``): its common-kernel rows, and its own
# entries for the row counts its table covers.
ROWS = {
    (N, K, "bf16"): tuple(
        (rows, _kernel.common_config(cfg)) for kind, rows, _, cfg in _kernel.LINEAR if kind == "common"
    )
}
LAUNCHERS = {}


def select(m: int, out_dtype: torch.dtype, allow_decode: bool):
    return _common.select_from_rows(ROWS, m, N, K, out_dtype, allow_decode)


def linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, out_dtype: torch.dtype):
    """``mxfp8_linear``'s call where ``kernel.LINEAR`` has a row, else None (the common dispatcher)."""
    if out_dtype != torch.bfloat16 or not x.is_contiguous() or _kernel.row(_kernel.LINEAR, x.shape[0]) is None:
        return None
    return mxfp8_linear(x, weight, weight_scale)


def covers(m: int, prequant: bool = False) -> bool:
    """Whether the folder's row for ``m`` rows (bf16, or row-major codes with ``prequant``) beat the served path."""
    return serves(m, table=_kernel.PREQUANT if prequant else None)


def gemm(codes: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
         out_dtype: torch.dtype):  # fmt: skip
    """``mxfp8_gemm``'s call on row-major codes and scales where ``kernel.PREQUANT`` has a serving row, else None."""
    if out_dtype != torch.bfloat16 or not serves(codes.shape[0], table=_kernel.PREQUANT) or not codes.is_contiguous():
        return None
    return mxfp8_linear_prequant(codes, scale, weight, weight_scale)


def warmup(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    warmup_mxfp8_linear(weight, weight_scale)
