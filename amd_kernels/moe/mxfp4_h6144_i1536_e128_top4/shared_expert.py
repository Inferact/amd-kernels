# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared H6144 I1536 expert: MXFP8 activations, weights and MFMA at every M.

The weights are either a TP2 rank's shard (I 1536) or the full expert (I 3072, ``SharedWeights.full``). With the full
copy the BF16 entry still computes the rank's shard (``tp_rank``): every shared kernel reads it in place, the gate and
up rows as two contiguous ranges of the gate/up weight and the down columns as a K slice (16-row block stride 2 I K
bytes, scale group stride 2 I bytes), the same bytes in the same order as the shard copy, so the same bits.
"""
from dataclasses import dataclass
import torch
from amd_kernels.warm_up import entry
from .shared_quant import shared_quant

H, I = 6144, 1536  # noqa: E741


# the one-launch MXFP4 + MXFP8 input quant wins at these rows only in the reduced (summed) mode
SHARED_QUANT_REDUCED_ONLY = (4032, 4096)
# first row of the combined routed / shared launch (below it the small-tile host)
SHARED_PERSISTENT_FROM = 641

# One side stream per device (``shared_parallel_for``), shared by every layer's SharedWeights: torch's caching
# allocator keeps one block pool per stream, and per-layer streams hold ~20 GiB of shared-expert outputs per rank
# under DBO until the allocator retries (sync + free every segment).
_SIDE_STREAMS = {}


def side_stream(device):
    """The shared expert's side stream of ``device`` (created on first use)."""
    device = torch.device(device)
    index = torch.cuda.current_device() if device.index is None else device.index
    stream = _SIDE_STREAMS.get(index)
    if stream is None:
        stream = _SIDE_STREAMS[index] = torch.cuda.Stream(device=torch.device("cuda", index))
    return stream


# Inside a DBO microbatch the shared expert stays on the caller's stream: DBO already overlaps one microbatch's
# compute with the other's collectives (serial, the 32768-token DBO steps took a stable 466-472 ms). A server passes
# its query with ``set_dbo_query`` (vLLM: ``vllm.v1.worker.ubatching.dbo_enabled``); without one ``DBO`` answers.
DBO = False
_dbo_query = None


def set_dbo_query(query) -> None:
    """``query()`` -> whether the current call runs inside a DBO microbatch; None returns to ``DBO``."""
    global _dbo_query
    _dbo_query = query


def _in_dbo() -> bool:
    return bool(_dbo_query()) if _dbo_query is not None else DBO


def _weight_layout(w):
    """Row-major e4m3 [N, K] -> AITER ``shuffle_weight(.., (16, 16))``: 16-row blocks of [K / 64, 4, 16 rows, 16 B]."""
    n, k = w.shape
    return w.view(torch.uint8).reshape(n // 16, 16, k // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()


def _scale_layout(s):
    """Row-major e8m0 [N, K / 32] -> AITER ``shuffle_scale_a16w4(.., 1, False)``: 32-row groups of [K / 256, 4, 16, 2,
    2]."""
    n, g = s.shape
    return s.view(torch.uint8).reshape(n // 32, 2, 16, g // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous()


@dataclass
class SharedWeights:
    """The shared expert's MXFP8 weights and scales in the kernels' layout, and the device's side stream.

    ``full``: the whole expert (I 3072); the BF16 entry computes TP2 rank ``tp_rank``'s shard of it in place
    (``tp_operands``), ``shared_expert_full`` the whole expert."""

    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    stream: object
    full: bool = False
    tp_rank: int | None = None

    @classmethod
    def from_row_major(cls, w13, s13, w2, s2, *, tp_rank=None):
        """From row-major e4m3 [N, K] weights and e8m0 [N, K / 32] scales (gate rows, then up rows in ``w13``): a
        rank's shard (w13 [3072, 6144], w2 [6144, 1536]) or the full expert (w13 [6144, 6144], w2 [6144, 3072]) with
        the TP2 rank ``tp_rank`` (0 or 1) whose shard the BF16 entry computes."""
        full = tuple(w13.shape) == (4 * I, H)
        i = 2 * I if full else I
        assert w13.shape == (2 * i, H) and w2.shape == (H, i), (w13.shape, w2.shape)
        assert s13.shape == (2 * i, H // 32) and s2.shape == (H, i // 32), (s13.shape, s2.shape)
        if full:
            assert tp_rank in (0, 1), f"the full shared expert needs tp_rank 0 or 1, got {tp_rank}"
        else:
            assert tp_rank in (None, 0, 1), tp_rank
        # A bit-exact permutation. The returned object holds only these layouts.
        return cls(_weight_layout(w13), _scale_layout(s13), _weight_layout(w2), _scale_layout(s2),
                   side_stream(w13.device), full, tp_rank)

    def tp_operands(self):
        """(w13, s13, w2, s2) flat u8 views of the TP shard the BF16 entry computes: the tensors themselves for a
        shard copy; for the full copy, each starting at rank ``tp_rank``'s first byte (gate rows r I, scale groups
        r I / 32, down K columns r I, scale K groups r I / 32), one length for both ranks."""
        flat = [t.view(torch.uint8).view(-1) for t in (self.w13, self.s13, self.w2, self.s2)]
        if not self.full:
            return tuple(flat)
        r = self.tp_rank
        spans = ((r * I * H, 3 * I * H), (r * I * (H // 32), 3 * I * (H // 32)),
                 (r * 16 * I, H * 2 * I - 16 * I), (r * I, H * (2 * I // 32) - I))
        return tuple(t[o:o + n] for t, (o, n) in zip(flat, spans))


def shared_expert(x, w, *, x_quant=None, scales_tiled=False, stream=None):
    """The shared expert of bf16 ``x`` [n, 6144] (``shared_dense``, from 96 rows) -> bf16 [n, 6144]; below 96 rows the
    shared tiles run folded into the routed small-tile launches.

    Args:
        x_quant: ``shared_quant(x)``'s (codes, scales) if already computed.
        stream: launch there instead of on the current stream; the buffers still come from the current stream's
            allocator pool and are recorded on ``stream`` (``shared_expert_parallel``).
    """
    n = x.shape[0]
    if not shared_dense_for(n):
        raise ValueError(f"the standalone shared expert takes >= 96 rows, got {n}")
    q, s = shared_quant(x, stream=stream) if x_quant is None else x_quant
    from .shared_dense import shared_dense
    return shared_dense(q, s, w, config=shared_dense_config_for(n), scales_tiled=scales_tiled, stream=stream)


def shared_expert_full(codes, scales, w, *, stream=None, keep=None):
    """The full shared expert (``w.full``) of MXFP8 codes u8 [n, 6144] and row-major e8m0 u8 [n, 192]
    (``prepare_mxfp4_moe``'s ``shared_codes`` / ``shared_scales``) -> bf16 [n, 6144].

    Args:
        stream: launch there instead of on the current stream (the caller orders it after the inputs' producer); the
            inputs and every buffer, which come from the current stream's allocator pool, are recorded on it.
        keep: a list; with a side ``stream``, the inputs and buffers are appended to it instead of recorded, and the
            caller drops them only after the current stream waited for ``stream``'s work. A recorded block is
            reusable only once the allocator sees the side stream's event complete, which lags while the host runs
            layers ahead: recorded, the pool grew by 17-20 GiB over 57 sequence-parallel layers at 32768 tokens.
    """
    from .shared_dense import shared_dense_full
    if not w.full:
        raise ValueError("shared_expert_full needs the full shared expert (SharedWeights.full)")
    n = codes.shape[0]
    if (codes.shape != (n, H) or scales.shape != (n, H // 32) or codes.dtype != torch.uint8
            or scales.dtype != torch.uint8 or not codes.is_contiguous() or not scales.is_contiguous() or n < 1):
        raise ValueError(f"expected contiguous u8 codes [n, {H}] and scales [n, {H // 32}], got "
                         f"{tuple(codes.shape)} / {tuple(scales.shape)}")
    if keep is None:
        _record(stream, codes, scales)
    elif stream is not None and stream != torch.cuda.current_stream():
        keep.extend((codes, scales))
    return shared_dense_full(codes, scales, w, stream=stream, keep=keep)


# one row count per tier of ``shared_dense.FULL_CONFIGS`` (the sequence-parallel own rows reach 1..16384)
FULL_WARM_UP_ROWS = (1, 1281, 2561, 4097)


def _full_inputs(mode, n, codes, scales, w, *, stream=None, keep=None):
    dev = codes.device
    def zeros(cols):
        return torch.zeros((n, cols), dtype=torch.uint8, device=dev)

    return (zeros(H), zeros(H // 32), w), {}


shared_expert_full = entry("moe/mxfp4_h6144_i1536_e128_top4:shared_full", mode=lambda *a, **k: ("full",),
                           sizes=lambda mode: FULL_WARM_UP_ROWS, inputs=_full_inputs)(shared_expert_full)


def _record(stream, *tensors):
    """Records the current stream's ``tensors`` on ``stream``, so their blocks wait for its work before reuse."""
    if stream is not None and stream != torch.cuda.current_stream():
        for t in tensors:
            t.record_stream(stream)


def shared_expert_parallel(x, w, current, **kwargs):
    """``shared_expert`` on ``w.stream`` (the device's side stream) after the work queued on ``current`` so far.

    Every buffer comes from ``current``'s allocator pool (recorded on the side stream), so the allocator caches nothing
    under the side stream. The caller joins with ``current.wait_stream(w.stream)`` before it reads the result."""
    side = w.stream
    side.wait_stream(current)
    _record(side, x, *(kwargs.get("x_quant") or ()))
    return shared_expert(x, w, stream=side, **kwargs)


def shared_folded_for(n):
    """Whether the shared tiles lead both routed 16 / 32-row GEMM launches (below 96 rows). From 96 rows the routed
    launches run alone, at the routed tiles' registers and LDS (the folded 64-row shared tiles hold them to two waves
    per SIMD), next to the shared expert on the side stream: -4..-15% at 128-512 rows."""
    return n < 96


# second stream vs serial: -2.1..-6.6% at 16385-28672 rows, -0.5..-1.5% to 30720, noise at 31744 / 32768
SHARED_PARALLEL_MAX = 30720


def shared_parallel_for(n):
    """Whether the shared expert runs on the side stream (96..``SHARED_PARALLEL_MAX`` rows, not inside DBO)."""
    return 96 <= n <= SHARED_PARALLEL_MAX and not _in_dbo()


def shared_dual_quant_for(n):
    # the MXFP4 + MXFP8 input quant in one launch on these rows (the same bits as two)
    return (n <= 1024 or SHARED_QUANT_REDUCED_ONLY[0] <= n <= SHARED_QUANT_REDUCED_ONLY[1]
            or 7425 <= n <= 8192
            or 10240 <= n <= 10880 or 13568 <= n <= 32768)


def shared_dense_for(n):
    return n >= 96


def shared_dense_config_for(n):
    if n <= 512:
        return (2,2),(2,2)
    if n <= 1024:
        return (2,4),(4,4)
    return (4,4),(4,4)


def shared_in_persistent_for(n):
    """Whether routed and shared GEMMs share one launch (the BM128 host) on BF16-entry rows (641-3583; below, the
    small-tile host's routed launches next to the shared expert on the second stream)."""
    # the adaptive 3584-4096 tier did not pass the entry's noise gate
    return SHARED_PERSISTENT_FROM<=n<=3583
