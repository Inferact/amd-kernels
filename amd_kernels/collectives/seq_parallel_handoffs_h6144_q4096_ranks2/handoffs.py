# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The hand-offs of a sequence-parallel decoder layer on two ranks (hidden 6144, 4096 attention-output columns per
rank): the multi-kernel chains between the head-sharded parts of a layer (qkv, attention, the FFN of every token) and
the token-sharded residual stream (rank r holds ``rows`` consecutive tokens, ``rows`` a multiple of 32).

``SeqParallelHandoffs`` owns the four communicators of ``pull_collectives_ranks2`` (and the pools of the unpacking
all-to-all), the per-phase pool slots, the MXFP8 pack of the attention output
(``mxfp8_pack_q4096``) and the warm-up of every kernel the phases launch (``amd_kernels.warm_up.entry``). The caller
passes tensors: this rank's rows, the layer's weights, epsilons. Per layer:

1. ``input_gather(x)``: the first layer's normed bf16 rows -> MXFP8 codes + AITER-shuffled scales of every token
   (``gemm/mxfp8_common``'s quantizer on the shard, then ``AllGatherComm``).
2. ``attention_moe(attn_output, o_weight, o_scale, residual, norm_weight, eps, moe_weights)``: this rank's heads of
   every token -> the pack into the pool slot, ``FencedAllGatherComm.all_to_all_unpack_shuffled`` (the whole o_proj
   input of this rank's rows, scales in the GEMM's layout), o_proj_full on ``mxfp8_gemm``, ``gemma_add_rmsnorm``
   (bf16), ``prepare_mxfp4_moe``; its routed operands gathered (``AllGatherComm``, 3296 B/token) -> the routed-only
   ``MoeInput`` of every token for ``mxfp4_moe(..., return_partials=True)``, the new residual shard and the
   ``SharedPending`` full shared expert of this rank's rows (``shared_expert_full`` on the instance's side stream,
   forked after the preparation, enqueued after the gather: it overlaps the gather and the routed experts).
   ``attention_dense``: the same o_proj, then ``NormAllGatherComm`` (add + Gemma RMSNorm + MXFP8 quant of the fp32
   norm + gather in one launch) -> the dense MLP's codes + scales of every token, the residual shard.
3. ``ffn_norm_gather(ffn_out, residual, norm_weight, eps)``: the FFN output of every token (MoE: ``(slots,
   shared)``, ``shared`` the ``SharedPending`` (joined first: the current stream waits on its event) or a [2 rows, H]
   partial; dense: the bf16 partial) -> the reduce-scatter (fp32 in rank order; the owner's shared output added to
   its own rows' slot sum), the next layer's input norm + MXFP8 quant on this rank's rows, gathered:
   ``SlotsReduceScatterComm.slots4_reduce_scatter_add_rmsnorm_mxfp8`` + ``AllGatherComm``, or the dense
   reduce-scatter + ``NormAllGatherComm``.
   After the last layer ``ffn_reduce_scatter(ffn_out)`` gives this rank's summed rows for the model's final norm, and
   ``gather(x)`` gathers bf16 rows (the normed output, an aux hidden state).

Every collective moves bf16 partial sums or quantized next-GEMM inputs (no quantized reduction, no atomics). The
phases run on the caller's current stream, eagerly (the pool slots carry host-side sequence state; not inside CUDA
graph capture); every rank calls them with the same shapes in the same order. The shared expert's side stream joins
by an event on its exact launch before the reduce-scatter's first kernel, whose norm workers spin on in-launch
tickets: the persistent shared GEMM never runs next to them, and it waits on nothing itself (next to the gather's
spinning pullers it only delays them). Its inputs and buffers belong to the current stream's pool and are not
recorded on the side stream: the ``SharedPending`` holds them until the join, so they are freed in current-stream
order and the pool does not grow while the host runs layers ahead.

Warm-up (root README rule 4): the communicators compile their transport kernels at construction (a collective);
``warm_up(o_weight, o_scale, moe_weights)`` makes the first call of each compute mode (the quantizer, o_proj, o_proj +
norm + MoE preparation, the pack, the routed-only partials mode of ``mxfp4_moe``, ``shared_expert_full``), each of
which compiles every variant of its rows range first (``ROWS``; ``amd_kernels.warm_up``). The compute modes run no
collective, so a rank's warm-up never waits for another.
"""

from dataclasses import dataclass, field

import torch

from amd_kernels.warm_up import entry

from .mxfp8_pack_q4096 import ROW_BYTES, mxfp8_pack_q4096

WORLD = 2
H = 6144
Q = 4096  # attention output per rank: 32 heads x 128
ROW_ALIGN = 32  # shard rows: whole 32-row scale groups (NormAllGatherComm, the GEMMs' scale layout)
# 16384 rows per rank: gather_mxfp4_moe's and NormAllGatherComm's bound at two ranks.
MAX_TOKENS = 32768
MAX_ROWS = MAX_TOKENS // WORLD
# The served range starts at 2048 tokens (1024 rows per rank; a MoE layer vs the TP path: -20% at 2048 tokens, even at
# 1024, benchmark_sp_layer.py): the warm-up covers MIN_ROWS..MAX_ROWS.
MIN_TOKENS = 2048
MIN_ROWS = MIN_TOKENS // WORLD
_FP8 = torch.float8_e4m3fn

# the routed-only MoeInput's fields in gather order: (columns, dtype), 3296 B/token
_FP4 = (
    (H // 2, torch.uint8),  # routed MXFP4 codes
    (H // 32, torch.uint8),  # their e8m0 scales
    (4, torch.float32),  # top-4 weights (routed scale included)
    (4, torch.int32),  # top-4 ids
)
# the full shared expert's output stays on its rows' owner (no link bytes), added in the MoE tail's reduce-scatter
OWNER_SHARED = True


@dataclass
class SharedPending:
    """The full shared expert's bf16 output of this rank's rows [rows, H], computed on a hand-offs instance's side
    stream, the event recorded after its launch there, the stream whose pool its buffers came from and the tensors
    the side stream reads or writes (released by the join)."""

    out: torch.Tensor
    event: torch.cuda.Event
    origin: torch.cuda.Stream | None = None
    keep: list = field(default_factory=list)

    def wait(self) -> None:
        """The current stream (and the pool's stream, if another) waits for the shared expert, then the held
        buffers are released."""
        stream = torch.cuda.current_stream(self.out.device)
        stream.wait_event(self.event)
        if self.origin is not None and self.origin != stream:
            self.origin.wait_event(self.event)
            self.out.record_stream(stream)
        self.keep.clear()

    def __del__(self):
        # never joined (an exception between fork and join): the held blocks must not return to the pool before
        # the side stream is done with them
        if self.keep and self.origin is not None:
            self.origin.wait_event(self.event)


# up to here the full shared expert runs on the current stream after the gather: on the side stream it collides with
# the routed experts instead of hiding under the gather (SP pull layer, benchmark_sp_layer.py: 2048 tokens +6%, 2304
# +6%, 2560 even, 2816 -2.4%, 4096 -4%, 8192 -5%)
SHARED_SERIAL_MAX_ROWS = 1152


def _fork_shared(side, local, moe_weights, after) -> SharedPending:
    """``shared_expert_full`` of the prepared ``local`` rows on ``side`` once the current stream's event ``after`` (the
    preparation) completed (``side`` None or at most ``SHARED_SERIAL_MAX_ROWS`` rows: on the current stream); its
    buffers come from the current stream's pool."""
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import shared_expert_full

    current = torch.cuda.current_stream()
    if side is None or local.shared_codes.shape[0] <= SHARED_SERIAL_MAX_ROWS:
        side = current
    else:
        side.wait_event(after)
    keep = []
    out = shared_expert_full(local.shared_codes, local.shared_scales, moe_weights.shared_weights, stream=side,
                             keep=keep)
    done = torch.cuda.Event()
    done.record(side)
    return SharedPending(out, done, current, keep)


def _ffn_parts(ffn_out):
    """``(slots, shared)`` with a ``SharedPending`` joined (the current stream waits) and replaced by its output."""
    slots, shared = ffn_out
    if isinstance(shared, SharedPending):
        shared.wait()
        shared = shared.out
    return slots, shared


def _full_shared(moe_weights) -> bool:
    return bool(getattr(moe_weights.shared_weights, "full", False))


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def shard_rows(num_tokens: int) -> int:
    """Rows per rank of a ``num_tokens`` call (``ROW_ALIGN``-aligned, the last ranks' rows padded with zeros)."""
    return _round_up(-(-num_tokens // WORLD), ROW_ALIGN)


def shard(x: torch.Tensor, rank: int) -> torch.Tensor:
    """Rank ``rank``'s ``shard_rows`` rows of every token's ``x`` ``[t, ...]``, zero padded past ``t``."""
    rows = shard_rows(x.shape[0])
    start = rank * rows
    n = min(rows, max(0, x.shape[0] - start))
    if n == rows:
        return x[start : start + n].contiguous()
    out = x.new_zeros((rows, *x.shape[1:]))
    if n:
        out[:n].copy_(x[start : start + n])
    return out


# The shard compute of the phases (no collective): one registered entry, a mode per kind.


def _shard_compute(kind, x, scales=None, o_weight=None, o_scale=None, residual=None, norm_weight=None, eps=0.0,
                   moe_weights=None):  # fmt: skip
    """``input``: bf16 rows ``x`` -> MXFP8 codes + shuffled scales. ``dense``: o_proj_full of the unpacked codes ``x``
    + ``scales`` -> bf16. ``moe``: that o_proj, the bf16 add + Gemma RMSNorm with ``residual``, the MoE preparation ->
    (``MoeInput``, new residual)."""
    from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_e4m3_quantize_flydsl, mxfp8_gemm

    if kind == "input":
        return mxfp8_e4m3_quantize_flydsl(x)
    h = mxfp8_gemm(x, scales, o_weight, o_scale, torch.bfloat16, activation_scale_preshuffled=True)
    if kind == "dense":
        return h
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import prepare_mxfp4_moe
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.host import gemma_add_rmsnorm

    y, r = gemma_add_rmsnorm(h, residual, norm_weight, eps)
    return prepare_mxfp4_moe(y, moe_weights), r


def _shard_mode(kind, *args, moe_weights=None, **kwargs):
    """What a call fixes about its kernels besides the row count (``amd_kernels.warm_up``)."""
    if kind != "moe":
        return (kind,)
    return kind, moe_weights.shared_weights is not None, moe_weights.router_weight_lines is not None


def _gemm_rows() -> tuple[int, ...]:
    """One row count per o_proj_full variant (``mxfp8_gemm``'s table row and its four-wave remainder plan) over the
    served rows: what ``warmup_mxfp8_gemm_weight`` walks, for this one shape and row-major codes."""
    from amd_kernels.gemm.mxfp8_common.gemm import _FOURWAVE_KERNELS, _fourwave_plan, _select_config

    seen = {}
    for m in range(MIN_ROWS, MAX_ROWS + 1, ROW_ALIGN):
        config = _select_config(m, H, WORLD * Q, torch.bfloat16, False)
        key: tuple = (config,)
        if config.kernel in _FOURWAVE_KERNELS:
            key += tuple(tiles for tiles, _, _ in _fourwave_plan(m, H, config))
        seen.setdefault(key, m)
    return tuple(sorted(seen.values()))


def _shard_sizes(mode) -> tuple[int, ...]:
    # the quantizer's variant is the same from 1024 rows (rows per wave by row count)
    return (MIN_ROWS,) if mode[0] == "input" else _gemm_rows()


def _shard_inputs(mode, n, kind, x, scales=None, o_weight=None, o_scale=None, residual=None, norm_weight=None,
                  eps=0.0, moe_weights=None):  # fmt: skip
    """One warm-up call of the mode at ``n`` rows (zeros: the dispatch depends on shapes only)."""
    dev = x.device
    if kind == "input":
        return ("input", torch.zeros((n, H), dtype=torch.bfloat16, device=dev)), {}
    codes = torch.zeros((n, WORLD * Q), dtype=_FP8, device=dev)
    sc = torch.full((_round_up(n, 32), WORLD * Q // 32), 127, dtype=torch.uint8, device=dev)
    if kind == "dense":
        return (kind, codes, sc, o_weight, o_scale), {}
    res = torch.zeros((n, H), dtype=torch.bfloat16, device=dev)
    return (kind, codes, sc, o_weight, o_scale, res, norm_weight, eps), {"moe_weights": moe_weights}


shard_compute = entry("collectives/seq_parallel_handoffs_h6144_q4096_ranks2", mode=_shard_mode, sizes=_shard_sizes,
                      inputs=_shard_inputs)(_shard_compute)  # fmt: skip


def _pack_mode(x, out=None, **kwargs):
    return tuple(sorted(kwargs.items()))


def _pack_inputs(mode, n, x, out=None, **kwargs):
    return (torch.zeros((n, Q), dtype=torch.bfloat16, device=x.device),), kwargs


# one variant (rows are a runtime argument)
pack = entry("collectives/seq_parallel_handoffs_h6144_q4096_ranks2:pack", mode=_pack_mode, sizes=lambda mode: (32,),
             inputs=_pack_inputs)(mxfp8_pack_q4096)  # fmt: skip


class SeqParallelHandoffs:
    """The four communicators of the phases, one of each per rank, sized for ``max_tokens`` tokens per call (a
    server's batch-token limit, clamped to ``MIN_TOKENS``..``MAX_TOKENS``). Collective: both ranks of the two-rank
    ``cpu_group`` (gloo) constructs it at the same point; it compiles the transport kernels (``warm_up`` of each
    communicator, real collectives on zero payloads). ``coded_tail``: the MoE tail's reduce-scatter in
    ``lossless_x3``'s exact code (the same bits, 0.781x its link bytes), else BF16."""

    def __init__(self, rank: int, cpu_group, device: torch.device, max_tokens: int = MAX_TOKENS,
                 coded_tail: bool = True, shared_stream: bool = True):
        """``shared_stream`` False: the full shared expert on the current stream after the gather (benchmarks)."""
        from amd_kernels.collectives.pull_collectives_ranks2 import (
            AllGatherComm,
            FencedAllGatherComm,
            NormAllGatherComm,
            Pools,
            SlotsReduceScatterComm,
        )
        from amd_kernels.collectives.pull_collectives_ranks2.pull_unpack_q4096_shuffled import FENCE_MAX_TOKENS

        max_tokens = min(max(max_tokens, MIN_TOKENS), MAX_TOKENS)
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.max_tokens = max_tokens
        self.rows = shard_rows(max_tokens)
        self.padded = WORLD * self.rows
        # the transport warm-up's larger unpack (FENCE_MAX_TOKENS // 2 + 32 rows per rank)
        a2a_tokens = max(self.padded, FENCE_MAX_TOKENS + WORLD * 32)
        self.pools = Pools(rank, WORLD, cpu_group, device, _round_up(a2a_tokens * ROW_BYTES, 256), sets=1)
        self.a2a = FencedAllGatherComm(self.pools)
        # the widest field set: one bf16 [rows, H] (aux / final output); the MoE payload is 3296 B/row in four 256 B
        # aligned fields (7328 in six with a shard copy of the shared expert)
        self.ag = AllGatherComm(rank, cpu_group, device, _round_up(self.rows * 2 * H + 6 * 256, 256))
        self.rs = SlotsReduceScatterComm(rank, cpu_group, device, max_rows=self.padded, coded=coded_tail)
        self.norm_gather = NormAllGatherComm(rank, cpu_group, device, max_rows=self.rows)
        # the full shared expert's stream (``attention_moe``)
        self.side = torch.cuda.Stream(device=device) if shared_stream else None
        for c in (self.a2a, self.ag, self.rs, self.norm_gather):
            c.warm_up()
        torch.accelerator.synchronize()

    def supports(self, num_tokens: int) -> bool:
        """Whether a ``num_tokens`` call fits the communicators."""
        return 0 < num_tokens <= self.max_tokens

    def warm_up(self, o_weight: torch.Tensor, o_scale: torch.Tensor, moe_weights=None) -> None:
        """Compile every compute variant the phases launch, before serving (module docstring): ``o_weight`` /
        ``o_scale`` a layer's preshuffled o_proj_full (``gemm.mxfp8_common``), ``moe_weights`` a MoE layer's
        ``MoeWeights``, with the full shared expert (``SharedWeights.full``; a shard copy warms the compact payload
        form) (the dispatch depends on shapes only, so one layer serves all)."""
        dev = self.device
        x = torch.zeros((MIN_ROWS, H), dtype=torch.bfloat16, device=dev)
        shard_compute("input", x)
        codes = torch.zeros((MIN_ROWS, WORLD * Q), dtype=_FP8, device=dev)
        sc = torch.full((MIN_ROWS, WORLD * Q // 32), 127, dtype=torch.uint8, device=dev)
        shard_compute("dense", codes, sc, o_weight, o_scale)
        pack(torch.zeros((ROW_ALIGN, Q), dtype=torch.bfloat16, device=dev))
        if moe_weights is not None:
            from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import mxfp4_moe, shared_expert_full

            w = torch.zeros(H, dtype=torch.bfloat16, device=dev)
            local, _ = shard_compute("moe", codes, sc, o_weight, o_scale, x, w, 1e-6, moe_weights=moe_weights)
            if _full_shared(moe_weights):
                # the consumers: the routed-only partials mode of the gathered payload, the full shared expert
                mxfp4_moe(local.routed(), moe_weights, return_partials=True)
                shared_expert_full(local.shared_codes, local.shared_scales, moe_weights.shared_weights)
            else:
                mxfp4_moe(local.compact(), moe_weights, return_partials=True)
        torch.accelerator.synchronize()

    def _gathered(self, rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty((WORLD * rows, cols), dtype=dtype, device=self.device)

    def input_gather(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's normed bf16 rows ``[rows, H]`` -> MXFP8 codes ``[2 rows, H]`` e4m3 + AITER-shuffled scales
        ``[2 rows, H / 32]`` of every token (the quantizer of the non-SP first layer, gathered)."""
        rows = x.shape[0]
        codes, scales = self._gathered(rows, H, _FP8), self._gathered(rows, H // 32, torch.uint8)
        q, s = shard_compute("input", x)
        self.ag.all_gather([codes, scales], [q, s])
        return codes, scales

    def _unpacked(self, attn_output: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's heads of every token ``[t, Q]`` -> the o_proj input of this rank's rows: codes ``[rows, 2 Q]``,
        scales in o_proj_full's GEMM layout."""
        from amd_kernels.collectives.pull_collectives_ranks2 import Collectives

        t, d = attn_output.shape
        padded = WORLD * rows
        assert d == Q and t <= padded, (attn_output.shape, rows)
        coll = Collectives(self.pools, 0)
        coll.bind(torch.cuda.current_stream(self.device))
        send = coll.alloc((padded, ROW_BYTES), torch.uint8)
        assert send is not None, "the all-to-all slot is sized for every call up to max_tokens"
        pack(attn_output, out=send[:t])
        send[t:].zero_()
        return self.a2a.all_to_all_unpack_shuffled(coll, send)

    def attention_moe(self, attn_output, o_weight, o_scale, residual, norm_weight, eps, moe_weights):
        """The o_proj, the post-attention norm and the MoE input preparation of this rank's rows (``residual`` its
        ``[rows, H]`` residual shard) -> (the routed-only ``MoeInput`` of every token, the new residual shard, the
        ``SharedPending`` full shared expert of this rank's rows). With a shard copy of the shared expert
        (``SharedWeights.full`` False): (``MoeInputCompact`` of every token, the residual shard, None)."""
        from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import MoeInput, MoeInputCompact

        rows = residual.shape[0]
        xq, xs = self._unpacked(attn_output, rows)
        local, r = shard_compute("moe", xq, xs, o_weight, o_scale, residual, norm_weight, eps, moe_weights=moe_weights)
        if not _full_shared(moe_weights):
            compact = local.compact()
            outs = [self._gathered(rows, t.shape[1], t.dtype) for t in compact.tensors()]
            self.ag.all_gather(outs, list(compact.tensors()))
            return MoeInputCompact(*outs), r, None
        prepared = torch.cuda.Event()
        prepared.record()
        outs = [self._gathered(rows, cols, dtype) for cols, dtype in _FP4]
        self.ag.all_gather(outs, list(local.routed().tensors()[:4]))
        shared = _fork_shared(self.side, local, moe_weights, prepared)
        return MoeInput(*outs), r, shared

    def attention_dense(self, attn_output, o_weight, o_scale, residual, norm_weight, eps):
        """The o_proj and the post-attention norm + MXFP8 quant of this rank's rows -> (codes ``[2 rows, H]``, scales,
        the new residual shard): the dense MLP's input of every token."""
        rows = residual.shape[0]
        codes, scales = self._gathered(rows, H, _FP8), self._gathered(rows, H // 32, torch.uint8)
        xq, xs = self._unpacked(attn_output, rows)
        h = shard_compute("dense", xq, xs, o_weight, o_scale)
        _, _, r = self.norm_gather.norm_all_gather(h, residual, norm_weight, eps, q_out=codes, scale_out=scales)
        return codes, scales, r

    def ffn_norm_gather(self, ffn_out, residual, norm_weight, eps):
        """The FFN output of every token (MoE: ``(slots [2 rows, 4, H], shared)``, ``shared`` the ``SharedPending`` of
        ``attention_moe`` or a [2 rows, H] partial; dense: the bf16 partial ``[2 rows, H]``) -> reduce-scatter, the
        next layer's input norm + MXFP8 quant on this rank's rows, gathered: (codes ``[2 rows, H]``, scales, the new
        residual shard)."""
        if isinstance(ffn_out, tuple):
            ffn_out = _ffn_parts(ffn_out)
        rows = residual.shape[0]
        codes, scales = self._gathered(rows, H, _FP8), self._gathered(rows, H // 32, torch.uint8)
        if isinstance(ffn_out, tuple):
            q, s, r = self.rs.slots4_reduce_scatter_add_rmsnorm_mxfp8(*ffn_out, residual, norm_weight, eps)
            self.ag.all_gather([codes, scales], [q, s])
        else:
            _, _, r = self.norm_gather.norm_all_gather(
                self.rs.reduce_scatter(ffn_out), residual, norm_weight, eps, q_out=codes, scale_out=scales
            )
        return codes, scales, r

    def ffn_reduce_scatter(self, ffn_out) -> torch.Tensor:
        """The FFN output of every token (as ``ffn_norm_gather``) -> this rank's summed bf16 rows ``[rows, H]``."""
        if isinstance(ffn_out, tuple):
            return self.rs.slots4_reduce_scatter(*_ffn_parts(ffn_out))
        return self.rs.reduce_scatter(ffn_out)

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's bf16 rows ``[rows, H]`` -> every token's ``[2 rows, H]``."""
        out = self._gathered(x.shape[0], x.shape[1], x.dtype)
        self.ag.all_gather([out], [x])
        return out
