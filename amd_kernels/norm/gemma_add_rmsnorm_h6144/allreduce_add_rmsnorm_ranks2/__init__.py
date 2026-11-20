# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The two-rank all-reduce fused in front of the residual add + Gemma RMSNorm (bf16 or MXFP8 out), FlyDSL (gfx950):
``staged_comm`` (the attention tail), ``slots4_comm`` (the MoE combine tail) and ``DecoderArNorm`` (both, and a decoder
layer's FFN between them); the API of ``allreduce_add_rmsnorm`` at two ranks."""

import dataclasses
import functools

import torch


def _expandable_segments() -> bool:
    """PyTorch's allocator hands out expandable segments (no ``hipIpcGetMemHandle`` on them)."""
    try:
        from aiter.dist.device_communicators.custom_all_reduce import _expandable_segments_enabled

        return _expandable_segments_enabled()
    except ImportError:
        import os

        conf = ",".join(
            os.environ.get(v, "") for v in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
        )
        return "expandable_segments:true" in conf.replace(" ", "").lower()


@functools.cache
def unsupported_reason(world_size: int, hidden_size: int = 6144) -> str | None:
    """None when this process can run the communicators, else why not: 2 ranks, hidden 6144, gfx950 with FlyDSL."""
    from .. import add_rmsnorm

    if world_size != 2:
        return f"needs 2 ranks (got {world_size})"
    if hidden_size != 6144:
        return f"needs hidden size 6144 (got {hidden_size})"
    if not add_rmsnorm.available():
        return "needs gfx950 with FlyDSL"
    if _expandable_segments():
        return "not with PyTorch expandable segments (the staging buffers are shared by IPC handle)"
    return None


def staged_comm(rank: int, cpu_group, device: torch.device, buffers=None):
    """The attention tail's communicator (``host.StagedArNormComm``: one bf16 partial per rank, bf16 or MXFP8 out; a
    collective on the CPU group, once per rank): to ``host.LIMIT_ROWS`` rows, eager or captured. Call ``warm_up`` on
    every rank before serving."""
    from .host import StagedArNormComm

    return StagedArNormComm(rank, cpu_group, device, buffers)


def slots4_comm(rank: int, cpu_group, device: torch.device, buffers=None):
    """The MoE combine tail's communicator (``host.SlotsArNormComm``; a collective on the CPU group, once per rank):
    eager and graph calls alike; call its ``warm_up`` on every rank before serving."""
    from .host import SlotsArNormComm

    return SlotsArNormComm(rank, cpu_group, device, buffers)


@dataclasses.dataclass
class DenseMlp:
    """A dense layer's MLP for ``DecoderArNorm.post_attention``: preshuffled MXFP8 gate_up [12288, 6144] and down
    [6144, 6144] with their scales, and SwiGLU-OAI's alpha, beta, limit."""

    gate_up: torch.Tensor
    gate_up_scale: torch.Tensor
    down: torch.Tensor
    down_scale: torch.Tensor
    alpha: float
    beta: float
    limit: float

    def __post_init__(self):
        if not dense_mlp_supported(self.gate_up, self.down):
            raise ValueError(f"dense MLP weights {tuple(self.gate_up.shape)} / {tuple(self.down.shape)} not supported")


def _dense_package():
    """``decoder.dense_mxfp8_h6144_i6144_ranks2``, or None where it is not installed (``post_attention`` then returns
    gate_up's codes for the caller's MLP)."""
    try:
        from amd_kernels.decoder import dense_mxfp8_h6144_i6144_ranks2
    except ImportError:
        return None
    return dense_mxfp8_h6144_i6144_ranks2


def dense_mlp_supported(gate_up: torch.Tensor, down: torch.Tensor) -> bool:
    """Whether ``post_attention`` takes a dense layer's MLP on these weights."""
    D = _dense_package()
    if D is not None:
        return D.mlp_supported(gate_up, down)
    return (tuple(gate_up.shape) == (12288, 6144) and tuple(down.shape) == (6144, 6144)
            and gate_up.dtype == down.dtype == torch.float8_e4m3fn)


@dataclasses.dataclass
class FfnPartial:
    """A decoder layer's FFN output before its TP all-reduce, which the next ``pre_attention`` runs: the MoE's four
    weighted slots [M, 4, 6144] + shared expert [M, 6144], or one bf16 partial [M, 6144]."""

    slots: torch.Tensor | None = None
    shared: torch.Tensor | None = None
    partial: torch.Tensor | None = None


class DecoderArNorm:
    """Both two-rank norm tails of a decoder layer on one stream (sharing one buffer pair): the staged
    communicator (one bf16 partial per rank) and the slots4 one (four weighted MoE slots + the shared expert), built
    and warmed up on every rank (a collective on the CPU group); with ``moe`` (one layer's ``MoeWeights`` of
    ``moe.mxfp4_h6144_i1536_e128_top4``) also the MoE modes ``post_attention`` runs."""

    def __init__(self, rank: int, cpu_group, device: torch.device, weight: torch.Tensor, eps: float, *, moe=None):
        from .host import SharedBuffers

        buffers = SharedBuffers(rank, cpu_group, device)
        self.staged = staged_comm(rank, cpu_group, device, buffers)
        self.slots = slots4_comm(rank, cpu_group, device, buffers)
        self.staged.warm_up(weight, eps)
        self.slots.warm_up(weight, eps)
        if moe is not None:
            from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import warm_up_mxfp4_moe

            warm_up_mxfp4_moe(moe, partials=True)
        torch.accelerator.synchronize(device)

    def capture(self):
        """Enter around every graph capture of the group (the four-rank API; nothing to register)."""
        return self.staged.capture()

    def supports(self, rows: int) -> bool:
        return self.staged.supports(rows) and self.slots.supports(rows)

    def accepts(self, x: torch.Tensor, residual: torch.Tensor) -> bool:
        return self.staged.accepts(x, residual)

    def add_rmsnorm(self, residual, weight, eps, *, partial=None, slots=None, shared=None, gemm_weight=None):
        """all-reduce(``partial``, or ``slots`` summed + ``shared``) + residual -> Gemma RMSNorm: (out, residual).

        ``out`` is bf16 without ``gemm_weight``, else (codes, scales) in the layout that K 6144 GEMM reads at this row
        count."""
        if slots is not None:
            assert gemm_weight is not None, "the slots4 tail writes MXFP8 only"
            q, s, residual = self.slots.slots4_ar_add_rmsnorm_mxfp8(
                slots, shared, residual, weight, eps, gemm_weight=gemm_weight)
            return (q, s), residual
        if gemm_weight is None:
            return self.staged.ar_add_rmsnorm(partial, residual, weight, eps)
        q, s, residual = self.staged.ar_add_rmsnorm_mxfp8(partial, residual, weight, eps, gemm_weight=gemm_weight)
        return (q, s), residual

    def post_attention(self, o_partial, residual, weight, eps, ffn, *, final=False):
        """all-reduce(``o_partial``) + residual -> Gemma RMSNorm -> ``ffn`` on this rank (``MoeWeights`` with its
        shared expert, or ``DenseMlp``): (``FfnPartial``, residual) for the next ``pre_attention`` (``final``: the one
        without a GEMM, bf16 out). None outside the communicators' rows; above the MLP's rows or without the dense MLP
        package, (gate_up's codes, scales) for the caller's MLP."""
        rows = o_partial.shape[0]
        if not (self.accepts(o_partial, residual) and self.supports(rows)):
            return None
        if isinstance(ffn, DenseMlp):
            D = _dense_package()
            (q, s), residual = self.add_rmsnorm(residual, weight, eps, partial=o_partial, gemm_weight=ffn.gate_up)
            if D is None or rows > D.MAX_ROWS:
                return (q, s), residual
            out = D.mxfp8_swiglu_oai_mlp(q, s, rows, ffn.gate_up, ffn.gate_up_scale, ffn.down, ffn.down_scale,
                                         ffn.alpha, ffn.beta, ffn.limit, form="fp32")
            return FfnPartial(partial=out), residual
        from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import mxfp4_moe

        x, residual = self.add_rmsnorm(residual, weight, eps, partial=o_partial)
        if final:  # the slots4 tail writes MXFP8 only
            return FfnPartial(partial=mxfp4_moe(x, ffn)), residual
        slots, shared = mxfp4_moe(x, ffn, return_partials=True)
        return FfnPartial(slots=slots, shared=shared), residual

    def pre_attention(self, ffn: FfnPartial, residual, weight, eps, gemm_weight=None):
        """all-reduce(``ffn``) + residual -> Gemma RMSNorm: ``add_rmsnorm``'s (out, residual)."""
        return self.add_rmsnorm(residual, weight, eps, partial=ffn.partial, slots=ffn.slots, shared=ffn.shared,
                                gemm_weight=gemm_weight)
