# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the layer tail: the launch, and the per-rank state (scratch, epoch, health words, peer memory)."""

from dataclasses import dataclass

import torch

from amd_kernels import warm_up
from amd_kernels.common.mono.abi import PEER_HEAD
from amd_kernels.common.mono.peer import PeerBuffer
from amd_kernels.common.mono.plan import HEALTH_WORDS

from . import kernel as K
from .kernel import K2Build, build

TP = K.TP_


def _run_compiled(launcher, *args) -> None:
    import flydsl.compiler as flyc
    from flydsl.utils import env

    if env.compile.compile_only:
        flyc.compile(launcher, *args)
        return
    compiled = getattr(launcher, "_cf", None)
    if compiled is not None:
        compiled(*args)
        return
    launcher._cf = flyc.compile(launcher, *args)


def _ptr(t):
    return 0 if t is None else t.data_ptr()


class LayerTailRunner:
    """Collective over the TP8 group: every rank builds it at the same point.

    ``scratch`` is shared by every launch of a step (their mailbox tags differ); ``min_scratch`` is the bytes a
    caller's other launches need, so the tagged-row region lies past them.
    """

    def __init__(self, device, group, rank, min_scratch: int = 0):
        self.rank = rank
        nbytes = max(K.tag_end(), min_scratch, K.scratch_bytes(K2Build(tokens=K.k2b.MAX_S, nblocks=8)))
        self.scratch = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)
        self.peers = PeerBuffer(PEER_HEAD + K.extra_peer_end(), group, rank, TP, device)
        self.peers.bytes.zero_()
        lo = PEER_HEAD + K.sent_offset()
        self.peers.bytes[lo : lo + K.sent_bytes()].fill_(0xFF)
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=group)
        table = self.peers.addresses
        self.peer_table = (table.repeat(TP) if table.numel() == 1 else table) + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * rank
        self.flags = self.peers.bytes[: 4 * TP].view(torch.int32)

    def step_begin(self) -> None:
        """Once a forward, before its first launch (a CUDA graph captures it)."""
        self.epoch.add_(1)

    def launch_args(self) -> dict:
        return {"scratch": self.scratch, "epoch": self.epoch, "health": self.health, "alarm": self.alarm}

    def peer_args(self) -> dict:
        return {"peers": self.peer_table, "rank": self.rank}

    def close(self) -> None:
        self.peers.close()


def k2_launch(
    key: K2Build,
    *,
    core,
    w_o,
    prefix,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_gate,
    bias,
    w_ld,
    w_sgu,
    w_sd,
    w13,
    w13s,
    w2,
    w2s,
    ln_w,
    w_up,
    out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    health,
    alarm,
    tl=None,
):
    """``core`` [S, 1536] (o_proj's input), ``prefix`` updated in place (written when ``key.reset``), ``out`` the
    MoE output [S, 7168]."""
    f = build(key)
    _run_compiled(
        f,
        core.data_ptr(),
        w_o.data_ptr(),
        prefix.data_ptr(),
        blocks.data_ptr(),
        blocks.stride(0),
        blocks.stride(1),
        ares_nw.data_ptr(),
        ares_qk.data_ptr(),
        in_nw.data_ptr(),
        w_gate.data_ptr(),
        bias.data_ptr(),
        w_ld.data_ptr(),
        w_sgu.data_ptr(),
        w_sd.data_ptr(),
        w13.data_ptr(),
        w13s.data_ptr(),
        w2.data_ptr(),
        w2s.data_ptr(),
        ln_w.data_ptr(),
        w_up.data_ptr(),
        out.data_ptr(),
        scratch.data_ptr(),
        peers.data_ptr(),
        rank,
        epoch.data_ptr(),
        layer,
        health.data_ptr(),
        alarm,
        _ptr(tl),
        torch.cuda.current_stream(),
    )


@dataclass
class LayerTailWeights:
    """A layer's weights as vLLM loads them (no copies, no re-quantization): ``w_o`` the rank's o_proj columns
    [7168, 1536]; ``w13`` / ``w2`` AITER's shuffled MXFP4 experts with ``w13s`` / ``w2s`` their E8M0 scales;
    ``w_up`` this rank's [896, 3584] rows of the latent up projection."""

    w_o: torch.Tensor
    ares_nw: torch.Tensor
    ares_qk: torch.Tensor
    in_nw: torch.Tensor
    w_gate: torch.Tensor
    bias: torch.Tensor
    w_ld: torch.Tensor
    w_sgu: torch.Tensor
    w_sd: torch.Tensor
    w13: torch.Tensor
    w13s: torch.Tensor
    w2: torch.Tensor
    w2s: torch.Tensor
    ln_w: torch.Tensor
    w_up: torch.Tensor
    eps: float = 1e-5
    out_eps: float = 1e-5
    ln_eps: float = 1e-5


SERVED_ROWS = (4, 6, 8, 12, 16)


def _mode(comm, w, core, prefix, blocks, *, nblocks, reset=False, layer, out=None):
    return (nblocks, bool(reset))


def _sizes(mode):
    return SERVED_ROWS


def _inputs(mode, n, comm, w, core, prefix, blocks, *, nblocks, reset=False, layer, out=None):
    z = torch.zeros
    blocks = z((n, *blocks.shape[1:]), dtype=blocks.dtype, device=blocks.device)
    return (comm, w, z((n, core.size(1)), dtype=core.dtype, device=core.device),
            z((n, prefix.size(1)), dtype=prefix.dtype, device=prefix.device), blocks), {
        "nblocks": nblocks, "reset": reset, "layer": layer}  # fmt: skip


@warm_up.entry("moe.layer_tail_k2_h7168_ranks8", mode=_mode, sizes=_sizes, inputs=_inputs)
def layer_tail(comm, w, core, prefix, blocks, *, nblocks, reset=False, layer, out=None):
    """One layer's tail on this rank: ``core`` [S, 1536] (o_proj's input) -> the MoE output [S, 7168] (``out``).
    ``prefix`` [S, 7168] is updated in place (written when ``reset``, a block-write layer); ``blocks`` [S, B, 7168]
    holds the AttnRes block sources, ``nblocks`` the layer's valid ones before this layer's own block. ``layer`` is
    the layer index; the caller bumps ``comm.step_begin()`` once before the first launch of a step."""
    out = torch.empty_like(prefix) if out is None else out
    key = K2Build(
        tokens=core.size(0), nblocks=nblocks + int(reset), reset=reset, eps=w.eps, out_eps=w.out_eps, ln_eps=w.ln_eps
    )
    k2_launch(
        key,
        core=core,
        w_o=w.w_o,
        prefix=prefix,
        blocks=blocks,
        ares_nw=w.ares_nw,
        ares_qk=w.ares_qk,
        in_nw=w.in_nw,
        w_gate=w.w_gate,
        bias=w.bias,
        w_ld=w.w_ld,
        w_sgu=w.w_sgu,
        w_sd=w.w_sd,
        w13=w.w13,
        w13s=w.w13s,
        w2=w.w2,
        w2s=w.w2s,
        ln_w=w.ln_w,
        w_up=w.w_up,
        out=out,
        layer=128 + layer,
        **comm.launch_args(),
        **comm.peer_args(),
    )
    return out
