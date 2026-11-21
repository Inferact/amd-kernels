# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Communicator, weight layout and entry of the DCP8 LSE combine + W_UV up-projection (see ``kernel.py``).

One communicator per stream. Create it on every rank before graph capture (a collective), and call
``dcp8_lse_combine_vup`` on every rank in the same order. Inputs and output may not alias the pools.
"""

import functools

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from .kernel import CTL_BYTES, compile_combine
from .reference import H, HPR, L, V, WORLD

MAX_TOKENS = 192


@functools.cache
def _kernel(m_tiles, base_e, gated):
    return compile_combine(m_tiles, base_e, gated)


def _exchange(obj, rank, group):
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    for i, r in enumerate(sorted(dist.get_process_group_ranks(group=group))):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


def prepare_w_uv(w_uv: torch.Tensor) -> torch.Tensor:
    """``W_UV`` of all 96 heads, [96, 512, 128] (head-major, DCP-rank-major), as the kernel reads it: [96, 128, 512]."""
    assert w_uv.shape == (H, L, V) and w_uv.dtype == torch.bfloat16
    return w_uv.transpose(1, 2).contiguous()


class DcpVupCombine:
    """Pools, control blocks and the device table of the 8 ranks (a collective: every rank of ``group`` creates one).

    The pools and control blocks are uncached device memory mapped on the peers (``hipDeviceMallocUncached`` through
    IPC), so a peer's stores are never shadowed by a stale line of this rank's L2.
    """

    def __init__(self, rank, group, device, max_tokens=MAX_TOKENS, is_lse_base_on_e=True):
        assert 1 <= max_tokens <= 256 and dist.get_world_size(group) == WORLD
        self.rank, self.device, self.max_tokens = rank, torch.device(device), max_tokens
        self.base_e = is_lse_base_on_e
        self.slot_bytes = -(-WORLD * max_tokens * HPR * (256 + 4) // 256) * 256
        pool, pool_h = ops.allocate_shared_buffer_and_handle(2 * self.slot_bytes)
        ctl, ctl_h = ops.allocate_shared_buffer_and_handle(CTL_BYTES)
        peers = _exchange((pool_h, ctl_h), rank, group)
        pools = [pool if r == rank else ops.open_mem_handle(h[0]) for r, h in enumerate(peers)]
        ctls = [ctl if r == rank else ops.open_mem_handle(h[1]) for r, h in enumerate(peers)]
        self.table = torch.tensor(pools + ctls, dtype=torch.int64, device=self.device)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def _launch(self, x, lse, w_uv_t, out, seq_lens, query_start_loc, stream, gate=None):
        m = x.shape[0]
        assert 0 < m <= self.max_tokens
        assert x.dtype == torch.bfloat16 and x.shape[1] >= H and x.shape[2] == L and x.stride(2) == 1
        assert x.stride(0) % 2 == 0 and x.stride(1) % 2 == 0
        assert lse.dtype == torch.float32 and lse.shape[0] == m and lse.shape[1] >= H
        assert w_uv_t.shape == (H, V, L) and w_uv_t.dtype == torch.bfloat16 and w_uv_t.is_contiguous()
        assert out.shape == (m, HPR, V) and out.dtype == torch.bfloat16 and out.is_contiguous()
        if gate is not None:
            assert gate.shape == (m, HPR * V) and gate.dtype == torch.bfloat16 and gate.is_contiguous()
        if seq_lens is None:
            seq_p = qsl_p = lse.data_ptr()
            nseq = 0
        else:
            assert seq_lens.dtype == query_start_loc.dtype == torch.int32
            seq_p, qsl_p, nseq = seq_lens.data_ptr(), query_start_loc.data_ptr(), seq_lens.shape[0]
        _run_compiled(
            _kernel(-(-m // 16), self.base_e, gate is not None),
            x.data_ptr(), x.stride(0) // 2, x.stride(1) // 2, lse.data_ptr(), lse.stride(0), lse.stride(1),
            w_uv_t.data_ptr(), seq_p, qsl_p, nseq, out.data_ptr(),
            (out if gate is None else gate).data_ptr(), self.table.data_ptr(), m, self.rank,
            self.max_tokens, self.slot_bytes, stream,
        )


def _combine(comm, attn_out, lse, w_uv_t, out, seq_lens=None, query_start_loc=None, gate=None):
    """Write this rank's [T, 12, 128] heads of the combined, up-projected attention output into ``out``.

    ``attn_out`` [T, >=96, 512] bf16 and ``lse`` [T, >=96] fp32 are this rank's partials of all 96 heads; ``w_uv_t`` is
    ``prepare_w_uv``'s layout. ``seq_lens`` / ``query_start_loc`` (int32) mask the LSE of empty shards and padding rows.
    With ``gate`` ([T, 1536] bf16) the result is ``bf16(bf16(combine) * sigmoid(gate))``.
    """
    if attn_out.shape[0]:
        comm._launch(attn_out, lse, w_uv_t, out, seq_lens, query_start_loc, torch.cuda.current_stream(comm.device), gate)
    return out


def _mode(comm, attn_out, lse, w_uv_t, out, seq_lens=None, query_start_loc=None, gate=None):
    return (gate is not None, comm.max_tokens)


def _sizes(mode):
    return [min(16 * t, mode[1]) for t in range(1, -(-mode[1] // 16) + 1)]


def _inputs(mode, n, comm, attn_out, lse, w_uv_t, out, seq_lens=None, query_start_loc=None, gate=None):
    dev = attn_out.device
    x = torch.zeros((n, attn_out.shape[1], L), dtype=attn_out.dtype, device=dev)
    return (
        comm,
        x,
        torch.zeros((n, lse.shape[1]), dtype=torch.float32, device=dev),
        w_uv_t,
        torch.zeros((n, HPR, V), dtype=out.dtype, device=dev),
        seq_lens,
        query_start_loc,
        torch.zeros((n, HPR * V), dtype=out.dtype, device=dev) if mode[0] else None,
    ), {}


dcp8_lse_combine_vup = entry(
    "attention/dcp8_lse_combine_vup_mla_h12_kv512", mode=_mode, sizes=_sizes, inputs=_inputs
)(_combine)
