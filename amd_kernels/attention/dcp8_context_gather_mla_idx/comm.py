# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host communicator for the DCP8 direct context gather."""

import functools

import torch
import torch.distributed as dist

from amd_kernels.common.mono.peer import PeerBuffer
from amd_kernels.common.tensor_shim import _run_compiled

from .kernel import DATA_OFF, WORLD, compile_context_push, compile_exchange_barrier


_exchange_barrier = functools.cache(compile_exchange_barrier)


@functools.cache
def _push_kernel(mla_row_bytes, index_row_bytes, index_scale_bytes, tiled_fp4):
    return compile_context_push(
        mla_row_bytes,
        index_row_bytes,
        index_scale_bytes,
        synchronize=True,
        tiled_fp4=tiled_fp4,
    )


class DcpContextGather:
    """One uncached workspace per rank, directly writable by every peer."""

    def __init__(self, nbytes: int, group, rank: int, device: torch.device):
        assert dist.get_world_size(group) == WORLD
        self.rank = rank
        self.group = group
        self.device = torch.device(device)
        self.nbytes = nbytes
        self.peer = PeerBuffer(DATA_OFF + nbytes, group, rank, WORLD, self.device)
        self.data = self.peer.bytes[DATA_OFF:]
        self.data_ptrs = self.peer.addresses + DATA_OFF
        self.exchange_epoch = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.state = torch.zeros(2, dtype=torch.int32, device=self.device)
        # PeerBuffer memory is not guaranteed zeroed.  Only flags require an
        # initial value; data rows are fully populated before consumption.
        self.peer.bytes[:DATA_OFF].zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def gather(
        self,
        *,
        mla_cache: torch.Tensor,
        mla_block_table: torch.Tensor,
        index_cache: torch.Tensor,
        index_block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        dst_block_table: torch.Tensor,
        max_local_pages: int,
        dst_index_offset: int,
        index_row_bytes: int,
        tiled_fp4: bool = False,
    ) -> None:
        if not max_local_pages or not seq_lens.numel():
            return
        mla_u8 = mla_cache.view(torch.uint8)
        index_u8 = index_cache.view(torch.uint8)
        index_scale_bytes = index_u8.shape[-1] - index_row_bytes
        kernel = _push_kernel(
            mla_u8.shape[-1], index_row_bytes, index_scale_bytes, tiled_fp4
        )
        _run_compiled(
            kernel,
            mla_u8.data_ptr(),
            mla_u8.stride(0),
            mla_u8.shape[1] // 64,
            mla_block_table.data_ptr(),
            mla_block_table.stride(0),
            index_u8.data_ptr(),
            index_u8.stride(0),
            index_block_table.data_ptr(),
            index_block_table.stride(0),
            seq_lens.data_ptr(),
            query_start_loc.data_ptr(),
            dst_block_table.data_ptr(),
            dst_block_table.stride(0),
            *self.peer.host_addresses,
            self.nbytes,
            dst_index_offset,
            self.rank,
            max_local_pages,
            seq_lens.numel(),
            self.state.data_ptr(),
            torch.cuda.current_stream(self.device),
        )

    def barrier(self) -> None:
        _run_compiled(
            _exchange_barrier(),
            *self.peer.host_addresses,
            self.rank,
            self.exchange_epoch.data_ptr(),
            torch.cuda.current_stream(self.device),
        )

    def close(self) -> None:
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.peer.close()
