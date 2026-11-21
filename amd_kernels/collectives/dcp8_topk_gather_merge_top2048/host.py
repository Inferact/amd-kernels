# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One graph-safe eight-rank communicator per stream, constructed collectively."""
import ctypes
import functools
import torch
import torch.distributed as dist
from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4.host import _alloc_pool, _exchange, _open_pool, _run
from amd_kernels.common import hip_runtime as ops
from .kernel import K, MAX_ROWS, SLOT_BYTES, WORLD, compile_topk_gather_merge


@functools.cache
def _exe():
    return compile_topk_gather_merge()


class TopkGatherMerge:
    def __init__(self, rank, group, device):
        assert dist.get_world_size(group) == WORLD and rank == dist.get_rank(group)
        self.rank, self.group, self.device = rank, group, torch.device(device)
        ptr, handle, self.pool = _alloc_pool(2 * SLOT_BYTES, self.device)
        self.pool.zero_()
        torch.cuda.synchronize(self.device)
        handles = _exchange(handle, rank, WORLD, group)
        self.ptrs = [ptr if r == rank else _open_pool(h, self.device) for r, h in enumerate(handles)]
        self.ctl = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.counts = torch.zeros(MAX_ROWS, dtype=torch.int64, device=self.device)
        self.dbg = torch.zeros(MAX_ROWS * 32, dtype=torch.int64, device=self.device)
        self.closed = False

    def __call__(self, scores, local_idx, block_table, out_slots, out_indptr, page_size, block_stride_rows):
        """Gather, merge and emit this rank's CSR slots of the global top-2048.

        scores fp32 [rows, 2048] (padding -inf) and local_idx int32 [rows, 2048] (-1 = none) are this rank's local
        top-2048; block_table int32 [rows, blocks] is each row's page table; out_slots int32 holds >= rows * 2048 and
        out_indptr int32 >= rows + 1. All ranks call in the same order with the same rows (<= 256). The caller
        supplies the outputs, so capture allocates nothing.
        """
        assert not self.closed
        rows = scores.shape[0]
        assert 0 < rows <= MAX_ROWS and scores.shape == (rows, K) and scores.dtype == torch.float32 and scores.stride(1) == 1
        assert local_idx.shape == (rows, K) and local_idx.dtype == torch.int32 and local_idx.stride(1) == 1
        assert block_table.ndim == 2 and block_table.shape[0] == rows and block_table.dtype == torch.int32
        assert block_table.stride(1) == 1
        assert out_slots.dtype == out_indptr.dtype == torch.int32 and out_slots.numel() >= rows * K and out_indptr.numel() > rows
        assert all(t.device == self.device for t in (scores, local_idx, block_table, out_slots, out_indptr))
        _run(_exe(), scores.data_ptr(), local_idx.data_ptr(), block_table.data_ptr(), out_slots.data_ptr(),
             out_indptr.data_ptr(), self.ctl.data_ptr(), self.counts.data_ptr(), self.dbg.data_ptr(), *self.ptrs,
             rows, self.rank, scores.stride(0), local_idx.stride(0), block_table.stride(0), page_size, block_stride_rows,
             torch.cuda.current_stream(self.device))
        return out_slots, out_indptr

    def warmup(self):
        """Collective: compile the single variant (rows are a runtime argument); every rank calls it before capture."""
        d = self.device
        s = torch.full((1, K), float("-inf"), device=d)
        i = torch.full((1, K), -1, dtype=torch.int32, device=d)
        self(s, i, torch.zeros(1, 1, dtype=torch.int32, device=d), torch.empty(K, dtype=torch.int32, device=d),
             torch.empty(2, dtype=torch.int32, device=d), 64, 64)
        torch.cuda.synchronize(d)

    def close(self):
        """Collective teardown after all graphs using this object are retired."""
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        lib = ops.load_hip_runtime()
        lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        lib.hipFree.argtypes = [ctypes.c_void_p]
        for r in range(WORLD):
            if r != self.rank:
                assert lib.hipIpcCloseMemHandle(ctypes.c_void_p(self.ptrs[r])) == 0
        dist.barrier(group=self.group)
        assert lib.hipFree(ctypes.c_void_p(self.ptrs[self.rank])) == 0
        self.pool = None
        self.closed = True
