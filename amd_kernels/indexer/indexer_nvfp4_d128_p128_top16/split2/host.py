# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Buffers and launches of the decode indexer split over two GPUs (TP2, gfx950).

``IndexSplit2`` is made by a collective (both ranks at the same point, outside graph capture) and owns, in uncached
memory the peer maps: the index-q receive buffer [max_rows, 2, 128] e4m3, the partial top-16 receive buffer
[2, max_rows, 16] x (key, count, block, count) int32 and the flag words; plus device call counters (graph-safe) and
the half's ids scratch. A call is two launches: ``decode_score.py`` (scores of all four heads over this rank's blocks,
index-q exchanged inside), then ``select_merge.py`` (partial top 16, exchange, merge into this rank's two heads).
"""

import torch
import torch.distributed as dist
from amd_kernels.warm_up import entry

from .. import hip_ipc
from ..host_stream import current_stream as _current_stream
from ..reference import score_width
from . import kernel as _kernel

WORLD = _kernel.WORLD
# two workgroups per CU; page ranges to 1.25 runs per workgroup (180K: -4..-11% vs runs at 12 / 14 requests, 48 runs)
SCORE_GRID = 512
PAGES_RUNS = 640
# Tests / benchmarks: pin the scorer's grid and page-range bound (0: runs always).
PIN_GRID: int | None = None
PIN_PAGES_RUNS: int | None = None


def _run_compiled(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _end_bytes(t: torch.Tensor) -> int:
    return (sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1) * t.element_size()


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def _exchange(obj, rank: int, group) -> list:
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    for i, r in enumerate(sorted(dist.get_process_group_ranks(group=group))):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class IndexSplit2:
    """This rank's half of the two-GPU decode indexer for up to ``max_rows`` rows (``query_len`` per request, 4 or 8)
    and ``max_width`` score positions (``reference.score_width`` of the longest context)."""

    def __init__(self, rank: int, group, device: torch.device, max_rows: int, max_width: int, query_len: int):
        assert dist.get_world_size(group) == WORLD
        assert query_len in (4, 8), query_len
        assert max_rows <= _kernel.MAX_ROWS and max_rows // query_len <= _kernel.MAX_REQS, max_rows
        self.rank, self.device = rank, torch.device(device)
        self.max_rows, self.max_width, self.query_len = max_rows, max_width, query_len
        qbytes = max_rows * 2 * _kernel.D
        lbytes = 2 * max_rows * 16 * 16
        mine = [hip_ipc.allocate_shared_buffer_and_handle(n) for n in (qbytes, lbytes, _kernel.FLAG_BYTES)]
        peers = _exchange([h for _, h in mine], rank, group)
        other = 1 - rank
        self.q_ptrs = [mine[0][0], hip_ipc.open_mem_handle(peers[other][0])]
        self.l_ptrs = [mine[1][0], hip_ipc.open_mem_handle(peers[other][1])]
        self.f_ptrs = [mine[2][0], hip_ipc.open_mem_handle(peers[other][2])]
        torch.as_tensor(_DevicePointer(self.f_ptrs[0], _kernel.FLAG_BYTES), device=self.device).zero_()
        self.counts = torch.zeros(_kernel.CNT_WORDS, dtype=torch.int32, device=self.device)
        self.ids = torch.empty(_kernel.HEADS, max_rows, 16, dtype=torch.int32, device=self.device)
        torch.accelerator.synchronize(self.device)
        dist.barrier(group=group)

    def score_topk(
        self,
        iq: torch.Tensor,  # [rows, 2, 128] e4m3, this rank's index heads
        index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8 (replicated)
        block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
        seq_lens: torch.Tensor,  # [num_reqs] int32
        init_blocks: int,
        local_blocks: int,
        topk_idx: torch.Tensor,  # [2, rows, 16] int32, written
        max_seq_len: int,
    ) -> None:
        """This rank's two heads' decode top-16 block ids, bit for bit ``score_decode.nvfp4_index_score_top16`` on
        one GPU; both ranks call it with the same lengths. Graph-safe (lengths are read on the device)."""
        from ..score_prefill.host import nvfp4_e4m3_lut
        from .decode_score import compile_score_half
        from .select_merge import compile_select_merge

        rows = iq.shape[0]
        ql = self.query_len
        assert rows % ql == 0 and rows <= self.max_rows, (rows, ql, self.max_rows)
        assert iq.dtype == torch.float8_e4m3fn and iq.shape[1:] == (2, _kernel.D) and iq.is_contiguous()
        assert topk_idx.dtype == torch.int32 and topk_idx.shape[0] == 2 and topk_idx.shape[1] >= rows
        assert topk_idx.stride(2) == 1
        width = score_width(max_seq_len)
        assert width <= self.max_width, (width, self.max_width)
        num_reqs = rows // ql
        if num_reqs == 0:
            return
        half = max(64, width // 2)
        score = torch.empty(_kernel.HEADS, rows, half, dtype=torch.float32, device=iq.device)
        stream = _current_stream()
        _run_compiled(
            compile_score_half(ql), iq.data_ptr(), self.q_ptrs[0], self.q_ptrs[1], index_cache.data_ptr(),
            nvfp4_e4m3_lut(iq.device).data_ptr(), score.data_ptr(), block_table.data_ptr(), seq_lens.data_ptr(),
            self.counts.data_ptr(), self.f_ptrs[0], self.f_ptrs[1], self.max_rows, int(index_cache.stride(0)),
            int(score.stride(0)), int(score.stride(1)), _end_bytes(score), int(block_table.stride(0)),
            _end_bytes(block_table), num_reqs, int(init_blocks), int(local_blocks), self.rank,
            PIN_GRID or SCORE_GRID, PAGES_RUNS if PIN_PAGES_RUNS is None else PIN_PAGES_RUNS, stream,
        )
        _run_compiled(
            compile_select_merge(), score.data_ptr(), int(score.stride(0)), int(score.stride(1)), half,
            seq_lens.data_ptr(), ql, self.ids.data_ptr(), self.l_ptrs[0], self.l_ptrs[1], self.counts.data_ptr(),
            topk_idx.data_ptr(), int(topk_idx.stride(0)), int(topk_idx.stride(1)),
            self.max_rows, self.rank, rows, stream,
        )


def _mode(cp, iq, *args, **kwargs):
    return cp.query_len, cp.max_rows // cp.query_len, cp.max_width, iq.device.index


def _sizes(mode):
    _, max_reqs, _, _ = mode
    return sorted({min(n, max_reqs) for n in (1, 64, max_reqs)})


def _inputs(mode, n, cp, iq, index_cache, block_table, seq_lens, init_blocks, local_blocks, topk_idx, max_seq_len):
    ql, _, width, _ = mode
    rows, dev = n * ql, iq.device
    args = (
        cp,
        torch.zeros(rows, 2, 128, dtype=torch.float8_e4m3fn, device=dev),
        index_cache,
        torch.zeros(n, width, dtype=torch.int32, device=dev),
        torch.full((n,), ql, dtype=torch.int32, device=dev),
        init_blocks, local_blocks,
        torch.empty(2, rows, 16, dtype=torch.int32, device=dev),
        width * 128,
    )
    return args, {}


IndexSplit2.score_topk = entry(
    "indexer/nvfp4_index_score_top16_ranks2_d128_p128", mode=_mode, sizes=_sizes, inputs=_inputs,
)(IndexSplit2.score_topk)
