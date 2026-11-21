# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One launch: the chunk workgroups (``lists.py``) followed by one merge workgroup (``merge.py``) a row.

Workgroups are dispatched in index order, so every chunk workgroup is resident or done before the first merge
workgroup starts; a merge workgroup waits for the ready count of its (sequence, row tile) (the chunk workgroups store
their lists write-through and count themselves in at agent scope), then reads the lists with coherent loads. The
roles share one LDS block (the larger of the two).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.warm_up import launcher

from ._wave import I, I64, carve
from .lists import NW, chunk_parts
from .merge import NT, merge_parts


PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


@launcher
def compile_fused(rows: int, blocks: int, timing: bool = False):
    """``rows``: query rows a chunk workgroup scores of a sequence; ``blocks``: the merge kernel's pool blocks."""
    assert NW * 64 == NT
    name = f"mxfp4_paged_score_topk_d128_h32_r{rows}_b{blocks}" + ("_t" if timing else "")
    chunk_role, CH_OFFS, CH_WORDS = chunk_parts(rows, timing, sync=True)
    merge_role, MG_OFFS, MG_WORDS = merge_parts(blocks, timing, sync=True)

    @fx.struct
    class Raw:
        u: fx.Array[fx.Int32, max(CH_WORDS, MG_WORDS)]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(arg_qsl: fx.Int64, arg_bt: fx.Int64, arg_ends: fx.Int64, arg_q: fx.Int64, arg_w: fx.Int64,
               arg_qs: fx.Int64, arg_kv: fx.Int64, i32_batch: fx.Int32, i32_nch_grid: fx.Int32, arg_kvs: fx.Int64,
               arg_keys: fx.Int64, arg_idx: fx.Int64, arg_meta: fx.Int64, arg_full: fx.Int64, arg_cnt: fx.Int64,
               arg_oidx: fx.Int64, arg_oval: fx.Int64, i32_bt_stride: fx.Int32, i32_page_stride: fx.Int32,
               i32_last_page: fx.Int32, i32_qlen: fx.Int32, i32_uniform: fx.Int32, i32_target: fx.Int32, i32_oidx_stride: fx.Int32,
               i32_oval_stride: fx.Int32, i32_tiles: fx.Int32, i32_tile_rows: fx.Int32, i32_nk1: fx.Int32,
               arg_dbg: fx.Int64):
        raw = fx.SharedAllocator().allocate(Raw).peek().u
        bid = I(fx.block_idx.x)
        if bid < i32_nk1:
            chunk_role(carve(raw, **CH_OFFS), arg_q, arg_qs, arg_kv, arg_kvs, arg_bt, arg_w, arg_ends, arg_qsl, arg_keys,
                       arg_idx, arg_meta, arg_full, i32_bt_stride, i32_page_stride, i32_last_page, i32_batch,
                       i32_nch_grid, i32_qlen, i32_target, i32_uniform, arg_cnt, i32_tiles, arg_dbg)
        else:
            merge_role(carve(raw, **MG_OFFS), bid - i32_nk1, arg_qsl, arg_keys, arg_idx, arg_meta, arg_full, arg_oidx,
                       arg_oval, i32_batch, i32_oidx_stride, i32_oval_stride, i32_qlen, i32_uniform, arg_cnt, i32_tiles,
                       i32_tile_rows, i32_nch_grid, arg_dbg + I64(4 << 20))

    @flyc.jit
    def launch_fused(arg_qsl: fx.Int64, arg_bt: fx.Int64, arg_ends: fx.Int64, arg_q: fx.Int64, arg_w: fx.Int64,
                     arg_qs: fx.Int64, arg_kv: fx.Int64, i32_batch: fx.Int32, i32_nch_grid: fx.Int32, arg_kvs: fx.Int64,
                     arg_keys: fx.Int64, arg_idx: fx.Int64, arg_meta: fx.Int64, arg_full: fx.Int64, arg_cnt: fx.Int64,
                     arg_oidx: fx.Int64, arg_oval: fx.Int64, i32_bt_stride: fx.Int32, i32_page_stride: fx.Int32,
                     i32_last_page: fx.Int32, i32_qlen: fx.Int32, i32_uniform: fx.Int32, i32_target: fx.Int32, i32_oidx_stride: fx.Int32,
                     i32_oval_stride: fx.Int32, i32_tiles: fx.Int32, i32_tile_rows: fx.Int32, i32_nk1: fx.Int32,
                     arg_dbg: fx.Int64, grid_x: fx.Int32, stream: fx.Stream):
        kernel(arg_qsl, arg_bt, arg_ends, arg_q, arg_w, arg_qs, arg_kv, i32_batch, i32_nch_grid, arg_kvs, arg_keys,
               arg_idx, arg_meta, arg_full, arg_cnt, arg_oidx, arg_oval, i32_bt_stride, i32_page_stride, i32_last_page,
               i32_qlen, i32_uniform, i32_target, i32_oidx_stride, i32_oval_stride, i32_tiles, i32_tile_rows, i32_nk1,
               arg_dbg).launch(grid=(I64(grid_x), 1, 1), block=(NT, 1, 1), stream=stream)

    launch_fused.kernel_name = name
    launch_fused.compile_hints = PRELOAD_HINTS  # the first 16 kernel argument dwords arrive in SGPRs
    return launch_fused
