# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Skeleton kernels: the memory dependence chain of ``qkv_a_norm_kv_insert`` and ``indexer_prep`` (same grids, same
loads and stores, the same cache hints) with the arithmetic replaced by an xor of what was loaded. A skeleton is the
empirical floor of its kernel: every byte it moves and every load -> store dependence is one the kernel needs."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def _x(a, b):
    return Vec.from_elements([fx.Int32(a[i]) ^ fx.Int32(b[i]) for i in range_constexpr(4)], fx.Int32)


def _build_qkv_a(index: bool):
    nt = 384 if index else 256

    @flyc.kernel(name="skeleton_qkv_a" + ("_index" if index else ""), known_block_size=[nt, 1, 1])
    def kernel(
        arg_x: fx.Int64, arg_qw: fx.Int64, arg_kw: fx.Int64, arg_q: fx.Int64, arg_kv: fx.Int64,
        arg_cache: fx.Int64, arg_slot: fx.Int64, arg_wo: fx.Int64, arg_ic: fx.Int64, i32_n: fx.Int32,
        i32_row_st: fx.Int32, i64_page_b: fx.Int64,
    ):  # fmt: skip
        lane = fx.thread_idx.x % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.thread_idx.x // 64))
        row = fx.block_idx.x
        is_kv = wave == 3
        if wave < 4:
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x + fx.Int64(row) * fx.Int64(i32_row_st * 2), num_records_bytes=4224
            )
            wr = buffer_ops.create_buffer_resource_from_addr(is_kv.select(arg_kw, arg_qw), num_records_bytes=3072)
            sl_r = buffer_ops.create_buffer_resource_from_addr(arg_slot, num_records_bytes=fx.Int64(i32_n) * 8)
            slot = fx.Int32(buffer_ops.buffer_load(sl_r, row * 2, vec_width=1, dtype=fx.Int32, is_scalar=True))
            pe = buffer_ops.buffer_load(xr, 1024 + lane * 4, vec_width=4, dtype=fx.Int32, mask=is_kv & (lane < 8))
            u = [
                buffer_ops.buffer_load(xr, (is_kv.select(fx.Int32(192), fx.Int32(64 * i)) + lane) * 4, vec_width=4,
                                       dtype=fx.Int32, mask=None if i == 0 else wave < 3)
                for i in range_constexpr(3)
            ]  # fmt: skip
            w = buffer_ops.buffer_load(wr, is_kv.select(lane, wave * 64 + lane) * 4, vec_width=4, dtype=fx.Int32)
            y = _x(_x(_x(u[0], u[1]), u[2]), w)
            out = buffer_ops.create_buffer_resource_from_addr(
                is_kv.select(arg_kv + fx.Int64(row) * 1024, arg_q + fx.Int64(row) * 3072), num_records_bytes=3072
            )
            buffer_ops.buffer_store(y, out, is_kv.select(lane, wave * 64 + lane) * 4)
            ok = (slot >= 0) & is_kv
            cr = buffer_ops.create_buffer_resource_from_addr(
                arg_cache + fx.Int64(slot // 64) * i64_page_b + fx.Int64((slot % 64) * 656), num_records_bytes=656
            )
            buffer_ops.buffer_store(Vec.from_elements([y[0], y[1]], fx.Int32), cr, lane * 2, mask=ok)
            buffer_ops.buffer_store(y[2], cr, 128 + lane // 16, mask=ok & ((lane % 16) == 0))
            buffer_ops.buffer_store(pe, cr, 132 + lane * 4, mask=ok & (lane < 8))
        if index:
            if wave == 4:
                kr = buffer_ops.create_buffer_resource_from_addr(
                    arg_x + fx.Int64(row) * fx.Int64(i32_row_st * 2) + 4224, num_records_bytes=256
                )
                v = fx.Int32(buffer_ops.buffer_load(kr, lane, vec_width=1, dtype=fx.Int32))
                sl_r = buffer_ops.create_buffer_resource_from_addr(arg_slot, num_records_bytes=fx.Int64(i32_n) * 8)
                s2 = fx.Int32(buffer_ops.buffer_load(sl_r, row * 2, vec_width=1, dtype=fx.Int32, is_scalar=True))
                cr = buffer_ops.create_buffer_resource_from_addr(
                    arg_ic + fx.Int64(s2 // 64) * i64_page_b, num_records_bytes=4352
                )
                buffer_ops.buffer_store(v.to(fx.Int8), cr, (lane // 16) * 1024 + (s2 % 64) * 16 + lane % 16,
                                        mask=s2 >= 0, offset_is_bytes=True)  # fmt: skip
                buffer_ops.buffer_store(v.to(fx.Int8), cr, 4096 + (lane // 16) * 64 + s2 % 64,
                                        mask=(s2 >= 0) & ((lane % 16) == 0), offset_is_bytes=True)  # fmt: skip
            if wave == 5:
                if lane < 32:
                    wr2 = buffer_ops.create_buffer_resource_from_addr(
                        arg_x + fx.Int64(row) * fx.Int64(i32_row_st * 2) + 4480, num_records_bytes=64
                    )
                    v = fx.Int16(buffer_ops.buffer_load(wr2, lane, vec_width=1, dtype=T.i16)).to(fx.Int32)
                    wo = buffer_ops.create_buffer_resource_from_addr(arg_wo, num_records_bytes=fx.Int64(i32_n) * 128)
                    buffer_ops.buffer_store(v.bitcast(fx.Float32), wo, row * 32 + lane)

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_qw: fx.Int64, arg_kw: fx.Int64, arg_q: fx.Int64, arg_kv: fx.Int64,
               arg_cache: fx.Int64, arg_slot: fx.Int64, arg_wo: fx.Int64, arg_ic: fx.Int64, i32_n: fx.Int32,
               i32_row_st: fx.Int32, i64_page_b: fx.Int64, stream: fx.Stream):  # fmt: skip
        kernel(arg_x, arg_qw, arg_kw, arg_q, arg_kv, arg_cache, arg_slot, arg_wo, arg_ic, i32_n, i32_row_st,
               i64_page_b).launch(grid=(i32_n, 1, 1), block=(nt, 1, 1), stream=stream)  # fmt: skip

    return launch


def _build_indexer(keys: bool):
    @flyc.kernel(name="skeleton_indexer" + ("" if keys else "_queries"), known_block_size=[256, 1, 1])
    def kernel(
        arg_q: fx.Int64, arg_k: fx.Int64, arg_w: fx.Int64, arg_slot: fx.Int64, arg_cache: fx.Int64,
        arg_qp: fx.Int64, arg_qs: fx.Int64, arg_wo: fx.Int64, arg_out: fx.Int64, i32_n: fx.Int32,
        i32_nout: fx.Int32, i64_page_b: fx.Int64,
    ):  # fmt: skip
        tx = fx.thread_idx.x
        row = fx.block_idx.x
        lane = tx % 64
        if row < i32_n:
            if tx < 128:
                qr = buffer_ops.create_buffer_resource_from_addr(arg_q + fx.Int64(row) * 8192, num_records_bytes=8192)
                v = [buffer_ops.buffer_load(qr, tx * 16 + i * 4, vec_width=4, dtype=fx.Int32) for i in range_constexpr(4)]
                w = _x(_x(v[0], v[1]), _x(v[2], v[3]))
                qp = buffer_ops.create_buffer_resource_from_addr(arg_qp, num_records_bytes=fx.Int64(i32_n) * 2048)
                buffer_ops.buffer_store(w, qp, row * 512 + tx * 4)
                qs = buffer_ops.create_buffer_resource_from_addr(arg_qs, num_records_bytes=fx.Int64(i32_n) * 256)
                buffer_ops.buffer_store(fx.Int32(w[0]).to(fx.Int8), qs, row * 256 + tx, offset_is_bytes=True)
            if keys:
                if tx >= 192:
                    if lane < 32:
                        wr = buffer_ops.create_buffer_resource_from_addr(arg_w + fx.Int64(row) * 64, num_records_bytes=64)
                        v = fx.Int16(buffer_ops.buffer_load(wr, lane, vec_width=1, dtype=T.i16)).to(fx.Int32)
                        wo = buffer_ops.create_buffer_resource_from_addr(arg_wo, num_records_bytes=fx.Int64(i32_n) * 128)
                        buffer_ops.buffer_store(v.bitcast(fx.Float32), wo, row * 32 + lane)
                if tx >= 128:
                    if tx < 192:
                        sl_r = buffer_ops.create_buffer_resource_from_addr(arg_slot, num_records_bytes=fx.Int64(i32_n) * 8)
                        slot = fx.Int32(buffer_ops.buffer_load(sl_r, row * 2, vec_width=1, dtype=fx.Int32, is_scalar=True))
                        kr = buffer_ops.create_buffer_resource_from_addr(arg_k + fx.Int64(row) * 256, num_records_bytes=256)
                        v = fx.Int32(buffer_ops.buffer_load(kr, lane, vec_width=1, dtype=fx.Int32))
                        cr = buffer_ops.create_buffer_resource_from_addr(
                            arg_cache + fx.Int64(slot // 64) * i64_page_b, num_records_bytes=4352
                        )
                        buffer_ops.buffer_store(v.to(fx.Int8), cr, (lane // 16) * 1024 + (slot % 64) * 16 + lane % 16,
                                                mask=slot >= 0, offset_is_bytes=True)  # fmt: skip
                        buffer_ops.buffer_store(v.to(fx.Int8), cr, 4096 + (lane // 16) * 64 + slot % 64,
                                                mask=(slot >= 0) & ((lane % 16) == 0), offset_is_bytes=True)  # fmt: skip
        r = row - i32_n
        if (r >= 0) & (r < i32_nout):
            o = buffer_ops.create_buffer_resource_from_addr(arg_out + fx.Int64(r) * 8192, num_records_bytes=8192)
            neg = Vec.from_elements([fx.Int32(-1)] * 4, fx.Int32)
            for i in range_constexpr(2):
                buffer_ops.buffer_store(neg, o, (i * 256 + tx) * 4, cache_modifier=2)

    @flyc.jit
    def launch(arg_q: fx.Int64, arg_k: fx.Int64, arg_w: fx.Int64, arg_slot: fx.Int64, arg_cache: fx.Int64,
               arg_qp: fx.Int64, arg_qs: fx.Int64, arg_wo: fx.Int64, arg_out: fx.Int64, i32_n: fx.Int32,
               i32_nout: fx.Int32, i64_page_b: fx.Int64, i32_grid: fx.Int32, stream: fx.Stream):  # fmt: skip
        kernel(arg_q, arg_k, arg_w, arg_slot, arg_cache, arg_qp, arg_qs, arg_wo, arg_out, i32_n, i32_nout,
               i64_page_b).launch(grid=(i32_grid, 1, 1), block=(256, 1, 1), stream=stream)  # fmt: skip

    return launch


def _run(exe, *args):
    if getattr(exe, "_cf", None) is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        exe._cf(*args)


_CACHE = {}


def skeleton_qkv_a(x, qw, kw, q_out, kv_out, cache, slots, weights=None, index_cache=None):
    index = index_cache is not None
    exe = _CACHE.setdefault(("qkv", index), _build_qkv_a(index))
    _run(exe, x.data_ptr(), qw.data_ptr(), kw.data_ptr(), q_out.data_ptr(), kv_out.data_ptr(), cache.data_ptr(),
         slots.data_ptr(), weights.data_ptr() if index else 0, index_cache.data_ptr() if index else 0, x.shape[0],
         x.stride(0), cache.stride(0), torch.cuda.current_stream())  # fmt: skip


def skeleton_indexer(q, k, w, slots, cache, q_payload, q_scales, weights, topk_out):
    keys = k is not None
    exe = _CACHE.setdefault(("ix", keys), _build_indexer(keys))
    n, n_out = q.shape[0], topk_out.shape[0]
    _run(exe, q.data_ptr(), k.data_ptr() if keys else 0, w.data_ptr() if keys else 0, slots.data_ptr() if keys else 0,
         cache.data_ptr(), q_payload.data_ptr(), q_scales.data_ptr(), weights.data_ptr() if keys else 0,
         topk_out.data_ptr(), n, n_out, cache.stride(0), n + n_out, torch.cuda.current_stream())  # fmt: skip
