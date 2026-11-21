# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The traffic floor of ``kda_decode_fb_gate_spec``: the same grid, table lookups, state read and per-token snapshot
writes (non-temporal), no math. A workgroup per (sequence, head) reads its 64 KB state and writes it to each token's slot."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

D, NH = 128, 12


def compile_skeleton(max_tok, waves=8, lpr=8, store_cm=2, split=1):
    rg = 64 // lpr
    rows_wg = D // split
    rpl = rows_wg // (waves * rg)
    nvec = D // lpr // 4
    block = 64 * waves

    @flyc.kernel(name=f"kda_decode_traffic_floor_t{max_tok}_w{waves}_r{split}", known_block_size=[block, 1, 1])
    def kernel(arg_cu: fx.Int64, arg_idx: fx.Int64, i32_idx_stride: fx.Int32, arg_acc: fx.Int64, arg_state: fx.Int64,
               i64_slot_bytes: fx.Int64, arg_out: fx.Int64):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        grp = bid // split
        rs = bid % split
        i_n = grp // NH
        hv = grp % NH
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        s = lane % lpr
        rgi = lane // lpr

        def sload(rsrc, off):
            return fx.Int32(buffer_ops.buffer_load(rsrc, off, vec_width=1, is_scalar=True))

        out_r = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=1 << 22)
        cu_r = buffer_ops.create_buffer_resource_from_addr(arg_cu)
        idx_r = buffer_ops.create_buffer_resource_from_addr(arg_idx)
        acc_r = buffer_ops.create_buffer_resource_from_addr(arg_acc)
        n_tok = sload(cu_r, i_n + 1) - sload(cu_r, i_n)
        n_acc = sload(acc_r, i_n)
        slots = []
        for h in range_constexpr((max_tok + 3) // 4):
            sv = Vec(buffer_ops.buffer_load(idx_r, i_n * i32_idx_stride + 4 * h, vec_width=4, is_scalar=True))
            slots += [fx.Int32(sv[k]) for k in range_constexpr(4)]

        def pick(vals, i):
            r = vals[len(vals) - 1]
            for k in range_constexpr(len(vals) - 2, -1, -1):
                r = (i == k).select(vals[k], r)
            return r

        slot0 = pick(slots, (n_acc > 1).select(n_acc - 1, fx.Int32(0)))
        v0 = rs * rows_wg + w * (rg * rpl) + rgi * rpl
        soff = [[(v0 + i) * D + 4 * (s + lpr * m) for m in range_constexpr(nvec)] for i in range_constexpr(rpl)]
        hv64 = fx.Int64(hv)
        srcr = buffer_ops.create_buffer_resource_from_addr(
            arg_state + fx.Int64(slot0) * i64_slot_bytes + hv64 * (D * D * 4), num_records_bytes=D * D * 4)
        st = [Vec(buffer_ops.buffer_load(srcr, soff[i][m], vec_width=4, dtype=fx.Float32))
              for i in range_constexpr(rpl) for m in range_constexpr(nvec)]
        acc = st[0][0]
        for x in st[1:]:
            acc = acc + x[1]
        for t in range_constexpr(max_tok):
            slot_t = pick(slots, fx.Int32(t))
            dstr = buffer_ops.create_buffer_resource_from_addr(
                arg_state + fx.Int64(slot_t) * i64_slot_bytes + hv64 * (D * D * 4),
                num_records_bytes=((fx.Int32(t) < n_tok) & (slot_t > 0)).select(fx.Int32(D * D * 4), fx.Int32(0)))
            for i in range_constexpr(rpl):
                for m in range_constexpr(nvec):
                    buffer_ops.buffer_store(st[i * nvec + m], dstr, soff[i][m], cache_modifier=store_cm)
        buffer_ops.buffer_store(acc, out_r, bid * 512 + tx)

    @flyc.jit
    def launch(arg_cu: fx.Int64, arg_idx: fx.Int64, i32_idx_stride: fx.Int32, arg_acc: fx.Int64, arg_state: fx.Int64,
               i64_slot_bytes: fx.Int64, arg_out: fx.Int64, i32_nseq: fx.Int32, stream: fx.Stream):
        kernel(arg_cu, arg_idx, i32_idx_stride, arg_acc, arg_state, i64_slot_bytes, arg_out).launch(
            grid=(fx.Int64(i32_nseq) * (NH * split), 1, 1), block=(block, 1, 1), stream=stream)

    return launch


def skeleton_calls(sets, max_tok, waves=8, lpr=8, split=1):
    """One closure per input set, each launching the floor kernel on that set's tables and state."""
    exe = compile_skeleton(max_tok, waves, lpr, 2, split)
    out = torch.zeros((1 << 20) + 4096, dtype=torch.float32, device="cuda")
    calls = []
    for a in sets:
        args = (a["cu_seqlens"].data_ptr(), a["state_indices"].data_ptr(), a["state_indices"].stride(0),
                a["num_accepted"].data_ptr(), a["state"].data_ptr(), a["state"].stride(0) * 4, out.data_ptr(),
                a["cu_seqlens"].numel() - 1)
        cf = flyc.compile(exe, *args, torch.cuda.current_stream())
        calls.append(lambda cf=cf, args=args: cf(*args, torch.cuda.current_stream()))
    return calls
