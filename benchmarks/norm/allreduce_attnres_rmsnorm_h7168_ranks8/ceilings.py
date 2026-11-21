# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Peer-memory ceilings behind the exchange protocol, in the kernel's own primitives (8 ranks, uncached mailboxes):
``--mode ping`` the round trip of a 16 B token between rank 0 and a peer (one-way latency = half), with one workgroup
or many pinging at once; ``--mode bw`` the push bandwidth of one rank to one and to seven peers; ``--mode wave`` the
time one wave takes to issue and to complete stores to a peer; ``--mode read`` the time a workgroup takes to read
1 KiB-per-wave-load batches from its own mailbox memory (system-coherent loads) split over 1..7 waves.
"""
import argparse
import functools
import json

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from amd_kernels.common import buffer_ops
from amd_kernels.common import hip_runtime as ops
from amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8.host import _DevicePointer
from amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8.ipc import _run_compiled, exchange
from amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8.norm_math import (
    SYS, _global_ptr, compiler_barrier, load_ptrs, sel8, vstore,
)
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from tests.norm.allreduce_attnres_rmsnorm_h7168_ranks8.multi_gpu8 import run_on_8_gpus


def _now():
    return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _vload(addr):
    return Vec(_llvm.LoadOp(T.vec(4, T.i32), _global_ptr(addr), nontemporal=True, alignment=16).result)


@flyc.kernel(known_block_size=[64, 1, 1])
def ping_kernel(arg_my: fx.Int64, arg_peer: fx.Int64, arg_out: fx.Int64, i32_iters: fx.Int32, i32_role: fx.Int32):
    """Workgroup w pings through its own 256 B window; lane 0 of workgroup 0 times the round trips."""
    tx, wg = fx.thread_idx.x, fx.block_idx.x
    my = arg_my + fx.Int64(wg) * 256
    peer = arg_peer + fx.Int64(wg) * 256
    if tx == 0:
        t0 = _now()
        for i in range(fx.Int32(0), i32_iters, fx.Int32(1)):
            tok = Vec.from_elements([i + 1, i + 1, i + 1, i + 1], fx.Int32)
            if i32_role == 0:
                vstore(peer, tok)
            v = _vload(my)
            while fx.Int32(v[0]) != i + 1:
                compiler_barrier()
                v = _vload(my)
            if i32_role != 0:
                vstore(peer, tok)
        t1 = _now()
        if wg == 0:
            buffer_ops.buffer_store(fx.Int32(t1 - t0), buffer_ops.create_buffer_resource_from_addr(
                arg_out, num_records_bytes=16), 0)


@flyc.jit
def launch_ping(arg_my: fx.Int64, arg_peer: fx.Int64, arg_out: fx.Int64, i32_iters: fx.Int32, i32_role: fx.Int32,
                i32_wgs: fx.Int32, stream: fx.Stream):
    ping_kernel(arg_my, arg_peer, arg_out, i32_iters, i32_role).launch(grid=(fx.Int64(i32_wgs), 1, 1), block=(64, 1, 1),
                                                                       stream=stream)


@flyc.kernel(known_block_size=[512, 1, 1])
def push_kernel(arg_table: fx.Int64, i32_me: fx.Int32, i32_iters: fx.Int32, i32_dests: fx.Int32):
    """Workgroup w streams 16 B stores into its own contiguous window of peer (me + 1 + w % dests) % 8."""
    tx, wg = fx.thread_idx.x, fx.block_idx.x
    bs = load_ptrs(arg_table)
    d = (i32_me + 1 + wg % i32_dests) % 8
    base = sel8(d, bs) + fx.Int64((wg // i32_dests) * i32_iters * 8192) + fx.Int64(tx) * 16
    v = Vec.from_elements([tx, tx, tx, tx], fx.Int32)
    for i in range(fx.Int32(0), i32_iters, fx.Int32(1)):
        vstore(base + fx.Int64(i) * 8192, v)


@flyc.jit
def launch_push(arg_table: fx.Int64, i32_me: fx.Int32, i32_iters: fx.Int32, i32_dests: fx.Int32, i32_wgs: fx.Int32,
                stream: fx.Stream):
    push_kernel(arg_table, i32_me, i32_iters, i32_dests).launch(grid=(fx.Int64(i32_wgs), 1, 1), block=(512, 1, 1),
                                                               stream=stream)


@flyc.kernel(known_block_size=[512, 1, 1])
def wave_kernel(arg_table: fx.Int64, arg_out: fx.Int64, i32_me: fx.Int32, i32_waves: fx.Int32, i32_stores: fx.Int32,
                i32_local: fx.Int32):
    """One workgroup; wave w < waves issues ``stores`` 1 KiB stores to peer (me + 1) % 8 (or to itself if local) and
    records the time it took to issue them and until their completion (vmcnt 0)."""
    tx = fx.thread_idx.x
    wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
    lane = tx % 64
    bs = load_ptrs(arg_table)
    dest = i32_local.select(i32_me, (i32_me + 1) % 8)
    base = sel8(dest, bs) + fx.Int64(wave) * (1 << 20) + fx.Int64(lane) * 16
    v = Vec.from_elements([tx, tx, tx, tx], fx.Int32)
    if wave < i32_waves:
        t0 = _now()
        for i in range(fx.Int32(0), i32_stores, fx.Int32(1)):
            vstore(base + fx.Int64(i) * 1024, v)
        t1 = _now()
        fx.rocdl.s_waitcnt(vmcnt=0)
        t2 = _now()
        if lane == 0:
            ro = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=64)
            buffer_ops.buffer_store(fx.Int32(t1 - t0), ro, wave * 2)
            buffer_ops.buffer_store(fx.Int32(t2 - t0), ro, wave * 2 + 1)


@flyc.jit
def launch_wave(arg_table: fx.Int64, arg_out: fx.Int64, i32_me: fx.Int32, i32_waves: fx.Int32, i32_stores: fx.Int32,
                i32_local: fx.Int32, stream: fx.Stream):
    wave_kernel(arg_table, arg_out, i32_me, i32_waves, i32_stores, i32_local).launch(grid=(1, 1, 1), block=(512, 1, 1),
                                                                                     stream=stream)


@functools.cache
def make_read(loads: int, cache: int):
    """Wave w < waves reads ``loads`` 1 KiB blocks (16 B per lane, all issued before the first use) at its own
    offsets of a 1 MiB region, ``reps`` times, and records the mean time of a batch (until the data is consumed)."""

    @flyc.kernel(known_block_size=[512, 1, 1])
    def read_kernel(arg_my: fx.Int64, arg_out: fx.Int64, i32_waves: fx.Int32, i32_reps: fx.Int32):
        tx = fx.thread_idx.x
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        lane = tx % 64
        rs = buffer_ops.create_buffer_resource_from_addr(arg_my, num_records_bytes=1 << 20)
        if wave < i32_waves:
            acc = fx.Int32(0)
            t0 = _now()
            for rep in range(fx.Int32(0), i32_reps, fx.Int32(1)):
                vs = [Vec(buffer_ops.buffer_load(rs, ((wave * 64 + rep % 8 * 8 + i) * 256 + lane * 4) , vec_width=4,
                                                 dtype=fx.Int32, cache_modifier=cache)) for i in range(loads)]
                for v in vs:
                    acc = acc ^ fx.Int32(v[0]) ^ fx.Int32(v[3])
                fx.rocdl.s_waitcnt(vmcnt=0)
            t1 = _now()
            if lane == 0:
                ro = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=64)
                buffer_ops.buffer_store(fx.Int32(t1 - t0), ro, wave * 2)
                buffer_ops.buffer_store(acc, ro, wave * 2 + 1)

    @flyc.jit
    def launch_read(arg_my: fx.Int64, arg_out: fx.Int64, i32_waves: fx.Int32, i32_reps: fx.Int32, i32_wgs: fx.Int32,
                    stream: fx.Stream):
        read_kernel(arg_my, arg_out, i32_waves, i32_reps).launch(grid=(fx.Int64(i32_wgs), 1, 1), block=(512, 1, 1),
                                                                 stream=stream)

    return launch_read


def worker(rank, errors, mode, iters):
    dev = torch.device("cuda", rank)
    ptr, handle = ops.allocate_shared_buffer_and_handle(64 << 20)
    peers = exchange(handle, rank, dist.group.WORLD)
    bufs = [ptr if r == rank else ops.open_mem_handle(peers[r]) for r in range(8)]
    torch.as_tensor(_DevicePointer(ptr, 64 << 20), device=dev).zero_()
    torch.cuda.synchronize()
    dist.barrier()
    out = torch.zeros(16, dtype=torch.int32, device=dev)
    stream = torch.cuda.current_stream()
    res = {}
    if mode == "ping":
        for wgs in (1, 16, 128):
            _run_compiled(launch_ping, ptr, bufs[(rank + 1) % 8], out.data_ptr(), 0, 0, wgs, stream)
            torch.cuda.synchronize()
            dist.barrier()
            for peer in (1, 5):
                dist.barrier()
                if rank in (0, peer):
                    torch.as_tensor(_DevicePointer(ptr, 128 * 256), device=dev).zero_()
                    torch.cuda.synchronize()
                dist.barrier()
                if rank in (0, peer):
                    _run_compiled(launch_ping, ptr, bufs[peer if rank == 0 else 0], out.data_ptr(), iters,
                                  0 if rank == 0 else 1, wgs, stream)
                    torch.cuda.synchronize()
                    if rank == 0:
                        res[f"workgroups={wgs} peer={peer} round trip us"] = round(int(out[0]) * 10 / iters / 1000, 3)
                dist.barrier()
    elif mode == "read":
        reps = 50
        for cache, cname in ((SYS, "sys"), (0, "cached")):
            for waves, loads, wgs in ((1, 14, 1), (2, 7, 1), (7, 2, 1), (7, 4, 1), (1, 14, 12), (7, 2, 12), (1, 14, 80),
                                      (7, 2, 80), (1, 2, 1), (1, 1, 1)):
                fn = make_read(loads, cache)
                for _ in range(2):
                    _run_compiled(fn, ptr, out.data_ptr(), waves, reps, wgs, stream)
                    torch.cuda.synchronize()
                oc = out.cpu()
                res[f"{cname} waves={waves} loads/wave={loads} workgroups={wgs}: us per batch"] = round(
                    int(oc[0]) * 0.01 / reps, 3)
    elif mode == "wave":
        table = torch.tensor(bufs, dtype=torch.int64, device=dev)
        for local in (1, 0):
            for waves, stores in ((1, 14), (1, 2), (2, 7), (7, 2), (1, 1), (1, 8)):
                for _ in range(2):
                    _run_compiled(launch_wave, table.data_ptr(), out.data_ptr(), rank, waves, stores, local, stream)
                    dist.barrier()
                    torch.cuda.synchronize()
                oc = out.cpu()
                res[f"{'local' if local else 'remote'} waves={waves} stores/wave={stores}: issue, complete us"] = [
                    round(int(oc[0]) * 0.01, 2), round(int(oc[1]) * 0.01, 2)]
    else:
        table = torch.tensor(bufs, dtype=torch.int64, device=dev)
        wgs = 224
        for dests in (1, 7):
            for _ in range(2):
                _run_compiled(launch_push, table.data_ptr(), rank, 32, dests, wgs, stream)
            dist.barrier()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            reps = 5
            a.record()
            for _ in range(reps):
                _run_compiled(launch_push, table.data_ptr(), rank, 32, dests, wgs, stream)
            b.record()
            b.synchronize()
            res[f"push to {dests} peer(s), GB/s per rank"] = round(
                reps * wgs * 512 * 16 * 32 / (a.elapsed_time(b) * 1e-3) / 1e9, 1)
    every = [None] * 8
    dist.all_gather_object(every, res)
    if rank == 0:
        print(json.dumps({"mode": mode, "rank0": every[0], "rank7": every[7]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("ping", "bw", "wave", "read"), default="ping")
    p.add_argument("--iters", type=int, default=1000)
    a = p.parse_args()
    run_on_8_gpus(worker, a.mode, a.iters, timeout=200)
