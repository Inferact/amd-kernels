# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Bandwidth of the one xGMI link of a GPU pair with CU kernels: a persistent grid of ``wgs`` workgroups copies
``mb`` MB from the peer (pull: system-coherent loads of the peer's pool, local stores) or to the peer (push: local
loads, stores into the peer's pool), one direction (rank 0 only) or both at once. Each workgroup moves ``unroll`` x 16 B
per thread per step. Prints GB/s per direction (event-timed, best of 5 rounds of 10 copies, max over ranks).

  python3 benchmarks/collectives/pull_collectives_ranks2/benchmark_link.py [mb] [--policies] [out.json]

``--policies``: the pull's load cache policy (``LOAD_POLICY``) instead of the workgroup / unroll sweep.
"""
import functools
import json
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from tests.multi_gpu import run_on_2_gpus

NT = 256
SYS = 0x11
# gfx950 cache policy bits of a load: sc0 = 1, nt = 2, sc1 = 16
LOAD_POLICY = {"pull": SYS, "pull_nt": SYS | 2, "pull_sc1": 16, "pull_plain": 0}


@functools.cache
def _copy_kernel(wgs: int, unroll: int, load_mod: int, store_mod: int):
    step = NT * 4 * unroll
    name = f"link_copy_w{wgs}_u{unroll}_l{load_mod}_s{store_mod}"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(arg_dst: fx.Int64, arg_src: fx.Int64, i32_bytes: fx.Int32):
        n_dw = i32_bytes // 4
        src = buffer_ops.create_buffer_resource_from_addr(arg_src, num_records_bytes=i32_bytes)
        dst = buffer_ops.create_buffer_resource_from_addr(arg_dst, num_records_bytes=i32_bytes)
        off = fx.block_idx.x * step + fx.thread_idx.x * 4
        while off < n_dw:
            vals = []
            for u in range_constexpr(unroll):
                o = off + u * NT * 4
                vals.append(Vec(buffer_ops.buffer_load(src, o, vec_width=4, dtype=fx.Int32, mask=o < n_dw,
                                                       cache_modifier=load_mod)))
            for u in range_constexpr(unroll):
                o = off + u * NT * 4
                buffer_ops.buffer_store(vals[u], dst, o, mask=o < n_dw, cache_modifier=store_mod)
            off = off + wgs * step

    @flyc.jit
    def launch(arg_dst: fx.Int64, arg_src: fx.Int64, i32_bytes: fx.Int32, stream: fx.Stream):
        kernel(arg_dst, arg_src, i32_bytes).launch(grid=(wgs, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch


def _worker(rank, errors, mb, configs, out):
    from amd_kernels.collectives.pull_collectives_ranks2.host import _alloc_pool, _exchange, _open_pool, _run_compiled

    dev = torch.device("cuda", rank)
    n = mb << 20
    ptr, handle, pool = _alloc_pool(2 * n, dev)
    hs = _exchange(handle, rank, 2, dist.group.WORLD)
    peer = 1 - rank
    peer_ptr = _open_pool(hs[peer], dev)
    local = torch.randint(0, 255, (n,), dtype=torch.uint8, device=dev)
    pool[:n].copy_(local)
    stream = torch.cuda.current_stream()
    results = {}
    for kind, wgs, unroll, both in configs:
        if kind.startswith("pull"):
            exe = _copy_kernel(wgs, unroll, LOAD_POLICY[kind], 0)
            dst, src = local.data_ptr(), peer_ptr
        elif kind == "push":
            exe = _copy_kernel(wgs, unroll, 0, 0)
            dst, src = peer_ptr + n, local.data_ptr()
        else:  # push with system-coherent stores
            exe = _copy_kernel(wgs, unroll, 0, SYS)
            dst, src = peer_ptr + n, local.data_ptr()
        active = both or rank == 0
        _run_compiled(exe, dst, src, n, stream)
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(5):
            dist.barrier()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            if active:
                for _ in range(10):
                    _run_compiled(exe, dst, src, n, stream)
            b.record()
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b) / 10)
        times = [None, None]
        dist.all_gather_object(times, best if active else 0.0)
        ms = max(times)
        key = f"{kind} wgs{wgs} u{unroll} {'bidir' if both else 'uni'}"
        results[key] = n / ms / 1e6
        if rank == 0:
            print(f"{key:32s} {n / ms / 1e6:7.1f} GB/s per direction ({ms * 1e3:.0f} us for {mb} MB)", flush=True)
    if rank == 0 and out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)


def main():
    args = sys.argv[1:]
    out = args.pop() if args and args[-1].endswith(".json") else None
    policies = "--policies" in args
    args = [a for a in args if a != "--policies"]
    mb = int(args[0]) if args else 256
    configs = []
    for both in (False, True):
        if policies:
            for kind in LOAD_POLICY:
                for wgs in (8, 16, 32):
                    configs.append((kind, wgs, 8, both))
            continue
        for kind in ("pull", "push", "push_sys"):
            for wgs in (4, 8, 16, 32, 64):
                for unroll in (4, 8, 16):
                    configs.append((kind, wgs, unroll, both))
    run_on_2_gpus(_worker, mb, configs, out, timeout=3000)


if __name__ == "__main__":
    main()
