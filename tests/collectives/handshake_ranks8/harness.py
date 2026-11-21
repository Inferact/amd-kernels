# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Run a body on 8 GPUs: one spawned process per GPU and a gloo group; ``fn(rank, errors, *args)`` appends mismatches
to ``errors`` instead of raising so every rank reaches the same collectives."""

import os
import queue
import socket
import time
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
EXIT_GRACE_S = 10


def _serve(rank, port, fn, args, results):
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    errors = []
    ok = True
    try:
        fn(rank, errors, *args)
        dist.barrier()
    except Exception:  # noqa: BLE001
        errors.append(f"rank {rank}: {traceback.format_exc()[-4000:]}")
        ok = False
    results.put((rank, errors, ok))
    results.close()
    results.join_thread()
    if not ok:
        os._exit(1)
    dist.destroy_process_group()


def run_on_8_gpus(fn, *args, timeout=240):
    if torch.cuda.device_count() < WORLD:
        pytest.skip("needs 8 GPUs")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    procs = [ctx.Process(target=_serve, args=(r, port, fn, args, results)) for r in range(WORLD)]
    for p in procs:
        p.start()
    done = {}
    deadline = time.monotonic() + timeout
    while len(done) < WORLD and time.monotonic() < deadline:
        try:
            rank, errors, ok = results.get(timeout=1)
            done[rank] = (errors, ok)
            if not ok:
                deadline = min(deadline, time.monotonic() + EXIT_GRACE_S)
        except queue.Empty:
            if any(not p.is_alive() for r, p in enumerate(procs) if r not in done):
                deadline = min(deadline, time.monotonic() + EXIT_GRACE_S)
    for p in procs:
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
    msgs = [m for r in sorted(done) for m in done[r][0]]
    missing = [r for r in range(WORLD) if r not in done]
    assert not missing, f"ranks {missing} did not finish; {msgs[:4]}"
    assert not msgs, "\n".join(msgs[:8])
