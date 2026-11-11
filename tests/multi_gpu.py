# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Run a test body on 4 GPUs: one process per GPU, a gloo process group (a TCPStore on a free local port: AITER's
custom all-reduce exchanges its IPC handles through a pure TCP store). The 4 rank processes start at the first call
and serve every later call of the same pytest process (a call of its own would pay for 4 new ones: Python, torch, a
HIP context, the group and the body's warm-up again). A failed call kills them; the next call starts new ones. A body
sees what earlier calls of its file left in the process (as it already saw their FlyDSL disk cache). The body
appends mismatches to ``errors`` instead of raising, so every rank reaches the same collectives; an exception is
recorded and ends that rank."""

import atexit
import os
import queue
import socket
import time
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4


CPU_THREADS = 8  # per rank: 4 ranks x all cores of torch's default spin on each other
EXIT_GRACE_S = 10  # after one rank failed: how long the others get to report before all are killed


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}  # pytest sets one per test


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(rank, port, jobs, results):
    torch.set_num_threads(CPU_THREADS)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD
    )
    parent = os.getppid()
    while True:
        try:
            job = jobs.get(timeout=5)
        except queue.Empty:
            if os.getppid() != parent:  # the caller died without closing us
                os._exit(1)
            continue
        if job is None:
            break
        fn, args = job
        errors = []
        try:
            fn(rank, errors, *args)
        except Exception:  # noqa: BLE001
            errors.append(f"rank {rank}: {traceback.format_exc()[-4000:]}")
            results.put((rank, errors, False))
            results.close()
            results.join_thread()  # the queue's feeder thread must flush before _exit
            os._exit(1)  # the peers may wait in a collective: the caller kills them
        dist.barrier()
        results.put((rank, errors, True))
    dist.barrier()
    dist.destroy_process_group()


class _Ranks:
    """The 4 rank processes, one job queue each, one result queue."""

    def __init__(self):
        ctx = mp.get_context("spawn")
        self.env = _env()
        self.results = ctx.Queue()
        self.jobs = [ctx.Queue() for _ in range(WORLD)]
        port = _free_port()
        self.procs = [
            ctx.Process(target=_serve, args=(r, port, self.jobs[r], self.results))
            for r in range(WORLD)
        ]
        for p in self.procs:
            p.start()

    def run(self, fn, args, timeout):
        """{rank: (errors, ok)} of the ranks that reported before the timeout."""
        for q in self.jobs:
            q.put((fn, args))
        done = {}
        deadline = time.monotonic() + timeout
        while len(done) < WORLD and time.monotonic() < deadline:
            try:
                rank, errors, ok = self.results.get(timeout=1)
                done[rank] = (errors, ok)
                if not ok:
                    deadline = min(deadline, time.monotonic() + EXIT_GRACE_S)
            except queue.Empty:
                if any(not p.is_alive() for r, p in enumerate(self.procs) if r not in done):
                    deadline = min(deadline, time.monotonic() + EXIT_GRACE_S)
        return done

    def close(self, kill):
        if not kill:
            for q in self.jobs:
                q.put(None)
            for p in self.procs:
                p.join(30)
        for p in self.procs:
            if p.is_alive():
                p.kill()
                p.join()


_ranks = None


@atexit.register
def _close_ranks():
    global _ranks
    if _ranks is not None:
        _ranks.close(kill=False)
        _ranks = None


def run_on_4_gpus(fn, *args, timeout=900):
    """``fn(rank, errors, *args)`` on 4 GPUs; fails on any recorded error, an
    exception, a rank that exits or a timeout."""
    global _ranks
    if torch.cuda.device_count() < WORLD:
        pytest.skip("needs 4 GPUs")
    if _ranks is not None and _ranks.env != _env():
        _close_ranks()  # the ranks run with the environment they started with
    if _ranks is None:
        _ranks = _Ranks()
    done = _ranks.run(fn, args, timeout)
    msgs = [m for r in sorted(done) for m in done[r][0]]
    ok = len(done) == WORLD and all(o for _, o in done.values()) and not msgs
    if not ok:
        codes = [p.exitcode for p in _ranks.procs]
        _ranks.close(kill=True)
        _ranks = None
        missing = [r for r in range(WORLD) if r not in done]
        assert not missing, f"ranks {missing} did not finish in {timeout} s (exit codes {codes}); {msgs[:4]}"
        assert not msgs, "\n".join(msgs[:8])
