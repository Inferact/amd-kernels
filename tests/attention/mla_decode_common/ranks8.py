# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the 8-rank entry tests share: the no-compile walk and the graph check, recorded into ``errors``."""

import importlib

import torch

from .cases import same_bytes, tensors


def walk_without_compiles(rank, errors, calls, tag):
    """Run ``calls`` (after the warm-ups) with ``compile_watch`` armed: no call may compile."""
    from amd_kernels import compile_watch

    late = []
    compile_watch.arm(late.append)
    before = compile_watch.late_compiles()
    for call in calls:
        call()
    torch.cuda.synchronize()
    if compile_watch.late_compiles() != before or late:
        errors.append(f"rank {rank} {tag}: {len(late)} compiles after the warm-up: {late[:2]}")
    compile_watch.disarm()


def graph_matches_eager(rank, errors, call, outputs, tag):
    """A graph of ``call()`` captured with ``importlib.import_module`` failing replays the eager result."""
    call()
    torch.cuda.synchronize()
    eager = [t.clone() for t in tensors(outputs)]
    for t in tensors(outputs):
        t.zero_()
    graph, real = torch.cuda.CUDAGraph(), importlib.import_module

    def no_import(name, package=None):
        raise AssertionError(f"import of {name} inside capture")

    importlib.import_module = no_import
    try:
        with torch.cuda.graph(graph):
            call()
    finally:
        importlib.import_module = real
    graph.replay()
    torch.cuda.synchronize()
    if not all(same_bytes(a, b) for a, b in zip(tensors(outputs), eager)):
        errors.append(f"rank {rank} {tag}: graph replay differs from the eager call")
    del graph
