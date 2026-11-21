# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 through the entries: the first call of each entry warms its routed package up; after that no served batch
compiles (``warm_up`` and ``compile_watch`` count nothing), and a CUDA graph of each entry call captures without an
import or an allocation and replays bit for bit as the eager call."""

import json
import os
import subprocess
import sys
from unittest.mock import patch


def scenario(workers):
    import torch
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.indexer.mxfp4_indexer_common import (
        mxfp4_paged_score_topk,
        mxfp4_prefill_score,
        mxfp4_prefill_topk,
        paged_score_topk_preallocate,
        paged_score_topk_reset_counts,
    )

    from tests.indexer.mxfp4_paged_score_topk_d128_h32_top2048.inputs import make_inputs as paged_inputs
    from tests.indexer.mxfp4_prefill_score_d128_h32.inputs import make_inputs as score_inputs
    from tests.indexer.mxfp4_prefill_topk_d128_h32_top2048.inputs import make_logits, make_windows

    warm_up.set_auto(True)
    warm_up.set_workers(workers)

    def paged(seqs, qlen, width):
        inputs = paged_inputs(seqs, qlen, width, seed=seqs + qlen + width)
        rows = seqs * qlen
        idx = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
        val = torch.full((rows, 2048), 3.0, dtype=torch.float32, device="cuda")
        return lambda: (paged_score_topk_reset_counts(inputs["query_start_loc"], qlen, cache=inputs["cache"],
                                                      out_indices=idx),
                        mxfp4_paged_score_topk(**inputs, out_indices=idx, out_scores=val)), (idx, val)

    def score(rows, keys):
        args = score_inputs(rows, keys, seed=rows + keys)
        out = torch.full((rows, args[3].shape[0] * 64), 7.0, device="cuda")
        return lambda: mxfp4_prefill_score(*args, out, 1.0), (out,)

    def topk(rows, width):
        logits = make_logits("gauss", rows, width, seed=rows)
        ks, ke = make_windows(rows, width, "causal")
        out = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
        return lambda: mxfp4_prefill_topk(logits, ks, ke, out), (out,)

    for make, shape in ((paged, (2, 4, 4096)), (score, (8, 4096)), (topk, (16, 8192))):
        make(*shape)[0]()
    torch.cuda.synchronize()
    warmed = sorted(name for name, _ in warm_up.warmed())
    warned = []
    compile_watch.arm(warned.append)
    walk = [paged(s, q, w) for w in (4096, 32768, 131072) for s, q in ((1, 1), (2, 3), (3, 4), (8, 5), (2, 6), (64, 4))]
    walk += [score(r, k) for r, k in ((1, 64), (5, 700), (64, 9000), (512, 40000))]
    walk += [topk(r, w) for r, w in ((1, 700), (16, 70000), (256, 20000))]
    for call, _ in walk:
        call()
    torch.cuda.synchronize()
    result = dict(warmed=warmed, late=warm_up.late_compiles(), watched=compile_watch.late_compiles(), warned=warned[:3])

    stream = torch.cuda.Stream()

    def captured_allocations(fn, g=None):
        """Allocations made while capturing ``fn`` (the capture's own allocations included)."""
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        with torch.cuda.graph(g or torch.cuda.CUDAGraph(), stream=stream):
            fn()
        return torch.cuda.memory_stats()["allocation.all.allocated"] - before

    paged_score_topk_preallocate("cuda", [stream], cache=paged_inputs(1, 1, 64)["cache"],
                                 out_indices=torch.empty(1, 2048, dtype=torch.int32, device="cuda"))
    graphs = {}
    for name, (call, outs) in (("paged", paged(3, 4, 32768)), ("score", score(64, 9000)), ("topk", topk(65, 20000))):
        init = [o.clone() for o in outs]  # the scorer leaves positions outside the windows' pages as they were
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        torch.cuda.synchronize()
        eager = [o.clone() for o in outs]
        allocs = captured_allocations(lambda: None)
        g = torch.cuda.CUDAGraph()
        with patch("importlib.import_module", side_effect=AssertionError("import inside capture")):
            allocs = captured_allocations(call, g) - allocs
        for o, i in zip(outs, init, strict=True):
            o.copy_(i)
        g.replay()
        torch.cuda.synchronize()
        graphs[name] = dict(equal=all(torch.equal(o, e) for o, e in zip(outs, eager, strict=True)), allocs=allocs)
    result["graphs"] = graphs
    result["late_after_graphs"] = warm_up.late_compiles() + compile_watch.late_compiles()
    return result


def routed_entries():
    import importlib

    from amd_kernels.indexer.mxfp4_indexer_common import paged, prefill

    return [importlib.import_module(m).mxfp4_paged_score_topk for m in paged._SHAPE_HOSTS.values()] + [
        importlib.import_module(m).score_rows for m in prefill._SCORE_HOSTS.values()] + [
        importlib.import_module(m).topk_rows for m in prefill._TOPK_HOSTS.values()]


def test_warm_up_walk_and_graph(tmp_path):
    from tests.indexer.mxfp4_paged_score_topk_d128_h32_top2048.inputs import require_gfx950

    require_gfx950()
    proc = subprocess.run(
        [sys.executable, "-m", __name__, "16"], capture_output=True, text=True, timeout=1800,
        env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1", "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")},
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(result)
    assert result["warmed"] == sorted(e.entry.name for e in routed_entries()), result
    assert result["late"] == 0 and result["watched"] == 0, result
    assert all(g["equal"] and g["allocs"] == 0 for g in result["graphs"].values()), result
    assert result["late_after_graphs"] == 0, result


if __name__ == "__main__":
    print(json.dumps(scenario(int(sys.argv[1]))))
